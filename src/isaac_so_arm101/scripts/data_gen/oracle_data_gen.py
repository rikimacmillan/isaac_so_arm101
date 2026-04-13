# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run VLA Inference in Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher

# 1. BOOT SEQUENCE: Parse arguments and launch Isaac Sim first!
parser = argparse.ArgumentParser(description="VLA Inference for Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate, default is 1.")
parser.add_argument("--task", type=str, default="None", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)

# Clean up the terminal output (as we did previously)
import carb
carb.settings.get_settings().set_string("/log/level", "error")
simulation_app = app_launcher.app


"""REST OF IMPORTS GO HERE (After Omniverse is running)"""
import torch
import numpy as np
import gymnasium as gym
from PIL import Image

import isaac_so_arm101.tasks  # Ensures your custom environments are registered
from isaaclab_tasks.utils import parse_env_cfg

# How high above the tree base to aim (targets trunk midpoint, not base)
# Auto-detected at startup from bounding box; this is the fallback default
TRUNK_TARGET_Z_OFFSET = 0.30  # meters

# Max position delta per step — increase if arm moves too slowly
ACTION_CLAMP = 0.05  # meters/step

class PickOracle:
    def __init__(self):
        self.state = 0 # 0: Hover, 1: Descend, 2: Grasp, 3: Lift
        self.wait_counter = 0
        
    def compute_action(self, ee_pos, target_pos):
        """
        Takes a current End-Effector position and Target position.
        Returns a 7D action vector: [X, Y, Z, Roll, Pitch, Yaw, Gripper]
        """
        
        # Default Action
        # Stay still, keep gripper open (-1.0 = open, 1.0 = closed)
        action = np.zeros(7)
        action[6] = -1.0
        
        # Calculate distance to target
        error_x = target_pos[0] - ee_pos[0]
        error_y = target_pos[1] - ee_pos[1]
        error_z = target_pos[2] - ee_pos[2]
        
        # Hover (15cm above the block)
        if self.state == 0:
            action[0:3] = [error_x, error_y, error_z + 0.15]
            
            # Transition condition: If close to hover point, move to descend
            if np.linalg.norm([error_x, error_y]) < 0.02 and abs(error_z + 0.15) < 0.02:
                self.state = 1
                print("[ORACLE]: Transitioning to DESCEND")
         
        # Descend (Drop down to block level) 
        elif self.state == 1:
            action[0:3] = [error_x, error_y, error_z]
            
            # Transition condition: If we are at the block height, move to Grasp
            if abs(error_z) < 0.01:
                self.state = 2
                self.wait_counter = 15 # Wait 15 simulation steps for the grasps to secure
                print("[ORACLE]: Transitioning to GRASP")
        
        # Grasp (Close gripper)
        elif self.state == 2:
            action[6] = 1.0 # Close Command
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.state = 3
                print(f"[ORACLE]: Transitioning to LIFT")
        
        # Lift (Pull the block up)
        elif self.state == 3:
            action[0:3] = [0.0, 0.0, 0.20] # Move Z up
            action[6] = 1.0
            if ee_pos[2] > (target_pos[2] + 0.15):
                print("[ORACLE]: Done. Resetting")
                self.state = 4 # Finish
                
        return action
        
def main():

    print("[INFO]: Setting up Isaac Lab Environment...")
    # Parse configuration and create environment using gym.make (Safest way)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment to get initial observation
    obs, _ = env.reset()
    
    # Initialize Oracle Brain
    oracle = PickOracle()
    
    print("[INFO]: Starting Oracle Data Generation Loop...")
    
    # print("ROBOT BODIES:", env.unwrapped.scene["robot"].body_names)
    # print("CUSTOM ENV DICT:", env.unwrapped.scene["custom_env"].__dict__.keys())
    # print("Current Goal/Command:", env.unwrapped.command_manager.get_command("object_pose"))
    
    print("Available Command Keys: ", env.unwrapped.command_manager._terms.keys())
    print("Available Scene Keys: ", env.unwrapped.scene.keys())
    
    if "custom_env" in env.unwrapped.scene.keys():
        tree_pos, _ = env.unwrapped.scene["custom_env"].get_world_poses()
        print(f"Tree is at: {tree_pos[0].cpu().numpy()}")
        # Auto-detect trunk height from bounding box so we can target the midpoint
        try:
            aabb = env.unwrapped.scene["custom_env"].root_physx_view.get_root_transforms()
            # Fallback: use AABB via prim bounds if available
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            from pxr import UsdGeom
            prim_path = env.unwrapped.scene["custom_env"].prim_paths[0]
            prim = stage.GetPrimAtPath(prim_path)
            bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])
            bbox = bbox_cache.ComputeWorldBound(prim)
            trunk_height = bbox.GetRange().GetMax()[2] - bbox.GetRange().GetMin()[2]
            global TRUNK_TARGET_Z_OFFSET
            TRUNK_TARGET_Z_OFFSET = trunk_height / 2.0
            print(f"[INFO]: Auto-detected trunk height: {trunk_height:.3f}m — targeting midpoint at +{TRUNK_TARGET_Z_OFFSET:.3f}m")
        except Exception as e:
            print(f"[INFO]: Could not auto-detect trunk height ({e}), using default offset {TRUNK_TARGET_Z_OFFSET}m")
    
    # Cache device and gripper index once outside the loop
    sim_device = env.unwrapped.device
    gripper_indices, _ = env.unwrapped.scene["robot"].find_bodies("moving_gripper")
    gripper_idx = gripper_indices[0]

    step = 0
    # Start simulation
    while simulation_app.is_running():
        # Pull the world position for the specific link
        # Shape is (num_envs, num_bodies, 3). Want env 0, for gripper_idx
        ee_position = env.unwrapped.scene["robot"].data.body_pos_w[0, gripper_idx].cpu().numpy()

        # Get block/trunk position using 'custom_env'
        block_pos_w, _ = env.unwrapped.scene["custom_env"].get_world_poses()
        block_position = block_pos_w[0].cpu().numpy()

        # Target trunk midpoint, not the base (Z=0)
        trunk_target = block_position.copy()
        trunk_target[2] += TRUNK_TARGET_Z_OFFSET

        # Ask Oracle for action vector
        optimal_action = oracle.compute_action(ee_position, trunk_target)

        # Clamp position deltas — IK controller uses relative mode, needs small steps
        optimal_action[:3] = np.clip(optimal_action[:3], -ACTION_CLAMP, ACTION_CLAMP)

        # Build float32 tensor directly on sim device (no CPU→GPU transfer)
        # Shape: (1, 7) = (num_envs, action_dim)
        action_tensor = torch.tensor(optimal_action, dtype=torch.float32, device=sim_device).unsqueeze(0)

        # Heartbeat: print every 50 steps
        if step % 50 == 0:
            print(f"[STEP {step:05d}] state={oracle.state} | "
                  f"ee={ee_position.round(3)} | block={block_position.round(3)} | "
                  f"action={action_tensor[0].cpu().numpy().round(3)} | "
                  f"tensor dtype={action_tensor.dtype} device={action_tensor.device}")

        env.step(action_tensor)
        step += 1

        # Reset environment once the task is complete
        if oracle.state == 4:
            print("[INFO]: Resetting environment for next sequence")
            env.reset()
            oracle = PickOracle() # Reset to HOVER
            step = 0
            
if __name__ == "__main__":
    main()

