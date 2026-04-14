# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run VLA Inference in Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments and launch Isaac Sim first.
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
import os
import cv2
import torch
import numpy as np
import gymnasium as gym
from PIL import Image

import isaac_so_arm101.tasks  # Ensures your custom environments are registered
from isaaclab_tasks.utils import parse_env_cfg

# Max position delta per step
ACTION_CLAMP = 0.15  # meters/step

# Hover height above spray target
HOVER_OFFSET_Z = 0.10  # meters — small offset since arm workspace is tight

# How long to hold the spray position (in simulation steps)
SPRAY_DURATION = 60  # steps (~2 seconds at 30fps)

# Domain Randomization Ranges
X_MIN, X_MAX = 0.25, 0.35 # Distance from the trunk
Y_MIN, Y_MAX = -0.10, 0.10 # Climber twist/side to side margin
Z_MIN, Z_MAX = 4.50, 4.80 # Climber height margin above the crown

def get_random_target():
    """Generates a randomized spray target within the defined tolerance volume."""
    return np.array([
        np.random.uniform(X_MIN, X_MAX),
        np.random.uniform(Y_MIN, Y_MAX),
        np.random.uniform(Z_MIN, Z_MAX)
    ])

"""
Non-randomized spray target value (place holder)
# Spray target: push slightly toward palm (X=0.56) and a bit higher into the canopy
# Arm natural hover is ~[0.5, 0.03, 5.07], so this is a small reachable adjustment
# SPRAY_TARGET = np.array([0.52, 0.03, 5.15])
"""

class SprayOracle:
    """
    Spray/retract oracle. States:
      0 - Approach : fly EE to hover point above canopy
      1 - Descend  : lower EE to spray position
      2 - Spray    : hold position, spray signal active
      3 - Retract  : pull back up to hover height
      4 - Done

    Each movement state advances when either:
      - EE is within POSITION_THRESHOLD of the target, OR
      - The state has been active for MAX_STATE_STEPS (arm is visually close enough)
    """
    POSITION_THRESHOLD = 0.08  # meters
    MAX_STATE_STEPS = 200      # advance after this many steps regardless

    def __init__(self):
        self.state = 0
        self.spray_counter = 0
        self.state_steps = 0

    def _advance(self, next_state, msg):
        print(f"[ORACLE]: {msg} (after {self.state_steps} steps)")
        self.state = next_state
        self.state_steps = 0

    def compute_action(self, ee_pos, spray_target):
        """
        Returns a 7D action: [dX, dY, dZ, dRoll, dPitch, dYaw, spray]
        spray: 1.0 = spraying, -1.0 = off
        """
        action = np.zeros(7)
        action[6] = -1.0  # spray off by default

        # Calculate hover position directly above the randomized target
        hover_pos = spray_target.copy()
        hover_pos[2] += HOVER_OFFSET_Z
    
        # Calculate Delta Action Vectors (Relative movement for VLA)
        err_to_hover = hover_pos - ee_pos
        err_to_target = spray_target - ee_pos
        self.state_steps += 1

        # State 0: Approach hover point above canopy
        if self.state == 0:
            action[0:3] = err_to_hover
            if np.linalg.norm(err_to_hover) < self.POSITION_THRESHOLD or self.state_steps >= self.MAX_STATE_STEPS:
                self._advance(1, "Hover reached — DESCENDING to spray position")

        # State 1: Descend to spray position
        elif self.state == 1:
            action[0:3] = err_to_target
            if np.linalg.norm(err_to_target) < self.POSITION_THRESHOLD or self.state_steps >= self.MAX_STATE_STEPS:
                self.spray_counter = SPRAY_DURATION
                self._advance(2, "At spray position — SPRAYING")

        # State 2: Hold position and spray
        elif self.state == 2:
            action[0:3] = err_to_target  # hold position
            action[6] = 1.0              # spray on
            self.spray_counter -= 1
            if self.spray_counter <= 0:
                self._advance(3, "Spray done — RETRACTING")

        # State 3: Retract back to hover height
        elif self.state == 3:
            action[0:3] = err_to_hover
            if np.linalg.norm(err_to_hover) < self.POSITION_THRESHOLD or self.state_steps >= self.MAX_STATE_STEPS:
                self._advance(4, "Retracted. Done.")

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
    
    # Create dataset directory
    dataset_dir = "vla_tree_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    img_dir = os.path.join(dataset_dir, "images")
    act_dir = os.path.join(dataset_dir, "actions")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(act_dir, exist_ok=True)
    print(f"[INFO]: Saving images to ./{img_dir}/ and actions to ./{act_dir}/")
    
    # Initialize Oracle Brain
    oracle = SprayOracle()
    current_spray_target = get_random_target()
    
    print("[INFO]: Starting Oracle Data Generation Loop...")
    
    # print("ROBOT BODIES:", env.unwrapped.scene["robot"].body_names)
    # print("CUSTOM ENV DICT:", env.unwrapped.scene["custom_env"].__dict__.keys())
    # print("Current Goal/Command:", env.unwrapped.command_manager.get_command("object_pose"))
    
    """ 
    Debug: Find available keys 
    print("Available Command Keys: ", env.unwrapped.command_manager._terms.keys())
    print("Available Scene Keys: ", env.unwrapped.scene.keys())
    
    if "custom_env" in env.unwrapped.scene.keys():
        tree_pos, _ = env.unwrapped.scene["custom_env"].get_world_poses()
        print(f"[INFO]: Background scene root at: {tree_pos[0].cpu().numpy()}")

    print(f"[INFO]: Spray target set to {SPRAY_TARGET} (test placeholder — update once canopy position is known)")
    # Debug: scan USD stage for prims that look like trees/plants and print their bounds
    try:
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])
        print("[INFO]: Scanning scene for tall prims (likely tree canopy candidates):")
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            # Only look at prims inside the custom_env scene, skip small/non-mesh prims
            if "Scene" not in path:
                continue
            if not prim.IsA(UsdGeom.Xform) and not prim.IsA(UsdGeom.Mesh):
                continue
            try:
                bbox = bbox_cache.ComputeWorldBound(prim)
                rng = bbox.GetRange()
                height = rng.GetMax()[2] - rng.GetMin()[2]
                if height > 1.0:  # only prims taller than 1m
                    center = (rng.GetMin() + rng.GetMax()) / 2
                    top = rng.GetMax()[2]
                    print(f"  {path} | height={height:.2f}m | center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) | top_z={top:.2f}")
            except Exception:
                pass
    except Exception as e:
        print(f"[INFO]: USD stage scan skipped ({e})")
    """
    # Cache device and gripper index once outside the loop
    sim_device = env.unwrapped.device
    gripper_indices, _ = env.unwrapped.scene["robot"].find_bodies("moving_gripper")
    gripper_idx = gripper_indices[0]

    step = 0
    global_record_step = 0 # Counter for dataset filenames
    # Start simulation
    while simulation_app.is_running():
        # Pull the world position for the specific link
        # Shape is (num_envs, num_bodies, 3). Want env 0, for gripper_idx
        ee_position = env.unwrapped.scene["robot"].data.body_pos_w[0, gripper_idx].cpu().numpy()

        # Ask Oracle for action vector (fixed test target for now)
        optimal_action = oracle.compute_action(ee_position, current_spray_target)

        # Clamp position deltas — IK controller uses relative mode, needs small steps
        optimal_action[:3] = np.clip(optimal_action[:3], -ACTION_CLAMP, ACTION_CLAMP)

        # Build float32 tensor directly on sim device (no CPU→GPU transfer)
        # Shape: (1, 7) = (num_envs, action_dim)
        action_tensor = torch.tensor(optimal_action, dtype=torch.float32, device=sim_device).unsqueeze(0)

        # Data Recording Engine
        if oracle.state < 4:
        
            # Grab raw RGB tensor from  camera and convert to Numpy
            img_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0]
            img_numpy = img_tensor.cpu().numpy()
            
            # Convert RGB to BGR for OpenCV
            if img_numpy.shape[-1] == 4:
                img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
                
            # Format filenames (I.E. frame_00142.jpg)
            step_str = str(global_record_step).zfill(5)
            
            # Save visual and numeric state to disk
            cv2.imwrite(os.path.join(img_dir, f"frame_{step_str}.jpg"), img_bgr)
            np.save(os.path.join(act_dir, f"action_{step_str}.npy"), optimal_action)
            global_record_step += 1
        
        # Heartbeat: print every 50 steps
        if step % 50 == 0:
            print(f"[STEP {step:05d}] state={oracle.state} | "
                  f"ee={ee_position.round(3)} | target={current_spray_target.round(3)} | "
                  f"action={action_tensor[0].cpu().numpy().round(3)} | "
                  f"tensor dtype={action_tensor.dtype} device={action_tensor.device}")

        env.step(action_tensor)
        step += 1

        # Reset environment once the spray cycle is complete
        if oracle.state == 4:
            print("[INFO]: Resetting environment for next sequence")
            env.reset()
            oracle = SprayOracle()
            current_spray_target = get_random_target()
            step = 0
            
if __name__ == "__main__":
    main()

