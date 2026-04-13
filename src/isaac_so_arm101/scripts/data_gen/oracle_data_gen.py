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
    
    # Start simulation
    while simulation_app.is_running():
        # Get exact coordinates from physics engine
        
        # Find where 'moving_gripper' is in the robot's body list
        # Get the first index of find_bodies
        gripper_indices, _ = env.unwrapped.scene["robot"].find_bodies("moving_gripper")
        gripper_idx = gripper_indices[0]
        
        # Pull the world position for the specific link 
        # Shape is (num_envs, num_bodies, 3). Want env 0, for gripper_idx
        ee_position = env.unwrapped.scene["robot"].data.body_pos_w[0, gripper_idx].cpu().numpy()
        
        # Get block position using 'custom_env'
        block_position = env.unwrapped.scene["custom_env"].data.root_pos_w[0].cpu().numpy()
        
        # Ask Oracle for action vector
        optimal_action = oracle.compute_action(ee_position, block_position)
        
        # Step the physics environment forward using that action
        env.step(optimal_action)
        
        # Reset environment once the task is complete
        if oracle.state == 4:
            print("[INFO]: Resetting environment for next sequence")
            env.reset()
            oracle = PickOracle() # Reset to HOVER
            
if __name__ == "__main__":
    main()

