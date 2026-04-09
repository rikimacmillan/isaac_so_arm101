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
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

import isaac_so_arm101.tasks  # Ensures your custom environments are registered
from isaaclab_tasks.utils import parse_env_cfg

def main():
    print("[INFO]: Loading OpenVLA Model...")
    # Load Model (OpenVLA 7B)
    model_id = "openvla/openvla-7b"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    vla = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        device_map={"": 0},
    )
    
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     model_id, 
    #     torch_dtype=torch.bfloat16, 
    #     low_cpu_mem_usage=True, 
    #     trust_remote_code=True,
    # ).to("cuda:0")

    prompt = "In: What action should the robot take to position the gripper above the palm's crown? \nOut:"

    print("[INFO]: Setting up Isaac Lab Environment...")
    # Parse configuration and create environment using gym.make (Safest way)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment to get initial observation
    obs, _ = env.reset()

    print("[INFO]: Starting Inference Loop!")
    while simulation_app.is_running():
        with torch.inference_mode():
            # 1. Extract image from the sensor
            # We pull this directly from the scene because it is not in the RL observation space
            # Shape is usually [num_envs, height, width, channels]
            raw_image_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0]
            
            # Convert to CPU, numpy, and drop the alpha channel if it's RGBA
            raw_image = raw_image_tensor.cpu().numpy()
            if raw_image.shape[-1] == 4: 
                raw_image = raw_image[:, :, :3]
                
            image_pil = Image.fromarray(raw_image)

            # 2. VLA Inference
            inputs = processor(prompt, image_pil).to("cuda:0", dtype=torch.bfloat16)
            vla_action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

            # 3. Map to Robot Actions
            # Arm: [dx, dy, dz, droll, dpitch, dyaw]
            arm_cmd = torch.tensor(vla_action[:6], device=env.unwrapped.device)
            
            # Gripper: Normalize to your URDF limits
            gripper_val = vla_action[6] * 1.57 
            gripper_cmd = torch.tensor([gripper_val], device=env.unwrapped.device)

            # 4. Step Simulation
            # Concatenate to make shape (7,), then unsqueeze to make it (1, 7) for num_envs=1
            actions = torch.cat([arm_cmd, gripper_cmd], dim=-1).unsqueeze(0)
            obs, rewards, terminations, truncations, extras = env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()