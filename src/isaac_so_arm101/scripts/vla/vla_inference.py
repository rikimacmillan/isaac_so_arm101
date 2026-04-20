# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run VLA Inference in Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher


def _patch_transformers_attention_dispatch() -> None:
    """Patch transformers attention dispatch checks for container compatibility.

    Some Isaac Sim container builds ship with a Torch/Transformers combo where
    model init attempts to read `self._supports_sdpa` but the attribute is not
    present on the base class. That causes an AttributeError when loading
    OpenVLA with `trust_remote_code=True`.

    We take a conservative approach:
      - define the expected `_supports_*` flags if missing
      - override the SDPA dispatch check to always return False

    This forces "eager" attention paths and avoids SDPA/FlashAttention dispatch.
    """

    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    for attr in ("_supports_sdpa", "_supports_flash_attn_2", "_supports_flex_attn"):
        if not hasattr(PreTrainedModel, attr):
            setattr(PreTrainedModel, attr, False)

    def _sdpa_can_dispatch(self, is_init_check: bool = False) -> bool:  # noqa: ARG001
        return False

    PreTrainedModel._sdpa_can_dispatch = _sdpa_can_dispatch  # type: ignore[assignment]

# 1. BOOT SEQUENCE: Parse arguments and launch Isaac Sim first!
parser = argparse.ArgumentParser(description="VLA Inference for Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate, default is 1.")
parser.add_argument("--task", type=str, default="None", help="Name of the task.")

# Prompt / model options
parser.add_argument(
    "--instruction",
    type=str,
    default="reach the target",
    help="High-level instruction describing the desired behavior (used to build the OpenVLA prompt if --prompt is not provided).",
)
parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    help="Optional full prompt to send to OpenVLA. If omitted, uses 'In: {instruction}\\nOut:'.",
)
parser.add_argument(
    "--model_id",
    type=str,
    default="openvla/openvla-7b",
    help="HuggingFace model id (or local path) for OpenVLA base model.",
)
parser.add_argument(
    "--unnorm_key",
    type=str,
    default="bridge_orig",
    help="Un-normalization key passed to OpenVLA predict_action (depends on how the model was trained).",
)

# Debug / control knobs
parser.add_argument(
    "--print_interval",
    type=int,
    default=500,
    help="How often to print debug info (in environment steps).",
)
parser.add_argument(
    "--arm_scale",
    type=float,
    default=1.0,
    help="Multiply the 6D arm command by this factor before env.step() (debugging only).",
)
parser.add_argument(
    "--gripper_scale",
    type=float,
    default=1.0,
    help="Multiply the 1D gripper command by this factor before env.step() (debugging only).",
)
parser.add_argument(
    "--save_debug_images_dir",
    type=str,
    default=None,
    help="If set, save the wrist camera RGB image every --save_debug_images_every steps into this directory.",
)
parser.add_argument(
    "--save_debug_images_every",
    type=int,
    default=500,
    help="Save a debug image every N steps (only if --save_debug_images_dir is set).",
)

parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Optional max number of env steps to run (0 = run until the sim stops).",
)
parser.add_argument(
    "--timing",
    action="store_true",
    default=False,
    help="Print per-iteration timing stats (camera, processor, predict_action, env.step).",
)
parser.add_argument(
    "--timing_warmup",
    type=int,
    default=25,
    help="Number of initial steps to skip for timing averages (warmup).",
)
parser.add_argument(
    "--reset_on_done",
    action="store_true",
    default=False,
    help="If set, call env.reset() when termination/truncation is True (prints reset time).",
)

# Optional LoRA adapter (PEFT)
parser.add_argument("--lora_path", type=str, default=None, help="Optional path (or HF repo id) to a PEFT/LoRA adapter directory to apply on top of the base OpenVLA model.",)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This script requires rendering to generate RGB images.
# Force-enable camera support even in headless mode.
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)

# Clean up the terminal output (as we did previously)
import carb
carb.settings.get_settings().set_string("/log/level", "error")
simulation_app = app_launcher.app

"""REST OF IMPORTS GO HERE (After Omniverse is running)"""
import time
import torch
import numpy as np
import gymnasium as gym
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

import isaac_so_arm101.tasks  # Ensures your custom environments are registered
from isaaclab_tasks.utils import parse_env_cfg

def main():
    print("[INFO]: Loading OpenVLA Model...")
    _patch_transformers_attention_dispatch()
    # Load Model (OpenVLA 7B)
    model_id = args_cli.model_id
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    # Hint transformers to use eager attention to avoid SDPA dispatch.
    try:
        vla = AutoModelForVision2Seq.from_pretrained(
            model_id,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": 0},
        )
    except TypeError:
        vla = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": 0},
        )

    # Apply LoRA adapter if provided.
    if args_cli.lora_path:
        print(f"[INFO]: Loading LoRA adapter: {args_cli.lora_path}")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "LoRA requested but 'peft' is not installed. Install it with: pip install peft"
            ) from exc
        vla = PeftModel.from_pretrained(vla, args_cli.lora_path, is_trainable=False)

    # vla = AutoModelForVision2Seq.from_pretrained(
    #     model_id, 
    #     torch_dtype=torch.bfloat16, 
    #     low_cpu_mem_usage=True, 
    #     trust_remote_code=True,
    # ).to("cuda:0")

    prompt = args_cli.prompt if args_cli.prompt else f"In: {args_cli.instruction}\nOut:"

    print("[INFO]: Setting up Isaac Lab Environment...")
    # Parse configuration and create environment using gym.make (Safest way)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment to get initial observation
    obs, _ = env.reset()

    print("[INFO]: Starting Inference Loop!")
    step_count = 0
    print_interval = int(args_cli.print_interval)

    # Timing accumulators (ms)
    t_cam_ms_sum = 0.0
    t_proc_ms_sum = 0.0
    t_pred_ms_sum = 0.0
    t_step_ms_sum = 0.0
    t_total_ms_sum = 0.0
    timing_samples = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            t0 = time.perf_counter()
            # 1. Extract image from the sensor
            # We pull this directly from the scene because it is not in the RL observation space
            # Shape is usually [num_envs, height, width, channels]
            raw_image_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0]
            
            # Convert to CPU, numpy, and drop the alpha channel if it's RGBA
            raw_image = raw_image_tensor.detach().cpu().numpy()
            if raw_image.shape[-1] == 4: 
                raw_image = raw_image[:, :, :3]

            # Convert to uint8 RGB for PIL/OpenVLA.
            if raw_image.dtype != np.uint8:
                rgb_max = float(np.max(raw_image)) if raw_image.size else 0.0
                if rgb_max <= 1.0:
                    raw_image = (np.clip(raw_image, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    raw_image = np.clip(raw_image, 0.0, 255.0).astype(np.uint8)

            image_pil = Image.fromarray(raw_image)

            t1 = time.perf_counter()

            # Optional: save frames to verify the camera is updating.
            if args_cli.save_debug_images_dir and (step_count % int(args_cli.save_debug_images_every) == 0):
                import os

                os.makedirs(args_cli.save_debug_images_dir, exist_ok=True)
                image_pil.save(os.path.join(args_cli.save_debug_images_dir, f"wrist_{step_count:06d}.png"))

            # 2. VLA Inference
            inputs = processor(prompt, image_pil).to("cuda:0", dtype=torch.bfloat16)
            t2 = time.perf_counter()
            vla_action = vla.predict_action(**inputs, unnorm_key=args_cli.unnorm_key, do_sample=False)
            t3 = time.perf_counter()

            # 3. Map to Robot Actions
            # Arm: [dx, dy, dz, droll, dpitch, dyaw]
            arm_cmd = torch.tensor(vla_action[:6], device=env.unwrapped.device) * float(args_cli.arm_scale)
            
            # Gripper: Normalize to your URDF limits
            gripper_val = float(vla_action[6]) * 1.57 * float(args_cli.gripper_scale)
            gripper_cmd = torch.tensor([gripper_val], device=env.unwrapped.device)

            # 4. Step Simulation
            # Concatenate to make shape (7,), then unsqueeze to make it (1, 7) for num_envs=1
            actions = torch.cat([arm_cmd, gripper_cmd], dim=-1).unsqueeze(0)
            
            if step_count % print_interval == 0:
                print("\n" + "="*40)
                print(f"[DEBUG] Sim Step: {step_count}")
                try:
                    ee_pose_cmd = env.unwrapped.command_manager.get_command("ee_pose")[0].detach().cpu().numpy()
                    print(f"[DEBUG] ee_pose command (7D): {ee_pose_cmd}")
                except Exception:
                    pass
                print(f"[DEBUG] Wrist image mean RGB: {raw_image.mean(axis=(0, 1))}")
                print(f"[DEBUG] Raw VLA Output (EE Command):")
                print(f"X, Y, Z: {vla_action[:3]}")
                print(f"Roll, Pitch, Yaw: {vla_action[3:6]}")
                print(f"Gripper: {vla_action[6]}")
                print(f"[DEBUG] Tensor sent to env.step(): {actions.cpu().numpy()}")
                print("="*40 + "\n")
            # Step zsimulation
            t4 = time.perf_counter()
            obs, rewards, terminations, truncations, extras = env.step(actions)
            t5 = time.perf_counter()

            # Timing (skip warmup)
            if args_cli.timing and step_count >= int(args_cli.timing_warmup):
                t_cam_ms = (t1 - t0) * 1000.0
                t_proc_ms = (t2 - t1) * 1000.0
                t_pred_ms = (t3 - t2) * 1000.0
                t_step_ms = (t5 - t4) * 1000.0
                t_total_ms = (t5 - t0) * 1000.0
                t_cam_ms_sum += t_cam_ms
                t_proc_ms_sum += t_proc_ms
                t_pred_ms_sum += t_pred_ms
                t_step_ms_sum += t_step_ms
                t_total_ms_sum += t_total_ms
                timing_samples += 1
                if step_count % print_interval == 0 and timing_samples > 0:
                    avg_cam = t_cam_ms_sum / timing_samples
                    avg_proc = t_proc_ms_sum / timing_samples
                    avg_pred = t_pred_ms_sum / timing_samples
                    avg_step = t_step_ms_sum / timing_samples
                    avg_total = t_total_ms_sum / timing_samples
                    fps = 1000.0 / avg_total if avg_total > 0 else float("inf")
                    print(
                        f"[TIMING] avg_ms: cam={avg_cam:.2f}  proc={avg_proc:.2f}  "
                        f"predict={avg_pred:.2f}  env_step={avg_step:.2f}  total={avg_total:.2f}  (~{fps:.2f} it/s)"
                    )

            # Episode end diagnostics
            try:
                done = bool(torch.any(terminations).item() or torch.any(truncations).item())
            except Exception:
                done = bool(terminations) or bool(truncations)

            if done:
                try:
                    term0 = bool(terminations[0].item())
                except Exception:
                    term0 = bool(terminations)
                try:
                    trunc0 = bool(truncations[0].item())
                except Exception:
                    trunc0 = bool(truncations)
                print(f"[DONE] step={step_count} termination={term0} truncation={trunc0}")

                if args_cli.reset_on_done:
                    tr0 = time.perf_counter()
                    env.reset()
                    tr1 = time.perf_counter()
                    print(f"[RESET] env.reset() took {(tr1 - tr0) * 1000.0:.2f} ms")

            step_count += 1

            if int(args_cli.max_steps) > 0 and step_count >= int(args_cli.max_steps):
                print(f"[INFO] Reached --max_steps={args_cli.max_steps}; exiting.")
                break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()