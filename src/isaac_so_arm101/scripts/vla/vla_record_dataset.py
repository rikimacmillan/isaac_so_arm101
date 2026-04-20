# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Record a simple JSONL dataset (images + instruction + action) from Isaac Lab.

This is a lightweight helper to generate the on-disk dataset that
`vla_lora_finetune.py` expects:

  data/vla_train/
    dataset.jsonl
    images/
      frame_000000.png
      ...

Each JSONL line contains:
  - image: relative path (resolved relative to --image_root)
  - instruction: a string prompt
  - action: list[float] (typically normalized to [-1, 1])

Important:
  - This script does *not* magically create good demonstrations.
    You must choose a policy that produces meaningful actions.
        The built-in `random`/`zero` policies are only for pipeline smoke-tests.
        The `openvla` policy records actions predicted by OpenVLA (optionally + LoRA).

Run (inside the Isaac Sim container):
  python src/isaac_so_arm101/scripts/vla/vla_record_dataset.py \
        --task Isaac-PING-TI-VLA-v0 \
    --num_steps 2000 \
    --instruction "reach the target" \
        --policy random \
        --headless

Then fine-tune:
  python src/isaac_so_arm101/scripts/vla/vla_lora_finetune.py \
    --data_jsonl data/vla_train/dataset.jsonl \
    --image_root data/vla_train/images ...
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# 1) Boot sequence: parse args + launch Isaac Sim first.
parser = argparse.ArgumentParser(description="Record a VLA JSONL dataset from Isaac Lab")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (recommended: 1).")
parser.add_argument("--task", type=str, required=True, help="Gym task name (must spawn a wrist_camera).")

parser.add_argument("--num_steps", type=int, default=2000, help="Number of environment steps to record.")
parser.add_argument(
    "--instruction",
    type=str,
    default="do the task",
    help="Instruction string to store for every recorded sample.",
)
parser.add_argument(
    "--policy",
    choices=["random", "zero", "openvla"],
    default="random",
    help="Action source. NOTE: random/zero are only useful for smoke-testing; openvla queries a model.",
)

# OpenVLA policy options (only used when --policy openvla)
parser.add_argument(
    "--openvla_model_id",
    type=str,
    default="openvla/openvla-7b",
    help="HuggingFace model id (or local path) for OpenVLA base model.",
)
parser.add_argument(
    "--openvla_lora_path",
    type=str,
    default=None,
    help="Optional PEFT/LoRA adapter directory (or HF repo id) to apply on top of the base model.",
)
parser.add_argument(
    "--openvla_unnorm_key",
    type=str,
    default="bridge_orig",
    help="Un-normalization key passed to OpenVLA predict_action (depends on how the model was trained).",
)
parser.add_argument(
    "--openvla_prompt",
    type=str,
    default=None,
    help="Optional prompt template to send to OpenVLA. If omitted, uses 'In: {instruction}\\nOut:'.",
)

parser.add_argument(
    "--out_dir",
    type=str,
    default="data/vla_train",
    help="Output directory (will create dataset.jsonl and images/ under this).",
)
parser.add_argument(
    "--append",
    action="store_true",
    default=False,
    help="Append to an existing dataset.jsonl instead of overwriting.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This script requires rendering to generate RGB images.
# Force-enable camera support even in headless mode.
args_cli.enable_cameras = True

# Vulkan driver workaround (Windows/WSL + Docker):
# Some setups set `VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json`, but only
# mount CUDA/ML driver libs (no `libGLX_nvidia.so.0`). In that case, Vulkan
# initialization fails and Isaac Sim cannot start. If Mesa's lavapipe ICD is
# available, fall back to it so cameras can still render (software).
vk_driver_files = os.environ.get("VK_DRIVER_FILES", "")
if vk_driver_files.endswith("/etc/vulkan/icd.d/nvidia_icd.json"):
    glx_nvidia = Path("/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0")
    lavapipe_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
    if (not glx_nvidia.exists()) and lavapipe_icd.exists():
        print(
            "[WARN] Vulkan is forced to NVIDIA ICD but NVIDIA GLX library is missing. "
            "Falling back to Mesa lavapipe (software Vulkan). This will be slow but should run."
        )
        os.environ["VK_DRIVER_FILES"] = str(lavapipe_icd)

# Helpful preflight hints for common container misconfiguration.
caps = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "")
if caps and ("graphics" not in caps) and (caps != "all"):
    print(
        "[WARN] NVIDIA_DRIVER_CAPABILITIES does not include 'graphics'. "
        "Isaac Sim cameras may fail to initialize (Vulkan/GL errors). "
        f"Current value: {caps!r}"
    )
if os.name == "posix":
    icd_dir = Path("/usr/share/vulkan/icd.d")
    if icd_dir.exists() and not any(p.name.startswith("nvidia") for p in icd_dir.glob("*.json")):
        print(
            "[WARN] No NVIDIA Vulkan ICD JSON found under /usr/share/vulkan/icd.d. "
            "If you see VkResult ERROR_INCOMPATIBLE_DRIVER, restart the container with "
            "-e NVIDIA_DRIVER_CAPABILITIES=all (or include 'graphics')."
        )

# launch omniverse app
app_launcher = AppLauncher(args_cli)

# Reduce terminal spam.
import carb

carb.settings.get_settings().set_string("/log/level", "error")

simulation_app = app_launcher.app

"""Rest of imports go here (after Omniverse is running)."""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


_STOP_REQUESTED = False


def _request_stop(_signum: int, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _make_action(policy: str, action_shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if policy == "zero":
        return torch.zeros(action_shape, device=device)
    if policy == "random":
        return 2 * torch.rand(action_shape, device=device) - 1
    raise ValueError(f"Unsupported policy: {policy}")


def _patch_transformers_attention_dispatch() -> None:
    """Patch transformers attention dispatch checks for container compatibility.

    Some Isaac Sim container builds ship with a Torch/Transformers combo where
    model init attempts to read `self._supports_sdpa` but the attribute is not
    present on the base class. That causes an AttributeError when loading
    OpenVLA with `trust_remote_code=True`.
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


def _load_openvla(model_id: str, lora_path: str | None):
    """Load OpenVLA + processor (optionally apply LoRA)."""

    _patch_transformers_attention_dispatch()
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

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

    if lora_path:
        from peft import PeftModel

        vla = PeftModel.from_pretrained(vla, lora_path, is_trainable=False)

    return processor, vla


def main() -> None:
    # Make Ctrl+C stop recording gracefully.
    # (Signals are process-wide; safe to register once per invocation.)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    out_dir = Path(args_cli.out_dir)
    image_dir = out_dir / "images"
    jsonl_path = out_dir / "dataset.jsonl"

    image_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if jsonl_path.exists() and not args_cli.append:
        # Safer default: require explicit append to avoid accidental mixing.
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {jsonl_path}. "
            "Pass --append to add more lines or delete the file first."
        )

    mode = "a" if args_cli.append else "w"

    # Create environment configuration and env.
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # NOTE: `InteractiveScene` does not implement Python container membership semantics,
    # so `'wrist_camera' in scene` may call `scene[0]` and raise a confusing KeyError.
    try:
        _ = env.unwrapped.scene["wrist_camera"]
    except KeyError as exc:
        raise KeyError(
            "This task does not define a 'wrist_camera' in env.unwrapped.scene. "
            "Use a VLA-enabled task config (e.g. ReachVlaEnvCfg) or add a camera to the scene."
        ) from exc

    try:
        print(f"[INFO] Recording dataset -> {out_dir}")
        print(f"[INFO] JSONL: {jsonl_path} (mode={mode})")
        print(f"[INFO] Images: {image_dir}")
        print(f"[INFO] Task: {args_cli.task}  num_envs={args_cli.num_envs}")
        print(f"[INFO] Policy: {args_cli.policy}")
        print("[INFO] Stop early: press Ctrl+C once; if hung, `docker stop isaac_sim` from host.")

        env.reset()

        # Use env's action_space shape; for vectorized envs this is usually (num_envs, action_dim).
        action_shape = env.action_space.shape
        device = env.unwrapped.device

        # Load OpenVLA if requested.
        openvla = None
        openvla_prompt = None
        if args_cli.policy == "openvla":
            if int(args_cli.num_envs) != 1:
                raise ValueError("--policy openvla currently requires --num_envs 1 (one camera image -> one action).")
            print(f"[INFO] Loading OpenVLA policy model: {args_cli.openvla_model_id}")
            if args_cli.openvla_lora_path:
                print(f"[INFO] Applying OpenVLA LoRA adapter: {args_cli.openvla_lora_path}")
            openvla = _load_openvla(args_cli.openvla_model_id, args_cli.openvla_lora_path)
            if args_cli.openvla_prompt:
                openvla_prompt = args_cli.openvla_prompt
            else:
                openvla_prompt = f"In: {args_cli.instruction}\nOut:"

        # Frame counter should not collide when appending.
        frame_idx = int(time.time() * 1000)  # ms timestamp base

        with jsonl_path.open(mode, encoding="utf-8") as f:
            for step in range(int(args_cli.num_steps)):
                if _STOP_REQUESTED:
                    print("[INFO] Stop requested; shutting down...")
                    break
                if not simulation_app.is_running():
                    break

                with torch.inference_mode():
                    # Grab wrist camera RGB for env 0.
                    raw = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0]
                    rgb = raw.detach().cpu().numpy()
                    if rgb.shape[-1] == 4:
                        rgb = rgb[:, :, :3]

                    # Convert to PIL and save.
                    rgb_u8 = rgb
                    if rgb_u8.dtype != np.uint8:
                        rgb_max = float(np.max(rgb_u8)) if rgb_u8.size else 0.0
                        if rgb_max <= 1.0:
                            rgb_u8 = (np.clip(rgb_u8, 0.0, 1.0) * 255.0).astype(np.uint8)
                        else:
                            rgb_u8 = np.clip(rgb_u8, 0.0, 255.0).astype(np.uint8)

                    img = Image.fromarray(rgb_u8, mode="RGB")
                    rel_name = f"frame_{frame_idx:012d}.png"
                    img_path = image_dir / rel_name
                    img.save(img_path)

                    # Build the action.
                    if args_cli.policy in ("random", "zero"):
                        actions = _make_action(args_cli.policy, action_shape=action_shape, device=device)
                    elif args_cli.policy == "openvla":
                        assert openvla is not None
                        assert openvla_prompt is not None
                        processor, vla = openvla

                        inputs = processor(openvla_prompt, img).to("cuda:0", dtype=torch.bfloat16)
                        vla_action = vla.predict_action(
                            **inputs, unnorm_key=args_cli.openvla_unnorm_key, do_sample=False
                        )

                        # Ensure float tensor in expected shape/range.
                        a = torch.as_tensor(vla_action, dtype=torch.float32, device=device)
                        if a.numel() != action_shape[-1]:
                            raise ValueError(
                                f"OpenVLA produced action with {a.numel()} dims, but env expects action_dim={action_shape[-1]}."
                            )
                        a = torch.clamp(a, -1.0, 1.0)
                        actions = a.unsqueeze(0) if len(action_shape) == 2 else a
                    else:
                        raise ValueError(f"Unsupported policy: {args_cli.policy}")

                    # Record the action for env 0.
                    if actions.ndim == 1:
                        a0 = actions.detach().cpu().to(torch.float32).tolist()
                    else:
                        a0 = actions[0].detach().cpu().to(torch.float32).tolist()

                    f.write(
                        json.dumps(
                            {
                                "image": rel_name,
                                "instruction": args_cli.instruction,
                                "action": a0,
                            }
                        )
                        + "\n"
                    )

                    # Step simulation.
                    env.step(actions)

                    if step % 100 == 0:
                        print(f"[REC] step={step} wrote={rel_name}")

                    frame_idx += 1
    finally:
        env.close()
        print(f"[INFO] Done. Wrote dataset at: {jsonl_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Fallback: some environments still raise KeyboardInterrupt instead of delivering SIGINT.
        print("[INFO] KeyboardInterrupt; shutting down...")
    finally:
        simulation_app.close()
