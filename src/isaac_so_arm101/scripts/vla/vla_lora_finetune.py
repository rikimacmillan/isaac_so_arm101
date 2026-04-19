"""LoRA fine-tuning for OpenVLA on a simple JSONL dataset.

This script is intentionally *minimal* and does not require RLDS/TFDS.
It fine-tunes OpenVLA via PEFT/LoRA and saves an adapter directory that can be
loaded by `vla_inference.py` using `--lora_path`.

Dataset format (JSONL): one JSON object per line, with keys:
  - "image": path to an RGB image (relative to --image_root or absolute)
  - "instruction": natural-language instruction string
  - "action": list[float] of length --action_dim (default: 7)

Action values are expected to already be normalized to [-1, 1] for each
dimension. This matches OpenVLA's default ActionTokenizer binning.

Example line:
  {"image": "frame_000123.png", "instruction": "reach the red block", "action": [0.1, 0.0, -0.2, 0.0, 0.0, 0.1, 1.0]}

Notes:
  - OpenVLA is loaded with `trust_remote_code=True`.
  - By default, we append exactly one EOS token so the model learns to stop.

"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

IGNORE_INDEX = -100


def _patch_transformers_attention_dispatch() -> None:
    """Patch transformers attention dispatch checks for container compatibility.

    In some Isaac Sim container builds, transformers will attempt to read
    `self._supports_sdpa` during model initialization, but that attribute may
    not exist on the base class. This leads to a hard crash when loading
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

    # Some transformer versions call `_sdpa_can_dispatch` during init and access
    # `_supports_sdpa` internally. Override to avoid touching the attribute.
    def _sdpa_can_dispatch(self, is_init_check: bool = False) -> bool:  # noqa: ARG001
        return False

    PreTrainedModel._sdpa_can_dispatch = _sdpa_can_dispatch  # type: ignore[assignment]


def _lower(s: str) -> str:
    return s.strip().lower()


def build_openvla_prompt(instruction: str, *, vla_path: Union[str, Path]) -> str:
    """Builds a text prompt consistent with OpenVLA's public deploy examples."""
    instruction = _lower(instruction)
    vla_path = str(vla_path)

    if "v01" in vla_path:
        system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return (
            f"{system_prompt} USER: What action should the robot take to {instruction}? "
            "ASSISTANT:"
        )

    return f"In: What action should the robot take to {instruction}?\nOut:"


class DiscreteActionTokenizer:
    """Discretizes continuous actions into OpenVLA-style action tokens.

    OpenVLA maps each action dimension to one token ID in the *tail* of the
    tokenizer vocabulary, using uniform binning into `bins` buckets.

    This is designed to match OpenVLA's default `ActionTokenizer` behavior.
    """

    def __init__(
        self,
        tokenizer,
        *,
        bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ) -> None:
        self._tokenizer = tokenizer
        self._bins = int(bins)
        self._min_action = float(min_action)
        self._max_action = float(max_action)
        self._edges = np.linspace(self._min_action, self._max_action, self._bins)

        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            raise ValueError("Tokenizer must expose a positive `vocab_size`.")
        self.action_token_begin_idx = int(vocab_size - (self._bins + 1))
        self._vocab_size = vocab_size

    def encode_to_token_ids(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float32), self._min_action, self._max_action)
        bucket_idx = np.digitize(action, self._edges)
        bucket_idx = np.clip(bucket_idx, 0, self._bins - 1)
        return (self.action_token_begin_idx + bucket_idx).astype(np.int64)

    def __call__(self, action: np.ndarray) -> str:
        token_ids = self.encode_to_token_ids(action)
        return self._tokenizer.decode(token_ids.tolist())


@dataclass(frozen=True)
class JsonlExample:
    image: str
    instruction: str
    action: List[float]


class JsonlVlaDataset(Dataset):
    def __init__(
        self,
        *,
        jsonl_path: Path,
        image_root: Optional[Path],
        tokenizer,
        image_transform,
        vla_path: Union[str, Path],
        action_tokenizer: DiscreteActionTokenizer,
        action_dim: int = 7,
        predict_stop_token: bool = True,
    ) -> None:
        self._jsonl_path = Path(jsonl_path)
        self._tokenizer = tokenizer
        self._image_transform = image_transform
        self._vla_path = str(vla_path)
        self._action_tokenizer = action_tokenizer
        self._action_dim = int(action_dim)
        self._predict_stop_token = bool(predict_stop_token)

        if image_root is None:
            self._image_root = self._jsonl_path.parent
        else:
            self._image_root = Path(image_root)

        self._examples: List[JsonlExample] = []
        with self._jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {self._jsonl_path}") from exc

                missing = [k for k in ("image", "instruction", "action") if k not in raw]
                if missing:
                    raise ValueError(f"Missing keys {missing} on line {line_no} of {self._jsonl_path}")

                ex = JsonlExample(
                    image=str(raw["image"]),
                    instruction=str(raw["instruction"]),
                    action=list(raw["action"]),
                )
                self._examples.append(ex)

        if not self._examples:
            raise ValueError(f"No usable examples found in {self._jsonl_path}")

        if self._tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define `eos_token_id`.")

        if self._tokenizer.pad_token_id is None:
            # OpenVLA tokenizers typically define this, but make a safe fallback.
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def __len__(self) -> int:  # noqa: D401
        return len(self._examples)

    def _resolve_image_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self._image_root / p

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self._examples[idx]
        image_path = self._resolve_image_path(ex.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to load image referenced by JSONL dataset. "
                f"image={image_path} jsonl={self._jsonl_path} idx={idx}"
            ) from exc

        action = np.asarray(ex.action, dtype=np.float32)
        if action.shape != (self._action_dim,):
            raise ValueError(
                f"Expected action shape ({self._action_dim},) but got {action.shape} for {image_path}"
            )

        if np.any(action < -1.001) or np.any(action > 1.001):
            raise ValueError(
                "Actions must be normalized to [-1, 1] per-dimension. "
                f"Found out-of-range values for {image_path}."
            )

        action_text = self._action_tokenizer(action)
        prompt = build_openvla_prompt(ex.instruction, vla_path=self._vla_path)

        # Tokenize prompt + action tokens
        base_text = f"{prompt}{action_text}"
        tokenized = self._tokenizer(base_text, add_special_tokens=True, return_attention_mask=False)
        input_ids: List[int] = list(tokenized["input_ids"])

        # Ensure exactly one EOS at the end.
        if input_ids[-1] != self._tokenizer.eos_token_id:
            input_ids.append(self._tokenizer.eos_token_id)

        labels = list(input_ids)

        # Keep loss on: action tokens + EOS (stop) token.
        keep = self._action_dim + 1
        if len(labels) < keep:
            raise ValueError(
                "Tokenization produced a sequence shorter than expected. "
                "Check that action tokenization yields one token per action dimension."
            )

        for i in range(0, len(labels) - keep):
            labels[i] = IGNORE_INDEX

        if not self._predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # Pixel values
        pixel_values = self._image_transform(image)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


class PaddedCollatorForActionPrediction:
    def __init__(
        self,
        *,
        model_max_length: int,
        pad_token_id: int,
        padding_side: str = "right",
    ) -> None:
        if padding_side != "right":
            raise ValueError("Only right padding is supported.")
        self._model_max_length = int(model_max_length)
        self._pad_token_id = int(pad_token_id)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [ex["input_ids"] for ex in instances]
        labels = [ex["labels"] for ex in instances]
        pixel_values = [ex["pixel_values"] for ex in instances]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self._pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, : self._model_max_length]
        labels = labels[:, : self._model_max_length]

        attention_mask = input_ids.ne(self._pad_token_id)

        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values_out: Union[torch.Tensor, Dict[str, torch.Tensor]] = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values_out = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}
        else:
            raise TypeError(f"Unsupported pixel_values type: {type(pixel_values[0])}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values_out,
        }


@dataclass
class TrainConfig:
    vla_path: str
    data_jsonl: str
    image_root: Optional[str]
    output_dir: str
    batch_size: int
    grad_accum_steps: int
    max_steps: int
    save_steps: int
    learning_rate: float
    lora_rank: int
    lora_dropout: float
    use_4bit: bool
    mixed_precision: str
    action_dim: int
    predict_stop_token: bool
    seed: int


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(description="LoRA fine-tuning for OpenVLA (JSONL dataset)")
    parser.add_argument("--vla_path", type=str, default="openvla/openvla-7b", help="Base OpenVLA model id/path")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory for relative image paths (default: JSONL directory)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the PEFT adapter (default: runs/vla_lora/<timestamp>)",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1_000, help="Number of optimizer steps")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=False,
        help="Enable QLoRA-style 4-bit base model loading (saves VRAM, can reduce quality)",
    )

    parser.add_argument(
        "--mixed_precision",
        choices=["bf16", "fp16", "none"],
        default="bf16",
        help="Autocast dtype for forward pass",
    )

    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument(
        "--predict_stop_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the EOS token in the supervised loss",
    )

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Fail fast on missing dataset paths (avoids wasting time downloading/loading the model).
    data_jsonl_path = Path(args.data_jsonl)
    if not data_jsonl_path.exists():
        raise FileNotFoundError(
            f"JSONL dataset not found: {data_jsonl_path.resolve()} (cwd={Path.cwd()})\n"
            "Fix: place your dataset inside the mounted repo or pass an absolute path inside the container."
        )
    if args.image_root is not None:
        image_root_path = Path(args.image_root)
        if not image_root_path.exists():
            raise FileNotFoundError(
                f"Image root not found: {image_root_path.resolve()} (cwd={Path.cwd()})\n"
                "Fix: pass the correct --image_root (directory containing the images referenced by the JSONL)."
            )

    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning OpenVLA requires a CUDA-capable GPU.")

    _set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Output directory
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / "vla_lora" / ts
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    train_cfg = TrainConfig(
        vla_path=args.vla_path,
        data_jsonl=args.data_jsonl,
        image_root=args.image_root,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
        mixed_precision=args.mixed_precision,
        action_dim=args.action_dim,
        predict_stop_token=args.predict_stop_token,
        seed=args.seed,
    )
    (output_dir / "train_config.json").write_text(json.dumps(asdict(train_cfg), indent=2), encoding="utf-8")

    device = torch.device("cuda:0")

    # Compatibility patch for some container transformer builds
    _patch_transformers_attention_dispatch()

    # Load processor
    print(f"[INFO] Loading processor: {args.vla_path}")
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)

    # Quantization (optional)
    quant_config = None
    model_dtype = torch.bfloat16
    if args.use_4bit:
        compute_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        model_dtype = compute_dtype
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )

    # Load model
    print(f"[INFO] Loading model: {args.vla_path}")
    model_kwargs = dict(
        torch_dtype=model_dtype,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": 0} if args.use_4bit else None,
    )
    # Hint transformers to use eager attention to avoid SDPA dispatch.
    try:
        model = AutoModelForVision2Seq.from_pretrained(args.vla_path, attn_implementation="eager", **model_kwargs)
    except TypeError:
        model = AutoModelForVision2Seq.from_pretrained(args.vla_path, **model_kwargs)

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(device)

    model.config.use_cache = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    if not hasattr(processor, "image_processor"):
        raise RuntimeError("Processor missing image_processor; cannot build pixel_values.")
    if not hasattr(processor.image_processor, "apply_transform"):
        raise RuntimeError("image_processor missing apply_transform; cannot build pixel_values.")

    action_tokenizer = DiscreteActionTokenizer(processor.tokenizer)

    dataset = JsonlVlaDataset(
        jsonl_path=Path(args.data_jsonl),
        image_root=Path(args.image_root) if args.image_root else None,
        tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        vla_path=args.vla_path,
        action_tokenizer=action_tokenizer,
        action_dim=args.action_dim,
        predict_stop_token=args.predict_stop_token,
    )

    collator = PaddedCollatorForActionPrediction(
        model_max_length=int(processor.tokenizer.model_max_length),
        pad_token_id=int(processor.tokenizer.pad_token_id),
        padding_side="right",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate)

    # Training loop
    amp_dtype: Optional[torch.dtype]
    if args.mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    print(
        "[INFO] Starting fine-tuning: "
        f"steps={args.max_steps}, bs={args.batch_size}, accum={args.grad_accum_steps}, lr={args.learning_rate}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    data_iter = iter(dataloader)
    global_step = 0
    micro_step = 0

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        micro_step += 1

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        pixel_values = batch["pixel_values"]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(device)
            if amp_dtype is not None:
                pixel_values = pixel_values.to(dtype=amp_dtype)
        else:
            pixel_values = {
                k: (v.to(device, dtype=amp_dtype) if amp_dtype is not None else v.to(device))
                for k, v in pixel_values.items()
            }

        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss

        (loss / args.grad_accum_steps).backward()

        if micro_step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 10 == 0:
                print(f"[TRAIN] step={global_step} loss={loss.item():.4f}")

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                print(f"[INFO] Saving adapter checkpoint at step={global_step} -> {output_dir}")
                processor.save_pretrained(output_dir)
                model.save_pretrained(output_dir)

    print(f"[INFO] Training complete. Saving final adapter -> {output_dir}")
    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
