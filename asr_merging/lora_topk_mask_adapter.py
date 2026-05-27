#!/usr/bin/env python3
"""Create a top-k masked copy of a PEFT LoRA adapter.

This utility keeps only the largest-magnitude non-zero values in each LoRA
matrix (A/B) and zeroes the rest.

It works directly on adapter checkpoint files (no base model load required).

Arguments:
    --adapter-path:
        Path to an adapter directory, or a parent directory containing final_model
        or checkpoint-* folders.
    --output-path:
        Optional destination directory for the pruned adapter. If omitted, the
        default is <resolved_adapter_dir>-topk.
    --keep-ratio:
        Fraction of non-zero values to keep in each LoRA matrix.
    --min-topk:
        Lower bound on kept entries per LoRA matrix when that matrix has at least
        one non-zero element. Effective k is:
            k = min(nnz, max(min_topk, ceil(keep_ratio * nnz)))
        This prevents tiny matrices from being entirely zeroed by very small
        keep-ratio values.
    --dry-run:
        Compute stats without writing output files.

Example:
  python -m asr_merging.lora_topk_mask_adapter \
    --adapter-path voxtral-lora-adapter-serbian \
    --keep-ratio 0.20

  python -m asr_merging.lora_topk_mask_adapter \
    --adapter-path experiments/run/checkpoint-7000 \
    --output-path experiments/run/checkpoint-7000-topk \
    --keep-ratio 0.15 \
    --min-topk 1
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
    from safetensors.torch import save_file as safetensors_save_file
except Exception:  # pragma: no cover
    safetensors_load_file = None
    safetensors_save_file = None


ADAPTER_CONFIG_NAME = "adapter_config.json"
SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
BIN_WEIGHTS_NAME = "adapter_model.bin"


def _resolve_adapter_dir(adapter_path: str) -> Path:
    root = Path(adapter_path)
    if not root.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    if (root / ADAPTER_CONFIG_NAME).exists():
        return root

    final_model = root / "final_model"
    if (final_model / ADAPTER_CONFIG_NAME).exists():
        return final_model

    checkpoints = sorted(
        [x for x in root.glob("checkpoint-*") if x.is_dir() and (x / ADAPTER_CONFIG_NAME).exists()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(
        f"No adapter directory found under: {adapter_path}. "
        f"Expected {ADAPTER_CONFIG_NAME} in path/final_model/checkpoint-*"
    )


def _find_weights_file(adapter_dir: Path) -> Path:
    safe_path = adapter_dir / SAFE_WEIGHTS_NAME
    if safe_path.exists():
        if safetensors_load_file is None or safetensors_save_file is None:
            raise RuntimeError(
                "safetensors is required to process adapter_model.safetensors but is not installed."
            )
        return safe_path

    bin_path = adapter_dir / BIN_WEIGHTS_NAME
    if bin_path.exists():
        return bin_path

    raise FileNotFoundError(
        f"No adapter weights found in {adapter_dir}. Expected {SAFE_WEIGHTS_NAME} or {BIN_WEIGHTS_NAME}."
    )


def _load_state_dict(weights_path: Path) -> Dict[str, torch.Tensor]:
    if weights_path.suffix == ".safetensors":
        return safetensors_load_file(str(weights_path), device="cpu")
    state = torch.load(str(weights_path), map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {weights_path}")
    return state


def _save_state_dict(state: Dict[str, torch.Tensor], output_weights_path: Path) -> None:
    output_weights_path.parent.mkdir(parents=True, exist_ok=True)
    if output_weights_path.suffix == ".safetensors":
        safetensors_save_file(state, str(output_weights_path))
    else:
        torch.save(state, str(output_weights_path))


def _iter_lora_weight_keys(state: Dict[str, torch.Tensor]) -> Iterable[str]:
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if ("lora_A" in key or "lora_B" in key) and key.endswith(".weight"):
            yield key


def _build_topk_mask(tensor: torch.Tensor, keep_ratio: float, min_topk: int) -> Tuple[torch.Tensor, int, int]:
    nonzero_mask = tensor != 0
    nnz = int(nonzero_mask.sum().item())

    if nnz == 0:
        return torch.zeros_like(tensor), 0, 0

    k = max(int(min_topk), int(math.ceil(float(keep_ratio) * nnz)))
    k = min(k, nnz)

    nz_vals = tensor[nonzero_mask].abs()
    if k >= nnz:
        keep_nz_mask = torch.ones_like(nz_vals, dtype=torch.bool)
    else:
        topk_idx = torch.topk(nz_vals, k=k, largest=True, sorted=False).indices
        keep_nz_mask = torch.zeros_like(nz_vals, dtype=torch.bool)
        keep_nz_mask[topk_idx] = True

    binary_mask = torch.zeros_like(tensor)
    binary_mask[nonzero_mask] = keep_nz_mask.to(dtype=tensor.dtype)
    kept = int((binary_mask != 0).sum().item())
    return binary_mask, nnz, kept


def _mask_lora_state(
    state: Dict[str, torch.Tensor],
    keep_ratio: float,
    min_topk: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]], List[Dict]]:
    masked_state: Dict[str, torch.Tensor] = {}
    mask_a: Dict[str, torch.Tensor] = {}
    mask_b: Dict[str, torch.Tensor] = {}
    stats: List[Dict] = []

    lora_keys = set(_iter_lora_weight_keys(state))

    for key, value in state.items():
        if key not in lora_keys:
            masked_state[key] = value
            continue

        tensor = value.clone()
        mask, nnz, kept = _build_topk_mask(tensor, keep_ratio=keep_ratio, min_topk=min_topk)
        tensor.mul_(mask)
        masked_state[key] = tensor

        row = {
            "name": key,
            "total": int(value.numel()),
            "nnz_before": int(nnz),
            "nnz_after": int(kept),
            "kept_ratio_vs_nnz": float((kept / nnz) if nnz > 0 else 0.0),
        }
        stats.append(row)

        if "lora_A" in key:
            mask_a[key] = mask
        elif "lora_B" in key:
            mask_b[key] = mask

    return masked_state, {"mask_A": mask_a, "mask_B": mask_b}, stats


def _copy_adapter_dir(input_dir: Path, output_dir: Path, exclude_names: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude = set(exclude_names)
    for child in input_dir.iterdir():
        if child.name in exclude:
            continue
        dst = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dst)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create top-k masked copy of a LoRA adapter.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "How k is chosen per LoRA matrix:\n"
            "  k = min(nnz, max(min_topk, ceil(keep_ratio * nnz)))\n"
            "\n"
            "Meaning of --min-topk:\n"
            "  It guarantees at least this many non-zero entries survive in each\n"
            "  LoRA matrix with nnz>0. Use it to avoid over-pruning tiny layers.\n"
            "\n"
            "Example:\n"
            "  nnz=3, keep_ratio=0.2, min_topk=1 -> ceil(0.6)=1, k=1\n"
            "  nnz=3, keep_ratio=0.2, min_topk=2 -> ceil(0.6)=1, k=2\n"
        ),
    )
    p.add_argument("--adapter-path", required=True, help="Adapter dir (or parent dir with final_model/checkpoint-*)")
    p.add_argument(
        "--output-path",
        default=None,
        help="Output adapter directory. Defaults to <resolved_adapter_dir>-topk",
    )
    p.add_argument(
        "--keep-ratio",
        type=float,
        default=0.20,
        help=(
            "Fraction of non-zero entries to keep in each LoRA matrix. "
            "Combined with --min-topk to determine final k."
        ),
    )
    p.add_argument(
        "--min-topk",
        type=int,
        default=1,
        help=(
            "Minimum number of non-zero entries to keep per LoRA matrix when nnz>0. "
            "Final k is min(nnz, max(min_topk, ceil(keep_ratio * nnz)))."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute statistics only, do not write output files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    keep_ratio = float(args.keep_ratio)
    if keep_ratio <= 0.0 or keep_ratio > 1.0:
        raise ValueError("--keep-ratio must be in (0, 1].")
    if int(args.min_topk) < 1:
        raise ValueError("--min-topk must be >= 1.")

    source_dir = _resolve_adapter_dir(str(args.adapter_path))
    weights_path = _find_weights_file(source_dir)

    default_output = source_dir.parent / f"{source_dir.name}-topk"
    output_dir = Path(args.output_path) if args.output_path else default_output

    state = _load_state_dict(weights_path)
    masked_state, masks_payload, stats = _mask_lora_state(
        state,
        keep_ratio=keep_ratio,
        min_topk=int(args.min_topk),
    )

    total_a = sum(1 for s in stats if "lora_A" in s["name"])
    total_b = sum(1 for s in stats if "lora_B" in s["name"])

    print(f"Source adapter: {source_dir}")
    print(f"Resolved weights: {weights_path.name}")
    print(f"Keep ratio: {keep_ratio:.2%}")
    print(f"LoRA A matrices masked: {total_a}")
    print(f"LoRA B matrices masked: {total_b}")

    for row in stats[:8]:
        print(
            f"- {row['name']}: nnz {row['nnz_before']} -> {row['nnz_after']} "
            f"(keep {row['kept_ratio_vs_nnz']:.2%} of non-zero)"
        )

    if args.dry_run:
        print("Dry run enabled: no files were written.")
        return

    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("Output directory must be different from source adapter directory.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep config/readme/tokenizer sidecar files if present.
    _copy_adapter_dir(
        source_dir,
        output_dir,
        exclude_names={SAFE_WEIGHTS_NAME, BIN_WEIGHTS_NAME, "topk_masks.pt", "topk_stats.json"},
    )

    output_weights_name = weights_path.name
    output_weights_path = output_dir / output_weights_name
    _save_state_dict(masked_state, output_weights_path)

    torch.save(masks_payload, str(output_dir / "topk_masks.pt"))
    (output_dir / "topk_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Pruned adapter saved to: {output_dir}")
    print(f"Saved masks: {output_dir / 'topk_masks.pt'}")
    print(f"Saved stats: {output_dir / 'topk_stats.json'}")


if __name__ == "__main__":
    main()
