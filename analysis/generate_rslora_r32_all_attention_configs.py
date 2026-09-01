#!/usr/bin/env python3
"""Generate matched r32 rsLoRA all-attention configs without changing other fields."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 123, 2026)
BASE_TARGETS = """  target_modules:
    - q_proj
    - v_proj
"""
ALL_ATTENTION_TARGETS = """  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - out_proj
"""


def source_path(seed: int) -> Path:
    return ROOT / f"configs/finetune_vgwd_clean_rslora_r32_seed{seed}.yaml"


def target_path(seed: int) -> Path:
    return ROOT / f"configs/finetune_vgwd_clean_rslora_r32_allattn_seed{seed}.yaml"


def render(source: Path) -> str:
    text = source.read_text()
    if text.count(BASE_TARGETS) != 1:
        raise RuntimeError(f"Expected exactly one q/v target block in {source}")
    return text.replace(BASE_TARGETS, ALL_ATTENTION_TARGETS)


def main() -> None:
    for seed in SEEDS:
        source = source_path(seed)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = target_path(seed)
        expected = render(source)
        if target.exists():
            if target.read_text() != expected:
                raise RuntimeError(f"Existing generated config differs: {target}")
            action = "Verified"
        else:
            target.write_text(expected)
            action = "Created"
        print(f"{action} {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
