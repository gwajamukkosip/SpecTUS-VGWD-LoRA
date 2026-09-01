#!/usr/bin/env python3
"""Build the lightweight frozen r32 all-attention inference artifact."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models/vgwd_clean_rslora_r32_all_attention_seed123_frozen"
SUMMARY = ROOT / "analysis/rslora_r32_all_attention_validation/summary.json"
CONFIG = ROOT / "configs/finetune_vgwd_clean_rslora_r32_allattn_seed123.yaml"
PREDICT_CONFIG = ROOT / "configs/predict_vgwd_clean_rslora_r32_valid_beam10.yaml"
BASE_MODEL = ROOT / "checkpoints/SpecTUS_pretrained/pytorch_model.bin"
TOKENIZER = ROOT / "tokenizer/tokenizer_mf10M.model"
CHECKPOINTS = {
    7: ROOT / "checkpoints/vgwd_clean_rslora_r32_allattn_seed7/2026-08-30-18_23_42_vgwd_clean_rslora_r32_allattn_seed7/checkpoint-3500",
    123: ROOT / "checkpoints/vgwd_clean_rslora_r32_allattn_seed123/2026-08-30-19_26_05_vgwd_clean_rslora_r32_allattn_seed123/checkpoint-3500",
    2026: ROOT / "checkpoints/vgwd_clean_rslora_r32_allattn_seed2026/2026-08-30-21_18_54_vgwd_clean_rslora_r32_allattn_seed2026/checkpoint-3500",
}
COPY_FILES = (
    "adapter_model.bin",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    if summary.get("select_all_attention_for_locked_holdout") is not True:
        raise RuntimeError("The preregistered all-attention selection gate did not pass")
    if summary.get("checkpoint") != 3500 or summary.get("beam") != 10:
        raise RuntimeError("Unexpected frozen evaluation protocol")

    selected_seed = 123
    selected = CHECKPOINTS[selected_seed]
    for path in [SUMMARY, CONFIG, PREDICT_CONFIG, BASE_MODEL, TOKENIZER, *CHECKPOINTS.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    for checkpoint in CHECKPOINTS.values():
        for name in COPY_FILES:
            if not (checkpoint / name).exists():
                raise FileNotFoundError(checkpoint / name)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        shutil.copy2(selected / name, OUTPUT / name)

    run_metrics = {int(row["seed"]): row for row in summary["run_metrics"] if row["condition"] == "all_attention"}
    manifest = {
        "artifact_name": "vgwd_clean_rslora_r32_all_attention_seed123_frozen",
        "frozen_on": "2026-09-01",
        "status": "validation-selected research candidate; not externally validated",
        "selected_seed": selected_seed,
        "selection_basis": "highest strict Top-1 on the fixed 664-row validation set among the three prespecified all-attention seed runs; no test result was used",
        "base_model": "checkpoints/SpecTUS_pretrained",
        "base_model_pytorch_model_sha256": sha256(BASE_MODEL),
        "tokenizer": "tokenizer/tokenizer_mf10M.model",
        "tokenizer_sha256": sha256(TOKENIZER),
        "adapter": {
            "peft": "rsLoRA",
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
            "checkpoint": 3500,
            "trainable_parameters": 9437184,
        },
        "inference": {"deterministic": True, "num_beams": 10, "num_return_sequences": 10, "max_length": 200},
        "validation": {
            "rows": 664,
            "selected_seed_strict_top1_correct": run_metrics[selected_seed]["strict_top1_correct"],
            "selected_seed_strict_top1_percent": run_metrics[selected_seed]["strict_top1_percent"],
            "selected_seed_strict_top10_correct": run_metrics[selected_seed]["strict_top10_correct"],
            "selected_seed_strict_top10_percent": run_metrics[selected_seed]["strict_top10_percent"],
            "all_seed_metrics": [run_metrics[seed] for seed in sorted(run_metrics)],
        },
        "test_used_for_selection_or_freeze": False,
        "source_checkpoint_hashes": {
            str(seed): sha256(checkpoint / "adapter_model.bin") for seed, checkpoint in CHECKPOINTS.items()
        },
        "artifact_files": {},
        "source_files": {
            "training_config": str(CONFIG.relative_to(ROOT)),
            "prediction_config": str(PREDICT_CONFIG.relative_to(ROOT)),
            "validation_summary": str(SUMMARY.relative_to(ROOT)),
        },
    }
    for name in COPY_FILES:
        path = OUTPUT / name
        manifest["artifact_files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksums = [f"{details['sha256']}  {name}" for name, details in sorted(manifest["artifact_files"].items())]
    (OUTPUT / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"OUTPUT: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
