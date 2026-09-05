#!/usr/bin/env python3
"""Generate paired q/v and all-attention main-model grouped-CV configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "configs/main_grouped_cv"
TRAIN_TEMPLATE = ROOT / "configs/finetune_vgwd_clean_rslora_r32_seed123.yaml"
PREDICT_TEMPLATE = ROOT / "configs/predict_vgwd_clean_rslora_r32_valid_beam10.yaml"
SEED = 123
FOLDS = range(5)
CONDITIONS = {
    "qv": ["q_proj", "v_proj"],
    "allattn": ["q_proj", "k_proj", "v_proj", "out_proj"],
}


def dump_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_base = yaml.safe_load(TRAIN_TEMPLATE.read_text())
    predict_base = yaml.safe_load(PREDICT_TEMPLATE.read_text())
    for fold in FOLDS:
        for condition, targets in CONDITIONS.items():
            train = copy.deepcopy(train_base)
            dataset = train["data_args"]["datasets"]["VGWD"]
            dataset["train_path"] = (
                f"data/vgwd_main_grouped_cv_locked/fold_{fold}/train.jsonl"
            )
            dataset["valid_path"] = (
                f"data/vgwd_main_grouped_cv_locked/fold_{fold}/oof.jsonl"
            )
            dataset["limit_example_split"] = 0
            train["data_args"]["data_seed"] = SEED
            train["lora_args"]["target_modules"] = targets
            args = train["hf_training_args"]
            args.update(
                {
                    "do_train": True,
                    "do_eval": False,
                    "max_steps": 3500,
                    "seed": SEED,
                    "evaluation_strategy": "no",
                    "eval_steps": 3500,
                    "save_strategy": "steps",
                    "save_steps": 500,
                    "save_total_limit": 1,
                    "load_best_model_at_end": False,
                    "report_to": "none",
                }
            )
            dump_yaml(OUTPUT / f"train_fold{fold}_{condition}_seed{SEED}.yaml", train)

            predict = copy.deepcopy(predict_base)
            predict["general"]["additional_naming_info"] = (
                f"main_grouped_cv_fold{fold}_{condition}_seed{SEED}_beam10"
            )
            predict["dataset"].update(
                {
                    "data_path": (
                        f"data/vgwd_main_grouped_cv_locked/fold_{fold}/oof.jsonl"
                    ),
                    "dataset_name": "VGWD_GROUPED_CV",
                    "data_split": f"oof_fold{fold}",
                }
            )
            dump_yaml(
                OUTPUT / f"predict_fold{fold}_{condition}_seed{SEED}_beam10.yaml",
                predict,
            )

    for condition, targets in CONDITIONS.items():
        smoke = copy.deepcopy(train_base)
        dataset = smoke["data_args"]["datasets"]["VGWD"]
        dataset.update(
            {
                "train_path": "data/vgwd_main_grouped_cv_locked/fold_0/train.jsonl",
                "valid_path": "data/vgwd_main_grouped_cv_locked/fold_0/oof.jsonl",
                "limit_train_split": 32,
                "limit_val_split": 8,
                "limit_example_split": 0,
            }
        )
        smoke["data_args"]["data_seed"] = SEED
        smoke["lora_args"]["target_modules"] = targets
        smoke["hf_training_args"].update(
            {
                "do_train": True,
                "do_eval": False,
                "max_steps": 2,
                "seed": SEED,
                "evaluation_strategy": "no",
                "eval_steps": 2,
                "save_strategy": "steps",
                "save_steps": 2,
                "save_total_limit": 1,
                "load_best_model_at_end": False,
                "report_to": "none",
            }
        )
        dump_yaml(OUTPUT / f"smoke_fold0_{condition}_seed{SEED}.yaml", smoke)
    print(f"Generated 22 configs in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
