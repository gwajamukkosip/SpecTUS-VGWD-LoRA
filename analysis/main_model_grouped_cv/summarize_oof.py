#!/usr/bin/env python3
"""Summarize completed paired OOF folds without touching the locked holdout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/vgwd_main_grouped_cv_locked"
PRED_ROOT = ROOT / "predictions/vgwd_main_grouped_cv"
OUTPUT = ROOT / "analysis/main_model_grouped_cv/results"
CONDITIONS = ("qv", "allattn")
N_FOLDS = 5
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 20260904


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_complete_run(fold: int, condition: str) -> Path | None:
    expected = sum(1 for _ in (DATA_ROOT / f"fold_{fold}/oof.jsonl").open())
    candidates = []
    root = PRED_ROOT / f"fold{fold}/{condition}"
    if not root.exists():
        return None
    for path in root.glob("**/predictions.jsonl"):
        if sum(1 for _ in path.open()) != expected:
            continue
        if not (path.parent / "strict_exact_topk.json").is_file():
            continue
        if not (path.parent / "strict_exact_topk_samples.csv").is_file():
            continue
        candidates.append(path.parent)
    if len(candidates) > 1:
        raise RuntimeError(
            f"More than one complete prediction for fold={fold}, condition={condition}"
        )
    return candidates[0] if candidates else None


def fold_group_hashes(fold: int) -> list[str]:
    assignments = read_csv(DATA_ROOT / "assignments.csv")
    groups = [
        row["group_sha256"]
        for row in assignments
        if row["partition"] == "development" and int(row["oof_fold"]) == fold
    ]
    expected = sum(1 for _ in (DATA_ROOT / f"fold_{fold}/oof.jsonl").open())
    if len(groups) != expected:
        raise RuntimeError(f"Assignment length mismatch for fold {fold}")
    return groups


def topk_hits(samples: list[dict[str, str]], k: int) -> np.ndarray:
    values = []
    for row in samples:
        rank = int(row["exact_rank"]) if row["exact_rank"] else None
        values.append(rank is not None and rank <= k)
    return np.asarray(values, dtype=bool)


def cluster_bootstrap(
    baseline: np.ndarray, candidate: np.ndarray, groups: list[str]
) -> tuple[float, float]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    keys = sorted(by_group)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    effects = np.empty(BOOTSTRAPS, dtype=float)
    for draw in range(BOOTSTRAPS):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = [index for group in sampled for index in by_group[group]]
        effects[draw] = candidate[indices].mean() - baseline[indices].mean()
    low, high = np.quantile(effects, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    lock = json.loads((DATA_ROOT / "LOCK.json").read_text())
    if lock["evaluation_state"] != "LOCKED_NOT_EVALUATED":
        raise RuntimeError("Unexpected holdout lock state")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    pooled_groups: list[str] = []

    completed_pairs = []
    for fold in range(N_FOLDS):
        runs = {condition: find_complete_run(fold, condition) for condition in CONDITIONS}
        samples: dict[str, list[dict[str, str]]] = {}
        for condition, run in runs.items():
            if run is None:
                continue
            summary = json.loads((run / "strict_exact_topk.json").read_text())
            sample_rows = read_csv(run / "strict_exact_topk_samples.csv")
            samples[condition] = sample_rows
            run_rows.append(
                {
                    "fold": fold,
                    "condition": condition,
                    "n": summary["num_samples"],
                    "top1": summary["accuracy_at_k"]["1"],
                    "top3": summary["accuracy_at_k"]["3"],
                    "top5": summary["accuracy_at_k"]["5"],
                    "top10": summary["accuracy_at_k"]["10"],
                    "invalid_top10": summary["invalid_prediction_count_in_top_k"],
                    "prediction_dir": run.relative_to(ROOT).as_posix(),
                }
            )
        if set(samples) != set(CONDITIONS):
            continue
        completed_pairs.append(fold)
        groups = fold_group_hashes(fold)
        qv = topk_hits(samples["qv"], 1)
        allattn = topk_hits(samples["allattn"], 1)
        if len(qv) != len(allattn) or len(qv) != len(groups):
            raise RuntimeError(f"Paired length mismatch for fold {fold}")
        promoted = int(np.sum(~qv & allattn))
        regressed = int(np.sum(qv & ~allattn))
        ci_low, ci_high = cluster_bootstrap(qv, allattn, groups)
        paired_rows.append(
            {
                "fold": fold,
                "n": len(qv),
                "qv_top1": float(qv.mean()),
                "allattn_top1": float(allattn.mean()),
                "gain_pp": float((allattn.mean() - qv.mean()) * 100),
                "cluster_bootstrap_ci_low_pp": ci_low * 100,
                "cluster_bootstrap_ci_high_pp": ci_high * 100,
                "qv_wrong_allattn_right": promoted,
                "qv_right_allattn_wrong": regressed,
                "mcnemar_exact_p": float(
                    binomtest(
                        min(promoted, regressed),
                        promoted + regressed,
                        0.5,
                        alternative="two-sided",
                    ).pvalue
                )
                if promoted + regressed
                else 1.0,
            }
        )
        pooled["qv"].append(qv)
        pooled["allattn"].append(allattn)
        pooled_groups.extend(groups)

    if not args.allow_partial and len(completed_pairs) != N_FOLDS:
        raise RuntimeError(
            f"Only {len(completed_pairs)}/5 paired folds are complete; use --allow-partial"
        )
    pooled_result: dict[str, Any] | None = None
    if completed_pairs:
        qv_all = np.concatenate(pooled["qv"])
        allattn_all = np.concatenate(pooled["allattn"])
        promoted = int(np.sum(~qv_all & allattn_all))
        regressed = int(np.sum(qv_all & ~allattn_all))
        ci_low, ci_high = cluster_bootstrap(qv_all, allattn_all, pooled_groups)
        pooled_result = {
            "scope": "complete_5fold_oof" if len(completed_pairs) == 5 else "partial_oof",
            "completed_folds": completed_pairs,
            "n": len(qv_all),
            "qv_top1": float(qv_all.mean()),
            "allattn_top1": float(allattn_all.mean()),
            "gain_pp": float((allattn_all.mean() - qv_all.mean()) * 100),
            "cluster_bootstrap_95ci_pp": [ci_low * 100, ci_high * 100],
            "qv_wrong_allattn_right": promoted,
            "qv_right_allattn_wrong": regressed,
            "mcnemar_exact_p": float(
                binomtest(
                    min(promoted, regressed),
                    promoted + regressed,
                    0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if promoted + regressed
            else 1.0,
        }

    write_csv(
        OUTPUT / "run_metrics.csv",
        run_rows,
        [
            "fold",
            "condition",
            "n",
            "top1",
            "top3",
            "top5",
            "top10",
            "invalid_top10",
            "prediction_dir",
        ],
    )
    write_csv(
        OUTPUT / "paired_fold_results.csv",
        paired_rows,
        [
            "fold",
            "n",
            "qv_top1",
            "allattn_top1",
            "gain_pp",
            "cluster_bootstrap_ci_low_pp",
            "cluster_bootstrap_ci_high_pp",
            "qv_wrong_allattn_right",
            "qv_right_allattn_wrong",
            "mcnemar_exact_p",
        ],
    )
    result = {
        "protocol": "paired main-model structure-grouped OOF; locked holdout untouched",
        "completed_paired_folds": completed_pairs,
        "required_folds": N_FOLDS,
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pooled": pooled_result,
        "holdout_state": lock["evaluation_state"],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = [
        "# Main-model grouped-CV OOF summary",
        "",
        f"- Completed paired folds: {completed_pairs} ({len(completed_pairs)}/5)",
        f"- Holdout state: `{lock['evaluation_state']}`",
        "- Partial-fold values are operational checks, not the final 5-fold conclusion.",
        "",
        "```json",
        json.dumps(pooled_result, indent=2),
        "```",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
