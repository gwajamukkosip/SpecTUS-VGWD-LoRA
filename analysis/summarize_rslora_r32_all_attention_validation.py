#!/usr/bin/env python3
"""Summarize the preregistered r32 all-attention validation-only screen."""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import binomtest, t as student_t


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/vgwd_clean/valid_filtered_mz500.jsonl"
SEEDS = (7, 123, 2026)
RUNS = {
    "qv": "predictions/vgwd_clean_rslora_r32_seed{seed}_valid_beam10",
    "all_attention": "predictions/vgwd_clean_rslora_r32_allattn_seed{seed}_valid_beam10",
}
EXPECTED_ROWS = 664


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None


def prediction_file(pattern: str, seed: int) -> Path:
    root = ROOT / pattern.format(seed=seed)
    if "test" in str(root).lower():
        raise RuntimeError(f"Validation-only guard rejected path: {root}")
    matches = [
        path for path in root.glob("**/predictions.jsonl")
        if sum(1 for _ in path.open()) == EXPECTED_ROWS
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one complete prediction below {root}; found {len(matches)}")
    return matches[0]


def outcomes(path: Path, truth: list[str]) -> tuple[list[bool], list[bool]]:
    rows = read_jsonl(path)
    if len(rows) != len(truth):
        raise RuntimeError(f"Row mismatch for {path}: {len(rows)} vs {len(truth)}")
    top1: list[bool] = []
    top10: list[bool] = []
    for prediction, expected in zip(rows, truth):
        ranked = sorted(prediction.items(), key=lambda item: float(item[1]), reverse=True)
        candidates = [canonical(smiles) for smiles, _ in ranked[:10]]
        top1.append(bool(candidates) and candidates[0] == expected)
        top10.append(expected in candidates)
    return top1, top10


def cluster_bootstrap_ci(
    left: list[bool], right: list[bool], groups: list[str], random_seed: int
) -> tuple[float, float]:
    unique = sorted(set(groups))
    locations = {value: index for index, value in enumerate(unique)}
    sums = np.zeros(len(unique), dtype=float)
    counts = np.zeros(len(unique), dtype=float)
    for a, b, group in zip(left, right, groups):
        index = locations[group]
        sums[index] += int(b) - int(a)
        counts[index] += 1
    rng = np.random.default_rng(random_seed)
    gains = []
    for _ in range(5000):
        sample = rng.integers(0, len(unique), len(unique))
        gains.append(100 * sums[sample].sum() / counts[sample].sum())
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def paired_row(
    seed: int, metric: str, left: list[bool], right: list[bool], groups: list[str]
) -> dict:
    promoted = sum((not a) and b for a, b in zip(left, right))
    demoted = sum(a and (not b) for a, b in zip(left, right))
    discordant = promoted + demoted
    p_value = binomtest(min(promoted, demoted), discordant, 0.5).pvalue if discordant else 1.0
    left_pct = 100 * sum(left) / len(left)
    right_pct = 100 * sum(right) / len(right)
    ci_low, ci_high = cluster_bootstrap_ci(
        left, right, groups, seed + zlib.crc32(f"r32:{metric}".encode())
    )
    return {
        "seed": seed,
        "metric": metric,
        "qv_percent": left_pct,
        "all_attention_percent": right_pct,
        "gain_pp": right_pct - left_pct,
        "promoted": promoted,
        "demoted": demoted,
        "mcnemar_exact_p": p_value,
        "structure_cluster_bootstrap_95ci_low_pp": ci_low,
        "structure_cluster_bootstrap_95ci_high_pp": ci_high,
    }


def main() -> None:
    options = args()
    output = options.output_dir if options.output_dir.is_absolute() else ROOT / options.output_dir
    if "test" in str(LABELS).lower():
        raise RuntimeError("Validation-only guard rejected labels")
    output.mkdir(parents=True, exist_ok=True)

    truth_optional = [canonical(row["smiles"]) for row in read_jsonl(LABELS)]
    if len(truth_optional) != EXPECTED_ROWS or any(item is None for item in truth_optional):
        raise RuntimeError("Invalid or incomplete validation labels")
    truth: list[str] = [item for item in truth_optional if item is not None]

    values: dict[tuple[str, int], dict[str, list[bool]]] = {}
    run_rows = []
    for condition, pattern in RUNS.items():
        for seed in SEEDS:
            path = prediction_file(pattern, seed)
            top1, top10 = outcomes(path, truth)
            values[(condition, seed)] = {"top1": top1, "top10": top10}
            run_rows.append({
                "condition": condition,
                "seed": seed,
                "n": len(truth),
                "strict_top1_correct": sum(top1),
                "strict_top1_percent": 100 * sum(top1) / len(top1),
                "strict_top10_correct": sum(top10),
                "strict_top10_percent": 100 * sum(top10) / len(top10),
                "predictions": str(path.relative_to(ROOT)),
            })

    paired_rows = []
    for seed in SEEDS:
        for metric in ("top1", "top10"):
            paired_rows.append(paired_row(
                seed,
                metric,
                values[("qv", seed)][metric],
                values[("all_attention", seed)][metric],
                truth,
            ))

    run_df = pd.DataFrame(run_rows).sort_values(["condition", "seed"])
    paired_df = pd.DataFrame(paired_rows).sort_values(["metric", "seed"])
    aggregate_rows = []
    for metric, frame in paired_df.groupby("metric"):
        gains = frame["gain_pp"].to_numpy(dtype=float)
        mean = float(gains.mean())
        sd = float(gains.std(ddof=1))
        critical = float(student_t.ppf(0.975, len(gains) - 1))
        half_width = critical * sd / math.sqrt(len(gains))
        aggregate_rows.append({
            "metric": metric,
            "number_of_seeds": len(gains),
            "mean_gain_pp": mean,
            "sd_gain_pp": sd,
            "min_gain_pp": float(gains.min()),
            "max_gain_pp": float(gains.max()),
            "positive_seed_count": int((gains > 0).sum()),
            "seed_t_95ci_low_pp": mean - half_width,
            "seed_t_95ci_high_pp": mean + half_width,
        })
    aggregate = pd.DataFrame(aggregate_rows).sort_values("metric")
    top1 = aggregate.loc[aggregate["metric"] == "top1"].iloc[0]
    select = bool(top1["mean_gain_pp"] > 0 and top1["positive_seed_count"] >= 2)

    run_df.to_csv(output / "run_metrics.csv", index=False)
    paired_df.to_csv(output / "paired_comparisons.csv", index=False)
    aggregate.to_csv(output / "aggregate_effects.csv", index=False)
    summary = {
        "design": "r32 alpha64 q/v versus q/k/v/out attention projections",
        "data": "VGWD clean validation only",
        "validation_rows": EXPECTED_ROWS,
        "seeds": list(SEEDS),
        "checkpoint": 3500,
        "beam": 10,
        "primary_endpoint": "strict_top1",
        "selection_rule": "mean Top-1 gain > 0 and positive Top-1 gain in at least 2/3 seeds",
        "select_all_attention_for_locked_holdout": select,
        "run_metrics": run_rows,
        "aggregate_effects": aggregate.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "# rsLoRA r32 all-attention validation-only screen",
        "",
        "Seeds 7, 123, 2026; checkpoint-3500; deterministic Beam 10; 664 validation rows.",
        "The VGWD test was not used or generated for this screen.",
        "",
        "## Individual validation runs",
        "",
        run_df.drop(columns="predictions").to_markdown(index=False),
        "",
        "## Paired all-attention minus q/v effects",
        "",
        paired_df.to_markdown(index=False),
        "",
        "## Seed aggregate",
        "",
        aggregate.to_markdown(index=False),
        "",
        "## Preregistered decision",
        "",
        f"Select r32 all-attention for a future locked/external evaluation: **{'YES' if select else 'NO'}**.",
        "The gate uses strict Top-1 only: mean gain > 0 and positive gain in at least 2/3 seeds.",
        "Top-10 is secondary. n=3 intervals are exploratory and do not establish external generalization.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    print(run_df.to_string(index=False))
    print("\nPAIRED EFFECTS")
    print(paired_df.to_string(index=False))
    print("\nAGGREGATE EFFECTS")
    print(aggregate.to_string(index=False))
    print(f"\nSELECT_ALL_ATTENTION_FOR_LOCKED_HOLDOUT={select}")
    print(f"OUTPUT: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
