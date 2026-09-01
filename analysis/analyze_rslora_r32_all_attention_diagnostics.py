#!/usr/bin/env python3
"""Descriptive validation diagnostics for r32 q/v versus all-attention."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/vgwd_clean/valid_filtered_mz500.jsonl"
TRAIN = ROOT / "data/vgwd_clean/train.jsonl"
OUTPUT = ROOT / "analysis/rslora_r32_all_attention_diagnostics"
SEEDS = (7, 123, 2026)
EXPECTED_ROWS = 664
RUNS = {
    "qv": "predictions/vgwd_clean_rslora_r32_seed{seed}_valid_beam10",
    "all_attention": "predictions/vgwd_clean_rslora_r32_allattn_seed{seed}_valid_beam10",
}


def read_jsonl(path: Path) -> list[dict]:
    if "test" in str(path).lower():
        raise RuntimeError(f"Validation-only guard rejected path: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No rows found: {path}")
    return rows


def prediction_file(pattern: str, seed: int) -> Path:
    root = ROOT / pattern.format(seed=seed)
    if "test" in str(root).lower():
        raise RuntimeError(f"Validation-only guard rejected path: {root}")
    matches = [path for path in root.glob("**/predictions.jsonl") if sum(1 for _ in path.open()) == EXPECTED_ROWS]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one 664-row prediction below {root}; found {len(matches)}")
    return matches[0]


def mol(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles) if smiles else None


def canonical(smiles: str, stereo: bool = True) -> str | None:
    molecule = mol(smiles)
    if molecule is None:
        return None
    if not stereo:
        Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=stereo)


def scaffold(molecule: Chem.Mol | None) -> str:
    if molecule is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)


def fixed_bin(value: float, boundaries: list[float], labels: list[str]) -> str:
    index = int(np.digitize([value], boundaries, right=False)[0])
    return labels[index]


def ranked_candidates(row: dict) -> list[str]:
    return [key for key, _ in sorted(row.items(), key=lambda item: float(item[1]), reverse=True)[:10]]


def label_frame() -> pd.DataFrame:
    rows = read_jsonl(LABELS)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} labels; found {len(rows)}")

    train_scaffolds: set[str] = set()
    for row in read_jsonl(TRAIN):
        value = scaffold(mol(row["smiles"]))
        if value:
            train_scaffolds.add(value)

    output = []
    for index, row in enumerate(rows):
        molecule = mol(row["smiles"])
        if molecule is None:
            raise RuntimeError(f"Invalid label at row {index}: {row['smiles']}")
        exact = canonical(row["smiles"], True)
        connectivity = canonical(row["smiles"], False)
        scaffold_value = scaffold(molecule)
        molecular_weight = float(Descriptors.MolWt(molecule))
        max_mz = float(max(row["mz"]))
        peak_count = len(row["mz"])
        output.append({
            "row": index,
            "label_smiles": row["smiles"],
            "exact": exact,
            "connectivity": connectivity,
            "scaffold": scaffold_value,
            "scaffold_status": "acyclic" if not scaffold_value else ("train_seen" if scaffold_value in train_scaffolds else "train_unseen"),
            "molecular_weight": molecular_weight,
            "molecular_weight_bin": fixed_bin(molecular_weight, [150, 250, 350], ["<150", "150-249", "250-349", ">=350"]),
            "max_mz": max_mz,
            "max_mz_bin": fixed_bin(max_mz, [100, 200, 300, 400], ["<100", "100-199", "200-299", "300-399", "400-500"]),
            "peak_count": peak_count,
            "peak_count_bin": fixed_bin(peak_count, [25, 50, 100], ["<25", "25-49", "50-99", ">=100"]),
            "stereo_annotated": exact != connectivity,
        })
    return pd.DataFrame(output)


def evaluate(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, int], Path]]:
    records = []
    paths: dict[tuple[str, int], Path] = {}
    for condition, pattern in RUNS.items():
        for seed in SEEDS:
            path = prediction_file(pattern, seed)
            paths[(condition, seed)] = path
            predictions = read_jsonl(path)
            for label, prediction in zip(labels.itertuples(index=False), predictions):
                raw = ranked_candidates(prediction)
                exact = [canonical(value, True) for value in raw]
                connectivity = [canonical(value, False) for value in raw]
                strict_rank = exact.index(label.exact) + 1 if label.exact in exact else None
                connectivity_rank = connectivity.index(label.connectivity) + 1 if label.connectivity in connectivity else None
                records.append({
                    "row": label.row,
                    "condition": condition,
                    "seed": seed,
                    "top1_smiles": raw[0] if raw else "",
                    "strict_rank": strict_rank,
                    "connectivity_rank": connectivity_rank,
                    "strict_top1": strict_rank == 1,
                    "strict_top10": strict_rank is not None,
                    "connectivity_top1": connectivity_rank == 1,
                    "connectivity_top10": connectivity_rank is not None,
                    "stereo_gap_top1": connectivity_rank == 1 and strict_rank != 1,
                    "top10_rescue": strict_rank is not None and strict_rank > 1,
                    "invalid_top1": bool(raw) and exact[0] is None,
                    "invalid_in_top10": sum(value is None for value in exact),
                })
    return pd.DataFrame(records), paths


def diagnostic_metrics(evaluations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, seed), frame in evaluations.groupby(["condition", "seed"], sort=True):
        n = len(frame)
        rows.append({
            "condition": condition,
            "seed": seed,
            "n": n,
            "strict_top1_percent": 100 * frame["strict_top1"].mean(),
            "strict_top10_percent": 100 * frame["strict_top10"].mean(),
            "connectivity_top1_percent": 100 * frame["connectivity_top1"].mean(),
            "connectivity_top10_percent": 100 * frame["connectivity_top10"].mean(),
            "stereo_gap_top1_count": int(frame["stereo_gap_top1"].sum()),
            "stereo_gap_top1_percent": 100 * frame["stereo_gap_top1"].mean(),
            "top10_rescue_count": int(frame["top10_rescue"].sum()),
            "top10_rescue_percent": 100 * frame["top10_rescue"].mean(),
            "invalid_top1_count": int(frame["invalid_top1"].sum()),
            "invalid_candidates_in_top10": int(frame["invalid_in_top10"].sum()),
        })
    return pd.DataFrame(rows)


def paired_transitions(evaluations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        qv = evaluations[(evaluations.condition == "qv") & (evaluations.seed == seed)].sort_values("row")
        all_attention = evaluations[(evaluations.condition == "all_attention") & (evaluations.seed == seed)].sort_values("row")
        left = qv["strict_top1"].to_numpy(dtype=bool)
        right = all_attention["strict_top1"].to_numpy(dtype=bool)
        rows.append({
            "seed": seed,
            "both_correct": int((left & right).sum()),
            "qv_only_demoted": int((left & ~right).sum()),
            "all_attention_only_promoted": int((~left & right).sum()),
            "both_wrong": int((~left & ~right).sum()),
            "net_gain": int(right.sum() - left.sum()),
        })
    return pd.DataFrame(rows)


def subgroup_tables(labels: pd.DataFrame, evaluations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = evaluations.merge(labels, on="row", validate="many_to_one")
    dimensions = ["molecular_weight_bin", "max_mz_bin", "peak_count_bin", "scaffold_status", "stereo_annotated"]
    rows = []
    for dimension in dimensions:
        for group, group_frame in merged.groupby(dimension, dropna=False):
            for seed in SEEDS:
                seed_frame = group_frame[group_frame.seed == seed]
                qv = seed_frame[seed_frame.condition == "qv"]
                all_attention = seed_frame[seed_frame.condition == "all_attention"]
                if len(qv) != len(all_attention):
                    raise RuntimeError(f"Unpaired subgroup {dimension}={group}, seed={seed}")
                rows.append({
                    "dimension": dimension,
                    "group": str(group),
                    "seed": seed,
                    "n": len(qv),
                    "qv_top1_percent": 100 * qv.strict_top1.mean(),
                    "all_attention_top1_percent": 100 * all_attention.strict_top1.mean(),
                    "top1_gain_pp": 100 * (all_attention.strict_top1.mean() - qv.strict_top1.mean()),
                    "qv_top10_percent": 100 * qv.strict_top10.mean(),
                    "all_attention_top10_percent": 100 * all_attention.strict_top10.mean(),
                    "top10_gain_pp": 100 * (all_attention.strict_top10.mean() - qv.strict_top10.mean()),
                })
    by_seed = pd.DataFrame(rows)
    summary = by_seed.groupby(["dimension", "group"], as_index=False).agg(
        n=("n", "first"),
        mean_qv_top1_percent=("qv_top1_percent", "mean"),
        mean_all_attention_top1_percent=("all_attention_top1_percent", "mean"),
        mean_top1_gain_pp=("top1_gain_pp", "mean"),
        positive_top1_seeds=("top1_gain_pp", lambda values: int((values > 0).sum())),
        mean_qv_top10_percent=("qv_top10_percent", "mean"),
        mean_all_attention_top10_percent=("all_attention_top10_percent", "mean"),
        mean_top10_gain_pp=("top10_gain_pp", "mean"),
        positive_top10_seeds=("top10_gain_pp", lambda values: int((values > 0).sum())),
    )
    return by_seed, summary


def consistency_tables(labels: pd.DataFrame, evaluations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = evaluations.groupby(["row", "condition"], as_index=False)["strict_top1"].sum()
    pivot = counts.pivot(index="row", columns="condition", values="strict_top1").reset_index()
    pivot = pivot.rename(columns={"qv": "qv_correct_seeds", "all_attention": "all_attention_correct_seeds"})
    pivot["delta_correct_seeds"] = pivot["all_attention_correct_seeds"] - pivot["qv_correct_seeds"]
    samples = labels.merge(pivot, on="row", validate="one_to_one")
    distribution = samples.groupby(["qv_correct_seeds", "all_attention_correct_seeds"], as_index=False).size().rename(columns={"size": "sample_count"})
    hard = samples[(samples.qv_correct_seeds == 0) & (samples.all_attention_correct_seeds == 0)].copy()
    promotions = samples[(samples.qv_correct_seeds == 0) & (samples.all_attention_correct_seeds >= 2)].copy()
    regressions = samples[(samples.qv_correct_seeds >= 2) & (samples.all_attention_correct_seeds == 0)].copy()
    return samples, distribution, hard, pd.concat([promotions.assign(category="consistent_promotion"), regressions.assign(category="consistent_regression")], ignore_index=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels = label_frame()
    evaluations, paths = evaluate(labels)
    metrics = diagnostic_metrics(evaluations)
    transitions = paired_transitions(evaluations)
    subgroup_by_seed, subgroup_summary = subgroup_tables(labels, evaluations)
    samples, consistency, hard, stable_changes = consistency_tables(labels, evaluations)

    evaluations.to_csv(OUTPUT / "per_sample_seed_outcomes.csv", index=False)
    metrics.to_csv(OUTPUT / "diagnostic_metrics.csv", index=False)
    transitions.to_csv(OUTPUT / "paired_top1_transitions.csv", index=False)
    subgroup_by_seed.to_csv(OUTPUT / "subgroup_by_seed.csv", index=False)
    subgroup_summary.to_csv(OUTPUT / "subgroup_summary.csv", index=False)
    samples.to_csv(OUTPUT / "cross_seed_sample_consistency.csv", index=False)
    consistency.to_csv(OUTPUT / "cross_seed_consistency_matrix.csv", index=False)
    hard.to_csv(OUTPUT / "common_hard_cases.csv", index=False)
    stable_changes.to_csv(OUTPUT / "consistent_promotions_regressions.csv", index=False)

    concise = subgroup_summary[subgroup_summary.n >= 20].copy()
    summary = {
        "design": "descriptive validation-only diagnostics; no architecture tuning",
        "validation_rows": EXPECTED_ROWS,
        "seeds": list(SEEDS),
        "checkpoint": 3500,
        "beam": 10,
        "test_used": False,
        "prediction_files": {f"{condition}_seed{seed}": str(path.relative_to(ROOT)) for (condition, seed), path in paths.items()},
        "diagnostic_metrics": metrics.to_dict(orient="records"),
        "paired_top1_transitions": transitions.to_dict(orient="records"),
        "common_hard_case_count": len(hard),
        "consistent_promotion_count": int((stable_changes.category == "consistent_promotion").sum()) if len(stable_changes) else 0,
        "consistent_regression_count": int((stable_changes.category == "consistent_regression").sum()) if len(stable_changes) else 0,
        "cross_seed_delta_distribution": {str(int(key)): int(value) for key, value in Counter(samples.delta_correct_seeds).items()},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "# r32 all-attention validation diagnostics",
        "",
        "Descriptive analysis of 664 filtered validation rows, seeds 7/123/2026, checkpoint-3500 and Beam 10.",
        "The VGWD test was not used. These diagnostics do not select another architecture.",
        "",
        "## Overall diagnostics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Paired strict Top-1 transitions",
        "",
        transitions.to_markdown(index=False),
        "",
        "## Prespecified subgroup summary",
        "",
        "Rows below have at least 20 validation samples. Percentages are means of the three seed-specific accuracies.",
        "",
        concise.to_markdown(index=False),
        "",
        "## Cross-seed consistency",
        "",
        consistency.to_markdown(index=False),
        "",
        f"- Both methods wrong in all three seeds: {len(hard)} samples.",
        f"- q/v wrong in all seeds but all-attention correct in at least two: {summary['consistent_promotion_count']} samples.",
        f"- q/v correct in at least two seeds but all-attention wrong in all seeds: {summary['consistent_regression_count']} samples.",
        "",
        "## Interpretation boundary",
        "",
        "Subgroup results are descriptive and share the validation data used for method selection. They are not independent confirmation, are not multiplicity-adjusted hypothesis tests, and must not be presented as external generalization.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    print(metrics.to_string(index=False))
    print("\nPAIRED TOP-1 TRANSITIONS")
    print(transitions.to_string(index=False))
    print("\nSUBGROUPS (N >= 20)")
    print(concise.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"prediction_files", "diagnostic_metrics", "paired_top1_transitions"}}, indent=2))
    print(f"OUTPUT: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
