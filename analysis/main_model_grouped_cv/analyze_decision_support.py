#!/usr/bin/env python3
"""Cross-fitted OOF calibration, abstention, risk-coverage, and Top-k utility."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/vgwd_main_grouped_cv_locked"
PRED_ROOT = ROOT / "predictions/vgwd_main_grouped_cv"
OUT = ROOT / "analysis/main_model_grouped_cv/decision_support"
CONDITIONS = ("qv", "allattn")
FOLDS = tuple(range(5))
EPS = 1e-300
PRIMARY_THRESHOLD = 0.80
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 20260905


def read_jsonl(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_complete_run(fold: int, condition: str) -> Path:
    expected = sum(1 for _ in (DATA_ROOT / f"fold_{fold}/oof.jsonl").open())
    found = []
    for path in (PRED_ROOT / f"fold{fold}/{condition}").glob("**/predictions.jsonl"):
        if sum(1 for _ in path.open()) != expected:
            continue
        audit = path.parent / "strict_exact_topk_samples.csv"
        if audit.is_file() and len(read_csv(audit)) == expected:
            found.append(path.parent)
    if len(found) != 1:
        raise RuntimeError(
            f"Expected exactly one complete run for fold={fold}, condition={condition}; "
            f"found {len(found)}"
        )
    return found[0]


def fold_groups(fold: int) -> list[str]:
    rows = read_csv(DATA_ROOT / "assignments.csv")
    groups = [
        row["group_sha256"]
        for row in rows
        if row["partition"] == "development" and int(row["oof_fold"]) == fold
    ]
    expected = sum(1 for _ in (DATA_ROOT / f"fold_{fold}/oof.jsonl").open())
    if len(groups) != expected:
        raise RuntimeError(f"Group length mismatch for fold {fold}")
    return groups


def confidence_features(prediction: dict[str, Any]) -> tuple[list[float], dict[str, float]]:
    scores = np.asarray(
        sorted((max(float(value), 0.0) for value in prediction.values()), reverse=True),
        dtype=float,
    )
    has_candidate = bool(len(scores))
    if not has_candidate:
        return (
            [math.log(EPS), 0.0, 0.0, 0.0],
            {
                "has_candidate": 0,
                "top1_score": 0.0,
                "top2_score": 0.0,
                "top1_score_share": 0.0,
                "normalized_entropy": 0.0,
                "candidate_count": 0,
            },
        )
    top1 = max(float(scores[0]), EPS)
    top2 = max(float(scores[1]), EPS) if len(scores) > 1 else top1
    total = max(float(scores.sum()), EPS)
    shares = scores / total
    if len(shares) > 1:
        entropy = float(-(shares * np.log(np.clip(shares, EPS, 1.0))).sum())
        normalized_entropy = entropy / math.log(len(shares))
    else:
        normalized_entropy = 0.0
    raw_share = float(shares[0])
    features = [
        math.log(top1),
        math.log(top1 / top2),
        normalized_entropy,
        float(len(scores)),
    ]
    diagnostics = {
        "has_candidate": 1,
        "top1_score": top1,
        "top2_score": top2,
        "top1_score_share": raw_share,
        "normalized_entropy": normalized_entropy,
        "candidate_count": int(len(scores)),
    }
    return features, diagnostics


def load_rows() -> list[dict[str, Any]]:
    lock = json.loads((DATA_ROOT / "LOCK.json").read_text())
    if lock["evaluation_state"] != "LOCKED_NOT_EVALUATED":
        raise RuntimeError("Locked holdout is not in the expected unevaluated state")
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        groups = fold_groups(fold)
        for condition in CONDITIONS:
            run = find_complete_run(fold, condition)
            predictions = read_jsonl(run / "predictions.jsonl")
            audits = read_csv(run / "strict_exact_topk_samples.csv")
            if not (len(predictions) == len(audits) == len(groups)):
                raise RuntimeError(f"Row alignment failure for fold={fold}, {condition}")
            for row_id, (prediction, audit, group) in enumerate(
                zip(predictions, audits, groups)
            ):
                features, diagnostics = confidence_features(prediction)
                exact_rank = int(audit["exact_rank"]) if audit["exact_rank"] else 0
                rows.append(
                    {
                        "condition": condition,
                        "fold": fold,
                        "fold_row": row_id,
                        "group_sha256": group,
                        "correct_top1": int(exact_rank == 1),
                        "exact_rank": exact_rank,
                        "feature_log_top1": features[0],
                        "feature_log_ratio": features[1],
                        "feature_entropy": features[2],
                        "feature_candidate_count": features[3],
                        **diagnostics,
                    }
                )
    return rows


FEATURES = (
    "feature_log_top1",
    "feature_log_ratio",
    "feature_entropy",
    "feature_candidate_count",
)


def add_cross_fitted_probabilities(rows: list[dict[str, Any]]) -> None:
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        for heldout_fold in FOLDS:
            train = [
                row for row in condition_rows
                if row["fold"] != heldout_fold and row["has_candidate"]
            ]
            test = [row for row in condition_rows if row["fold"] == heldout_fold]
            x_train = np.asarray([[row[key] for key in FEATURES] for row in train])
            y_train = np.asarray([row["correct_top1"] for row in train])
            valid_test = [row for row in test if row["has_candidate"]]
            x_test = np.asarray([[row[key] for key in FEATURES] for row in valid_test])
            calibrator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=BOOTSTRAP_SEED,
                ),
            )
            calibrator.fit(x_train, y_train)
            probabilities = calibrator.predict_proba(x_test)[:, 1]
            for row in test:
                if not row["has_candidate"]:
                    row["calibrated_probability"] = 0.0
            for row, probability in zip(valid_test, probabilities):
                row["calibrated_probability"] = float(probability)


def fit_frozen_calibrators(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit the already evaluated fixed calibrator on all OOF rows for holdout use."""
    artifacts: dict[str, Any] = {
        "format": "standard_scaler_then_logistic_regression",
        "sklearn_version": sklearn.__version__,
        "features_in_order": list(FEATURES),
        "primary_condition": "allattn",
        "primary_abstention_threshold": PRIMARY_THRESHOLD,
        "conditions": {},
    }
    for condition in CONDITIONS:
        selected = [
            row for row in rows
            if row["condition"] == condition and row["has_candidate"]
        ]
        x = np.asarray([[row[key] for key in FEATURES] for row in selected])
        y = np.asarray([row["correct_top1"] for row in selected])
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=BOOTSTRAP_SEED,
            ),
        )
        model.fit(x, y)
        scaler = model.named_steps["standardscaler"]
        logistic = model.named_steps["logisticregression"]
        artifacts["conditions"][condition] = {
            "n_oof_fit": len(selected),
            "positive_rows": int(y.sum()),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "logistic_coefficients": logistic.coef_[0].tolist(),
            "logistic_intercept": float(logistic.intercept_[0]),
            "classes": logistic.classes_.tolist(),
        }
    return artifacts


def equal_frequency_bins(
    rows: list[dict[str, Any]], probability_key: str, bins: int = 10
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row[probability_key], row["fold"], row["fold_row"]))
    index_groups = np.array_split(np.arange(len(ordered)), bins)
    output = []
    for bin_index, indices in enumerate(index_groups, start=1):
        selected = [ordered[int(index)] for index in indices]
        output.append(
            {
                "bin": bin_index,
                "n": len(selected),
                "mean_confidence": float(np.mean([row[probability_key] for row in selected])),
                "empirical_accuracy": float(np.mean([row["correct_top1"] for row in selected])),
                "min_confidence": float(min(row[probability_key] for row in selected)),
                "max_confidence": float(max(row[probability_key] for row in selected)),
            }
        )
    return output


def calibration_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = []
    reliability = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        y = np.asarray([row["correct_top1"] for row in selected], dtype=int)
        for method, key in (
            ("uncalibrated_beam_share", "top1_score_share"),
            ("cross_fitted_logistic", "calibrated_probability"),
        ):
            probabilities = np.clip(
                np.asarray([row[key] for row in selected], dtype=float), 1e-12, 1 - 1e-12
            )
            bins = equal_frequency_bins(selected, key)
            ece = sum(
                item["n"] / len(selected)
                * abs(item["mean_confidence"] - item["empirical_accuracy"])
                for item in bins
            )
            metrics.append(
                {
                    "condition": condition,
                    "method": method,
                    "n": len(selected),
                    "prevalence": float(y.mean()),
                    "brier": float(brier_score_loss(y, probabilities)),
                    "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
                    "ece_10_equal_frequency": float(ece),
                }
            )
            reliability.extend(
                {"condition": condition, "method": method, **item} for item in bins
            )
    return metrics, reliability


def policy_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    accepted = [row for row in rows if row["calibrated_probability"] >= threshold]
    n = len(rows)
    correct = sum(row["correct_top1"] for row in accepted)
    coverage = len(accepted) / n if n else 0.0
    accuracy = correct / len(accepted) if accepted else float("nan")
    return {
        "threshold": threshold,
        "n": n,
        "accepted": len(accepted),
        "abstained": n - len(accepted),
        "coverage": coverage,
        "selective_accuracy": accuracy,
        "selective_risk": 1.0 - accuracy if accepted else float("nan"),
        "correct_accepted": correct,
    }


def primary_cluster_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group_sha256"]].append(row)
    keys = sorted(by_group)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    coverage = np.empty(BOOTSTRAPS, dtype=float)
    accuracy = np.full(BOOTSTRAPS, np.nan, dtype=float)
    for draw in range(BOOTSTRAPS):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        sampled_rows = [row for group in sampled for row in by_group[group]]
        accepted = [
            row for row in sampled_rows
            if row["calibrated_probability"] >= PRIMARY_THRESHOLD
        ]
        coverage[draw] = len(accepted) / len(sampled_rows)
        if accepted:
            accuracy[draw] = np.mean([row["correct_top1"] for row in accepted])
    return {
        "draws": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "coverage_95ci": [float(value) for value in np.quantile(coverage, [0.025, 0.975])],
        "selective_accuracy_95ci": [
            float(value) for value in np.nanquantile(accuracy, [0.025, 0.975])
        ],
        "draws_without_accepted_rows": int(np.isnan(accuracy).sum()),
    }


def risk_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for condition in CONDITIONS:
        selected = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: row["calibrated_probability"],
            reverse=True,
        )
        n = len(selected)
        for target_percent in range(5, 101, 5):
            accepted_n = max(1, math.ceil(n * target_percent / 100))
            accepted = selected[:accepted_n]
            accuracy = float(np.mean([row["correct_top1"] for row in accepted]))
            output.append(
                {
                    "condition": condition,
                    "target_coverage": target_percent / 100,
                    "actual_coverage": accepted_n / n,
                    "accepted": accepted_n,
                    "threshold_at_boundary": accepted[-1]["calibrated_probability"],
                    "selective_accuracy": accuracy,
                    "selective_risk": 1.0 - accuracy,
                }
            )
    return output


def topk_utility(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        n = len(selected)
        top1_hits = sum(row["exact_rank"] == 1 for row in selected)
        reciprocal_ranks = [
            1.0 / row["exact_rank"] if 1 <= row["exact_rank"] <= 10 else 0.0
            for row in selected
        ]
        mrr10 = float(np.mean(reciprocal_ranks))
        for k in (1, 3, 5, 10):
            hits = sum(1 <= row["exact_rank"] <= k for row in selected)
            additional = hits - top1_hits
            extra_slots = n * (k - 1)
            output.append(
                {
                    "condition": condition,
                    "k": k,
                    "n": n,
                    "hits": hits,
                    "accuracy": hits / n,
                    "incremental_hits_vs_top1": additional,
                    "incremental_accuracy_pp_vs_top1": additional / n * 100,
                    "extra_candidate_slots_vs_top1": extra_slots,
                    "extra_slots_per_additional_hit": (
                        extra_slots / additional if additional else ""
                    ),
                    "mrr_at_10": mrr10,
                }
            )
    return output


def save_figures(
    reliability: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    utility: list[dict[str, Any]],
) -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, condition in zip(axes, CONDITIONS):
        axis.plot([0, 1], [0, 1], "--", color="0.5", label="Ideal")
        for method, marker in (("uncalibrated_beam_share", "o"), ("cross_fitted_logistic", "s")):
            values = [
                row for row in reliability
                if row["condition"] == condition and row["method"] == method
            ]
            label = "Raw beam share" if method.startswith("uncalibrated") else "Cross-fitted calibrated"
            axis.plot(
                [row["mean_confidence"] for row in values],
                [row["empirical_accuracy"] for row in values],
                marker=marker,
                label=label,
            )
        axis.set(title=condition, xlabel="Mean confidence", ylabel="Empirical Top-1 accuracy", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(OUT / "calibration_reliability.png", bbox_inches="tight")
    fig.savefig(OUT / "calibration_reliability.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    for condition, marker in (("qv", "o"), ("allattn", "s")):
        values = [row for row in risk_rows if row["condition"] == condition]
        ax.plot(
            [100 * row["actual_coverage"] for row in values],
            [100 * row["selective_risk"] for row in values],
            marker=marker,
            label=condition,
        )
    ax.set(xlabel="Coverage (%)", ylabel="Selective risk (%)", xlim=(0, 100), ylim=(0, None))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "risk_coverage.png", bbox_inches="tight")
    fig.savefig(OUT / "risk_coverage.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
    width = 0.34
    ks = [1, 3, 5, 10]
    for index, condition in enumerate(CONDITIONS):
        values = [row for row in utility if row["condition"] == condition]
        ax.bar(
            np.arange(len(ks)) + (index - 0.5) * width,
            [100 * row["accuracy"] for row in values],
            width,
            label=condition,
        )
    ax.set_xticks(np.arange(len(ks)), [f"Top-{k}" for k in ks])
    ax.set(ylabel="Strict exact accuracy (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "topk_quantitative_utility.png", bbox_inches="tight")
    fig.savefig(OUT / "topk_quantitative_utility.pdf", bbox_inches="tight")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    add_cross_fitted_probabilities(rows)
    metrics, reliability = calibration_metrics(rows)
    abstention = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        abstention.extend(
            {"condition": condition, **policy_metrics(selected, threshold)}
            for threshold in THRESHOLDS
        )
    risk_rows = risk_coverage(rows)
    utility = topk_utility(rows)
    frozen_calibrators = fit_frozen_calibrators(rows)
    primary_rows = [row for row in rows if row["condition"] == "allattn"]
    primary_policy = policy_metrics(primary_rows, PRIMARY_THRESHOLD)
    primary_policy["cluster_bootstrap"] = primary_cluster_bootstrap(primary_rows)

    sample_fields = [
        "condition", "fold", "fold_row", "group_sha256", "correct_top1", "exact_rank",
        "has_candidate", "top1_score", "top2_score", "top1_score_share", "normalized_entropy",
        "candidate_count", *FEATURES, "calibrated_probability",
    ]
    write_csv(OUT / "sample_scores.csv", rows, sample_fields)
    write_csv(
        OUT / "calibration_metrics.csv", metrics,
        ["condition", "method", "n", "prevalence", "brier", "log_loss", "ece_10_equal_frequency"],
    )
    write_csv(
        OUT / "reliability_bins.csv", reliability,
        ["condition", "method", "bin", "n", "mean_confidence", "empirical_accuracy", "min_confidence", "max_confidence"],
    )
    write_csv(
        OUT / "abstention_thresholds.csv", abstention,
        ["condition", "threshold", "n", "accepted", "abstained", "coverage", "selective_accuracy", "selective_risk", "correct_accepted"],
    )
    write_csv(
        OUT / "risk_coverage.csv", risk_rows,
        ["condition", "target_coverage", "actual_coverage", "accepted", "threshold_at_boundary", "selective_accuracy", "selective_risk"],
    )
    write_csv(
        OUT / "topk_quantitative_utility.csv", utility,
        ["condition", "k", "n", "hits", "accuracy", "incremental_hits_vs_top1", "incremental_accuracy_pp_vs_top1", "extra_candidate_slots_vs_top1", "extra_slots_per_additional_hit", "mrr_at_10"],
    )
    save_figures(reliability, risk_rows, utility)

    summary = {
        "scope": "grouped_5fold_oof_only",
        "holdout_state": "LOCKED_NOT_EVALUATED",
        "primary_condition": "allattn",
        "primary_threshold": PRIMARY_THRESHOLD,
        "primary_policy": primary_policy,
        "calibration_metrics": metrics,
        "topk_quantitative_utility": utility,
        "method_limitations": [
            "Internal re-split only; no external or locked-holdout result.",
            "Five folds are not five independent training seeds.",
            "Top-k workload metrics are proxies, not a researcher user study.",
            "Sequence scores are empirically calibrated ranking scores, not native class probabilities.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "frozen_calibrators.json").write_text(
        json.dumps(frozen_calibrators, indent=2) + "\n"
    )

    allattn_cal = next(
        row for row in metrics
        if row["condition"] == "allattn" and row["method"] == "cross_fitted_logistic"
    )
    allattn_raw = next(
        row for row in metrics
        if row["condition"] == "allattn" and row["method"] == "uncalibrated_beam_share"
    )
    allattn_utility = [row for row in utility if row["condition"] == "allattn"]
    report = [
        "# OOF calibration, abstention, risk-coverage, and Top-k utility",
        "",
        "- Scope: completed grouped 5-fold OOF only; locked holdout untouched.",
        "- Primary candidate: r32/alpha64 all-attention.",
        "- Calibration: five-fold cross-fitted logistic model; each fold calibrated from the other four.",
        f"- All-attention raw/calibrated Brier: {allattn_raw['brier']:.6f} / {allattn_cal['brier']:.6f}.",
        f"- All-attention raw/calibrated ECE: {allattn_raw['ece_10_equal_frequency']:.6f} / {allattn_cal['ece_10_equal_frequency']:.6f}.",
        f"- Frozen abstention threshold: calibrated P(correct) >= {PRIMARY_THRESHOLD:.2f}.",
        f"- Primary coverage: {primary_policy['coverage']:.4%} ({primary_policy['accepted']:,}/{primary_policy['n']:,}).",
        f"- Primary selective accuracy: {primary_policy['selective_accuracy']:.4%}.",
        "",
        "## All-attention Top-k quantitative workload proxies",
        "",
        "| K | Strict accuracy | Hits | Added hits vs Top-1 | Extra slots per added hit |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in allattn_utility:
        burden = row["extra_slots_per_additional_hit"]
        burden_text = "—" if burden == "" else f"{float(burden):.2f}"
        report.append(
            f"| {row['k']} | {row['accuracy']:.4%} | {row['hits']:,} | "
            f"{row['incremental_hits_vs_top1']:,} | {burden_text} |"
        )
    report.extend(
        [
            "",
            "These workload figures are quantitative proxies, not an actual researcher user study.",
            "The threshold is frozen for a future one-time holdout evaluation and was not changed after viewing results.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(report) + "\n")

    hash_targets = [
        ROOT / "analysis/main_model_grouped_cv/decision_support_protocol.md",
        ROOT / "analysis/main_model_grouped_cv/analyze_decision_support.py",
        OUT / "summary.json",
        OUT / "frozen_calibrators.json",
        OUT / "calibration_metrics.csv",
        OUT / "abstention_thresholds.csv",
        OUT / "risk_coverage.csv",
        OUT / "topk_quantitative_utility.csv",
    ]
    manifest = "\n".join(f"{sha256(path)}  {path.relative_to(ROOT)}" for path in hash_targets) + "\n"
    (OUT / "SHA256SUMS").write_text(manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
