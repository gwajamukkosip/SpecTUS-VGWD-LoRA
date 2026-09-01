#!/usr/bin/env python3
"""Audit exact molecular-structure hits at each probability-ranked Top-K."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare probability-ranked SMILES predictions with JSONL labels and "
            "report cumulative exact-structure accuracy at every K."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: directory containing predictions.jsonl)",
    )
    parser.add_argument(
        "--ignore-stereochemistry",
        action="store_true",
        help=(
            "Compare connectivity only by removing stereochemistry "
            "(default keeps stereo, matching evaluate_predictions.py)"
        ),
    )
    return parser.parse_args()


def canonicalize(smiles: str, ignore_stereochemistry: bool) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if ignore_stereochemistry:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=not ignore_stereochemistry
    )


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return rows


def main() -> None:
    args = parse_args()
    if args.max_k < 1:
        raise ValueError("--max-k must be at least 1")

    predictions = read_jsonl(args.predictions)
    labels = read_jsonl(args.labels)
    if len(predictions) != len(labels):
        raise ValueError(
            f"Row count mismatch: {len(predictions)} predictions vs {len(labels)} labels"
        )

    output_dir = args.output_dir or args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "strict_exact_topk_samples.csv"
    summary_path = output_dir / "strict_exact_topk.json"

    hits_at_k = [0] * args.max_k
    invalid_predictions = 0
    invalid_labels = 0
    samples: list[dict[str, Any]] = []

    for sample_id, (prediction, label) in enumerate(zip(predictions, labels)):
        if not isinstance(prediction, dict):
            raise TypeError(f"Prediction row {sample_id} is not a JSON object")
        if not isinstance(label, dict) or "smiles" not in label:
            raise TypeError(f"Label row {sample_id} has no 'smiles' field")

        true_smiles = str(label["smiles"])
        true_canonical = canonicalize(true_smiles, args.ignore_stereochemistry)
        if true_canonical is None:
            invalid_labels += 1

        ranked = sorted(prediction.items(), key=lambda item: float(item[1]), reverse=True)
        exact_rank: int | None = None
        top1_smiles = ranked[0][0] if ranked else ""
        top1_canonical: str | None = None

        for rank, (predicted_smiles, _) in enumerate(ranked[: args.max_k], start=1):
            predicted_canonical = canonicalize(
                str(predicted_smiles), args.ignore_stereochemistry
            )
            if predicted_canonical is None:
                invalid_predictions += 1
                continue
            if rank == 1:
                top1_canonical = predicted_canonical
            if exact_rank is None and predicted_canonical == true_canonical:
                exact_rank = rank

        if exact_rank is not None:
            for index in range(exact_rank - 1, args.max_k):
                hits_at_k[index] += 1

        samples.append(
            {
                "sample_id": sample_id,
                "true_smiles": true_smiles,
                "true_canonical": true_canonical or "",
                "top1_smiles": top1_smiles,
                "top1_canonical": top1_canonical or "",
                "exact_rank": exact_rank or "",
                "candidate_count": len(ranked),
            }
        )

    total = len(predictions)
    summary = {
        "predictions": str(args.predictions.resolve()),
        "labels": str(args.labels.resolve()),
        "normalization": (
            "canonical_connectivity_smiles_without_stereochemistry"
            if args.ignore_stereochemistry
            else "canonical_isomeric_smiles"
        ),
        "num_samples": total,
        "max_k": args.max_k,
        "invalid_label_count": invalid_labels,
        "invalid_prediction_count_in_top_k": invalid_predictions,
        "hits_at_k": {str(k): hits_at_k[k - 1] for k in range(1, args.max_k + 1)},
        "accuracy_at_k": {
            str(k): hits_at_k[k - 1] / total if total else 0.0
            for k in range(1, args.max_k + 1)
        },
    }

    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=samples[0].keys() if samples else [])
        if samples:
            writer.writeheader()
            writer.writerows(samples)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote sample audit: {sample_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
