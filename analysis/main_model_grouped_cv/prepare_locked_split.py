#!/usr/bin/env python3
"""Create or verify the prospectively locked internal VGWD split.

This script deliberately has no option to overwrite an existing split.  The
locked holdout is never used by the grouped-CV runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/vgwd_main_grouped_cv_locked"
SOURCES = {
    "train": ROOT / "data/vgwd_clean/train.jsonl",
    "valid": ROOT / "data/vgwd_clean/valid.jsonl",
    "test": ROOT / "data/vgwd_clean/test.jsonl",
}
SPLIT_SEED = 20260904
HOLDOUT_FRACTION = 0.20
N_FOLDS = 5
MAX_MZ = 500
MAX_PEAKS = 300
MAX_SMILES_LENGTH = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_smiles(smiles: str, *, stereo: bool) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if not stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=stereo)


def spectrum_hash(row: dict[str, Any]) -> str:
    peaks = sorted(
        (round(float(mz), 6), round(float(intensity), 6))
        for mz, intensity in zip(row["mz"], row["intensity"])
    )
    payload = json.dumps(peaks, separators=(",", ":"))
    return stable_hash(payload)


def eligibility(row: dict[str, Any]) -> tuple[bool, str, str, str]:
    if not {"smiles", "mz", "intensity"}.issubset(row):
        return False, "missing_required_field", "", ""
    if not isinstance(row["mz"], list) or not isinstance(row["intensity"], list):
        return False, "invalid_peak_arrays", "", ""
    if not row["mz"] or len(row["mz"]) != len(row["intensity"]):
        return False, "empty_or_mismatched_peaks", "", ""
    exact = canonical_smiles(str(row["smiles"]), stereo=True)
    connectivity = canonical_smiles(str(row["smiles"]), stereo=False)
    if exact is None or connectivity is None or not connectivity.strip():
        return False, "invalid_smiles", "", ""
    if len(connectivity) > MAX_SMILES_LENGTH:
        return False, "smiles_too_long", exact, connectivity
    if max(float(value) for value in row["mz"]) > MAX_MZ:
        return False, "mz_above_500", exact, connectivity
    peak_count = len(row["mz"]) - (1 if float(row["mz"][0]) == 0 else 0)
    if peak_count > MAX_PEAKS:
        return False, "too_many_peaks", exact, connectivity
    return True, "eligible", exact, connectivity


def load_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source_split, path in SOURCES.items():
        with path.open(encoding="utf-8") as handle:
            for source_row, line in enumerate(handle):
                row = json.loads(line)
                eligible, reason, exact, connectivity = eligibility(row)
                record_digest = stable_hash(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                )
                metadata = {
                    "source_split": source_split,
                    "source_row": source_row,
                    "record_sha256": record_digest,
                    "group_sha256": stable_hash(connectivity) if connectivity else "",
                    "spectrum_sha256": spectrum_hash(row)
                    if "mz" in row and "intensity" in row
                    else "",
                }
                if not eligible:
                    excluded.append({**metadata, "reason": reason})
                    continue
                records.append(
                    {
                        **metadata,
                        "row": row,
                        "exact": exact,
                        "connectivity": connectivity,
                    }
                )
    return records, excluded


def choose_holdout(records: list[dict[str, Any]]) -> set[str]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["connectivity"]].append(record)
    ordered_groups = sorted(
        by_group,
        key=lambda group: stable_hash(f"holdout:{SPLIT_SEED}:{group}"),
    )
    target_rows = math.ceil(len(records) * HOLDOUT_FRACTION)
    chosen: set[str] = set()
    chosen_rows = 0
    for group in ordered_groups:
        if chosen_rows >= target_rows:
            break
        chosen.add(group)
        chosen_rows += len(by_group[group])
    return chosen


def assign_folds(records: list[dict[str, Any]], holdout: set[str]) -> dict[str, int]:
    group_sizes = Counter(
        record["connectivity"]
        for record in records
        if record["connectivity"] not in holdout
    )
    ordered = sorted(
        group_sizes,
        key=lambda group: (
            -group_sizes[group],
            stable_hash(f"fold:{SPLIT_SEED}:{group}"),
        ),
    )
    fold_rows = [0] * N_FOLDS
    assignment: dict[str, int] = {}
    for group in ordered:
        fold = min(range(N_FOLDS), key=lambda value: (fold_rows[value], value))
        assignment[group] = fold
        fold_rows[fold] += group_sizes[group]
    return assignment


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record["row"], ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Locked split already exists: {OUTPUT}. Use --verify; overwrite is disabled."
        )
    records, excluded = load_records()
    holdout_groups = choose_holdout(records)
    fold_by_group = assign_folds(records, holdout_groups)

    dev = [r for r in records if r["connectivity"] not in holdout_groups]
    holdout = [r for r in records if r["connectivity"] in holdout_groups]
    OUTPUT.mkdir(parents=True)
    write_jsonl(OUTPUT / "dev.jsonl", dev)
    write_jsonl(OUTPUT / "holdout_LOCKED.jsonl", holdout)

    for fold in range(N_FOLDS):
        fold_train = [r for r in dev if fold_by_group[r["connectivity"]] != fold]
        fold_oof = [r for r in dev if fold_by_group[r["connectivity"]] == fold]
        write_jsonl(OUTPUT / f"fold_{fold}/train.jsonl", fold_train)
        write_jsonl(OUTPUT / f"fold_{fold}/oof.jsonl", fold_oof)

    assignments = []
    for record in records:
        is_holdout = record["connectivity"] in holdout_groups
        assignments.append(
            {
                "record_sha256": record["record_sha256"],
                "group_sha256": record["group_sha256"],
                "spectrum_sha256": record["spectrum_sha256"],
                "source_split": record["source_split"],
                "source_row": record["source_row"],
                "partition": "holdout_LOCKED" if is_holdout else "development",
                "oof_fold": "" if is_holdout else fold_by_group[record["connectivity"]],
            }
        )
    write_csv(
        OUTPUT / "assignments.csv",
        assignments,
        [
            "record_sha256",
            "group_sha256",
            "spectrum_sha256",
            "source_split",
            "source_row",
            "partition",
            "oof_fold",
        ],
    )
    write_csv(
        OUTPUT / "excluded.csv",
        excluded,
        [
            "source_split",
            "source_row",
            "record_sha256",
            "group_sha256",
            "spectrum_sha256",
            "reason",
        ],
    )

    fold_summary = []
    for fold in range(N_FOLDS):
        oof = [r for r in dev if fold_by_group[r["connectivity"]] == fold]
        train = [r for r in dev if fold_by_group[r["connectivity"]] != fold]
        fold_summary.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "train_groups": len({r["connectivity"] for r in train}),
                "oof_rows": len(oof),
                "oof_groups": len({r["connectivity"] for r in oof}),
            }
        )
    summary = {
        "protocol_name": "prospectively_locked_internal_resplit_grouped_cv",
        "created_date": "2026-09-04",
        "split_seed": SPLIT_SEED,
        "group_definition": "RDKit canonical connectivity SMILES without stereochemistry",
        "holdout_target_fraction": HOLDOUT_FRACTION,
        "eligible_rows": len(records),
        "eligible_groups": len({r["connectivity"] for r in records}),
        "excluded_rows": len(excluded),
        "development_rows": len(dev),
        "development_groups": len({r["connectivity"] for r in dev}),
        "holdout_rows": len(holdout),
        "holdout_groups": len(holdout_groups),
        "holdout_row_fraction": len(holdout) / len(records),
        "source_rows": dict(Counter(r["source_split"] for r in records)),
        "development_source_rows": dict(Counter(r["source_split"] for r in dev)),
        "holdout_source_rows": dict(Counter(r["source_split"] for r in holdout)),
        "folds": fold_summary,
        "guards": {
            "holdout_evaluation_allowed": False,
            "holdout_predictions_created": False,
            "same_connectivity_group_cross_partition": 0,
            "same_connectivity_group_cross_fold": 0,
        },
        "limitation": (
            "All source splits had prior experimental roles. This is a prospectively "
            "locked internal re-split, not a never-observed external validation set."
        ),
    }
    (OUTPUT / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    generated = sorted(
        path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    output_hashes = {
        path.relative_to(OUTPUT).as_posix(): sha256_file(path) for path in generated
    }
    lock = {
        "protocol_name": summary["protocol_name"],
        "created_date": summary["created_date"],
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in SOURCES.values()
        },
        "output_sha256_before_lock": output_hashes,
        "split_parameters": {
            "seed": SPLIT_SEED,
            "holdout_fraction": HOLDOUT_FRACTION,
            "folds": N_FOLDS,
            "group": summary["group_definition"],
        },
        "evaluation_state": "LOCKED_NOT_EVALUATED",
    }
    lock_path = OUTPUT / "LOCK.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    output_hashes["LOCK.json"] = sha256_file(lock_path)
    with (OUTPUT / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for relative, digest in sorted(output_hashes.items()):
            handle.write(f"{digest}  {relative}\n")
    os.chmod(OUTPUT / "holdout_LOCKED.jsonl", 0o444)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Created locked split: {OUTPUT.relative_to(ROOT)}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def connectivity_groups(path: Path) -> set[str]:
    groups = set()
    for row in read_jsonl(path):
        value = canonical_smiles(str(row["smiles"]), stereo=False)
        if value is None:
            raise ValueError(f"Invalid SMILES in locked output: {path}")
        groups.add(value)
    return groups


def verify() -> None:
    lock_path = OUTPUT / "LOCK.json"
    checksums_path = OUTPUT / "SHA256SUMS"
    if not lock_path.is_file() or not checksums_path.is_file():
        raise FileNotFoundError("LOCK.json or SHA256SUMS is missing")
    lock = json.loads(lock_path.read_text())
    for relative, expected in lock["source_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Source hash mismatch: {relative}")
    for line in checksums_path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = OUTPUT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Locked output hash mismatch: {relative}")

    holdout_groups = connectivity_groups(OUTPUT / "holdout_LOCKED.jsonl")
    dev_groups = connectivity_groups(OUTPUT / "dev.jsonl")
    if holdout_groups & dev_groups:
        raise RuntimeError("Connectivity group overlap between development and holdout")
    oof_groups: list[set[str]] = []
    for fold in range(N_FOLDS):
        train_groups = connectivity_groups(OUTPUT / f"fold_{fold}/train.jsonl")
        fold_groups = connectivity_groups(OUTPUT / f"fold_{fold}/oof.jsonl")
        if train_groups & fold_groups:
            raise RuntimeError(f"Connectivity leakage in fold {fold}")
        if train_groups | fold_groups != dev_groups:
            raise RuntimeError(f"Fold {fold} does not cover development groups")
        oof_groups.append(fold_groups)
    if set().union(*oof_groups) != dev_groups:
        raise RuntimeError("OOF folds do not cover every development group")
    if sum(len(groups) for groups in oof_groups) != len(dev_groups):
        raise RuntimeError("A connectivity group appears in multiple OOF folds")
    summary = json.loads((OUTPUT / "split_summary.json").read_text())
    print(
        json.dumps(
            {
                "verification": "PASS",
                "evaluation_state": lock["evaluation_state"],
                "development_rows": summary["development_rows"],
                "holdout_rows": summary["holdout_rows"],
                "fold_oof_rows": [item["oof_rows"] for item in summary["folds"]],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    create() if args.create else verify()


if __name__ == "__main__":
    main()
