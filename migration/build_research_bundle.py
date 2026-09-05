#!/usr/bin/env python3
"""Build a private migration ZIP containing the current research state."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = "SpecTUS-research-2026-09-05"
OUTPUT = ROOT / "dist/spectus_research_migration_2026-09-05.zip"
TOP_LEVEL_CHECKSUMS = ROOT / "dist/MIGRATION_SHA256SUMS"
INCLUDES = (
    "analysis",
    "configs",
    "config_runners",
    "docs",
    "deployment",
    "migration",
    "models/vgwd_clean_rslora_r32_all_attention_seed123_frozen",
    "data/vgwd_main_grouped_cv_locked",
    "predictions/vgwd_main_grouped_cv",
    "checkpoints/vgwd_main_grouped_cv_fold0_qv_seed123",
    "checkpoints/vgwd_main_grouped_cv_fold0_allattn_seed123",
    "checkpoints/vgwd_main_grouped_cv_fold1_qv_seed123",
    "checkpoints/vgwd_main_grouped_cv_fold1_allattn_seed123",
    "checkpoints/vgwd_main_grouped_cv_fold2_qv_seed123",
    "checkpoints/vgwd_main_grouped_cv_fold2_allattn_seed123",
    "checkpoints/vgwd_main_grouped_cv_fold3_qv_seed123",
    "checkpoints/vgwd_main_grouped_cv_fold3_allattn_seed123",
    "checkpoints/vgwd_main_grouped_cv_fold4_qv_seed123",
    "checkpoints/vgwd_main_grouped_cv_fold4_allattn_seed123",
    "spectus",
    "tokenizer",
    "README.md",
    "README_KO.md",
    "CITATION.cff",
    ".gitignore",
    ".gitmodules",
    "codex-conversation-2026-09-04.md",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".swp", ".tmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    files: set[Path] = set()
    for relative in INCLUDES:
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"Required migration path is missing: {relative}")
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            relative_candidate = candidate.relative_to(ROOT)
            if any(part in EXCLUDED_PARTS for part in relative_candidate.parts):
                continue
            if candidate.suffix in EXCLUDED_SUFFIXES:
                continue
            files.add(candidate)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def git_state() -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return {"commit": commit, "working_tree_dirty_at_packaging": dirty}


def write_top_level_checksums() -> None:
    names = (
        "spectus_research_migration_2026-09-05.zip",
        "spectus_r32_all_attention_offline_2026-09-02.zip",
        "spectus_r32_all_attention_online_2026-09-02.zip",
    )
    lines = []
    for name in names:
        path = ROOT / "dist" / name
        if not path.is_file():
            raise SystemExit(f"Required transfer artifact is missing: {path}")
        lines.append(f"{sha256(path)}  {name}")
    TOP_LEVEL_CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    files = selected_files()
    manifest_files = {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    }
    manifest = {
        "bundle": "SpecTUS research migration state",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "privacy": "private; contains locked data and model checkpoints",
        "holdout_state": "LOCKED_NOT_EVALUATED",
        "git": git_state(),
        "files": manifest_files,
    }
    with zipfile.ZipFile(
        OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for path in files:
            archive.write(path, f"{BUNDLE_ROOT}/{path.relative_to(ROOT).as_posix()}")
        archive.writestr(
            f"{BUNDLE_ROOT}/MIGRATION_BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
    write_top_level_checksums()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "sha256": sha256(OUTPUT),
                "files": len(files) + 1,
                "holdout_state": "LOCKED_NOT_EVALUATED",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
