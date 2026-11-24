# SPDX-License-Identifier: AGPL-3.0-only
# Extracted/adapted for standalone Core CV, 2026-11-24. No product service code.
"""Audit then freeze an authorized, local copy of FER2013."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import numpy as np

from runtime import CONTRACT, LABELS, SIDE
from dataset import SPLITS, audit_rows, sha_file

BYTE_LIMIT = 400 * 1024 * 1024
SOURCE = "https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data"
TERMS = SOURCE.removesuffix("data") + "rules"
PROVENANCE_FIELDS = ("source_url", "terms_url", "authorized_use", "downloaded_at", "sha256")


def archive_csv(archive: Path, destination: Path) -> dict:
    expanded = 0
    with gzip.open(archive, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            expanded += len(chunk)
            if expanded > BYTE_LIMIT:
                raise ValueError("FER2013 archive exceeds the expansion budget")
    with tarfile.open(archive, "r:gz") as tar:
        entries = []
        seen = set()
        candidates = []
        unexpected = []
        for member in tar:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.name or ":" in member.name
                    or not (member.isfile() or member.isdir()) or path in seen):
                raise ValueError("Archive includes an unsafe or duplicate member")
            seen.add(path)
            if member.isdir():
                continue
            entries.append({"name": str(path), "bytes": member.size})
            if path.name == "fer2013.csv":
                candidates.append(member)
            elif path.name.casefold() not in {"readme", "readme.txt", "fer2013.bib"}:
                unexpected.append(str(path))
        if len(candidates) != 1 or not 0 < candidates[0].size <= BYTE_LIMIT:
            raise ValueError("Archive must contain one bounded fer2013.csv")
        with tar.extractfile(candidates[0]) as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
    return {"format": "tar.gz", "integrity": "read", "expanded_bytes": expanded,
            "entries": entries, "unexpected_files": unexpected}


def validate_provenance(path: Path, raw: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or any(not isinstance(data.get(key), str) or not data[key].strip()
                                         for key in PROVENANCE_FIELDS):
        raise ValueError("Provenance needs source_url, terms_url, authorized_use, downloaded_at, sha256")
    if data["source_url"] != SOURCE or data["terms_url"] != TERMS:
        raise ValueError("This FER2013 recipe expects the official competition source and terms")
    if datetime.fromisoformat(data["downloaded_at"]).utcoffset() is None:
        raise ValueError("Download timestamp must include timezone")
    if not raw.is_file() or not 0 < raw.stat().st_size <= BYTE_LIMIT or data["sha256"].lower() != sha_file(raw):
        raise ValueError("Raw file is missing, oversized or differs from its recorded SHA-256")
    return {key: data[key] for key in PROVENANCE_FIELDS}


def freeze(raw: Path, is_archive: bool, provenance_path: Path, output: Path) -> bool:
    provenance = validate_provenance(provenance_path, raw)
    output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="fer2013-audit-") as temp:
        csv_path = Path(temp) / "fer2013.csv" if is_archive else raw
        archive = archive_csv(raw, csv_path) if is_archive else {"format": None, "integrity": "not_supplied"}
        rows, audit = audit_rows(csv_path)
        csv_sha = sha_file(csv_path)
    ready = (not audit["invalid_rows"] and not archive.get("unexpected_files")
             and all(sum(audit["valid_before_dedup_by_class"][split]) == expected
                     and all(audit["kept_by_class"][split]) for split, expected in SPLITS.items()))
    artifacts = {}
    if ready:
        for split, samples in rows.items():
            pixels, labels, ids = zip(*samples)
            artifact = output / f"{split}.npz"
            np.savez_compressed(artifact, pixels=np.stack(pixels), labels=np.array(labels, dtype=np.int64),
                                ids=np.array(ids, dtype=np.int64))
            artifacts[split] = {"name": artifact.name, "sha256": sha_file(artifact), "bytes": artifact.stat().st_size,
                                "samples": len(samples)}
    manifest = {
        "dataset": "FER2013", "dataset_version": f"fer2013-prep1-{csv_sha[:16]}",
        "created_at": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
        "raw": {"name": raw.name, "bytes": raw.stat().st_size, "sha256": provenance["sha256"].lower()},
        "csv_sha256": csv_sha, "archive": archive, "labels": LABELS, "splits": SPLITS,
        "observed_total_rows": sum(map(sum, audit["valid_before_dedup_by_class"].values())) + len(audit["invalid_rows"]),
        "preprocessing": CONTRACT, "artifacts": artifacts, "training_ready": ready, **audit,
    }
    (output / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return bool(ready)


def load_prepared(folder: Path, *, include_test: bool = True):
    """Load the existing split contract, optionally restricting access to development data."""
    manifest = json.loads((folder / "dataset-manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("training_ready") is not True or manifest.get("labels") != list(LABELS)
            or manifest.get("splits") != SPLITS or manifest.get("preprocessing") != CONTRACT):
        raise ValueError("Prepared data failed its readiness gate")
    rows = {}
    used_ids = set()
    for split in SPLITS:
        if split == "PrivateTest" and not include_test:
            continue
        path = folder / f"{split}.npz"
        record = manifest["artifacts"][split]
        if sha_file(path) != record["sha256"]:
            raise ValueError("Prepared split hash differs from the manifest")
        with np.load(path, allow_pickle=False) as data:
            pixels, labels, ids = data["pixels"], data["labels"], data["ids"]
        size = record["samples"]
        if (pixels.shape != (size, SIDE, SIDE) or pixels.dtype != np.uint8 or labels.shape != (size,)
                or labels.dtype != np.int64 or ids.shape != (size,) or ids.dtype != np.int64
                or not size or labels.min() < 0 or labels.max() >= len(LABELS)
                or np.bincount(labels, minlength=7).tolist() != manifest["kept_by_class"][split]):
            raise ValueError("Invalid prepared split")
        keys = set(ids.tolist())
        if min(keys) < 2 or len(keys) != size or used_ids & keys:
            raise ValueError("Sample ID repeated across prepared splits")
        used_ids.update(keys)
        rows[split] = list(zip(pixels, labels, ids))
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--archive", type=Path)
    inputs.add_argument("--csv", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        ready = freeze(args.archive or args.csv, args.archive is not None, args.provenance, args.out)
    except (OSError, ValueError, tarfile.TarError, EOFError, KeyError) as error:
        parser.error(str(error))
    print(args.out / "dataset-manifest.json")
    if not ready:
        raise SystemExit("Dataset audit found issues. Check the manifest before training.")


if __name__ == "__main__":
    main()
