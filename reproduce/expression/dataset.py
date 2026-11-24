# SPDX-License-Identifier: AGPL-3.0-only
# Extracted/adapted for standalone Core CV, 2026-11-24. No product service code.
"""FER2013 CSV validation and isolation of exact/mirrored duplicates."""
from __future__ import annotations

import csv
from collections import defaultdict
from hashlib import file_digest, sha256
from pathlib import Path

import numpy as np

SPLITS = {"Training": 28709, "PublicTest": 3589, "PrivateTest": 3589}


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def audit_rows(path: Path):
    groups = defaultdict(list)
    raw_counts = {split: [0] * 7 for split in SPLITS}
    invalid = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None or set(rows.fieldnames) != {"emotion", "pixels", "Usage"} or len(rows.fieldnames) != 3:
            raise ValueError("Expected the official FER2013 emotion,pixels,Usage columns")
        for line, row in enumerate(rows, start=2):
            label, raw, split = row["emotion"], row["pixels"], row["Usage"]
            if None in row or label not in tuple("0123456") or split not in SPLITS or raw is None:
                invalid.append({"line": line, "reason": "invalid_label_or_split"})
                continue
            tokens = raw.split()
            if len(tokens) != 2304 or any(not token.isascii() or not token.isdecimal() or len(token) > 3 for token in tokens):
                invalid.append({"line": line, "reason": "invalid_pixel_encoding"})
                continue
            pixels = np.array(tokens, dtype=np.int16).reshape(48, 48)
            if np.any(pixels > 255):
                invalid.append({"line": line, "reason": "pixel_out_of_range"})
                continue
            pixels = pixels.astype(np.uint8)
            key = sha256(min(pixels.tobytes(), pixels[:, ::-1].tobytes())).digest()
            raw_counts[split][int(label)] += 1
            groups[key].append((split, int(label), pixels, line))

    kept = {split: [] for split in SPLITS}
    excluded = {"cross_split_rows": 0, "conflicting_label_rows": 0, "within_split_duplicate_rows": 0}
    for group in groups.values():
        split, label, pixels, line = group[0]
        if any(item[0] != split for item in group):
            excluded["cross_split_rows"] += len(group)
        elif any(item[1] != label for item in group):
            excluded["conflicting_label_rows"] += len(group)
        else:
            excluded["within_split_duplicate_rows"] += len(group) - 1
            kept[split].append((pixels, label, line))
    remaining = {
        split: np.bincount([label for _, label, _ in rows], minlength=7).tolist()
        for split, rows in kept.items()
    }
    return kept, {"valid_before_dedup_by_class": raw_counts, "kept_by_class": remaining, "removed": excluded,
                  "invalid_rows": invalid, "original_sample_id": "CSV physical line number starting at line 2",
                  "subject_disjoint_verifiable": False}
