"""
I/O helpers: reading the challenge's TSV files and writing the two
required output files in the exact format the validator expects.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: Path) -> pd.DataFrame:
    """Read one of the source TSVs (S1/S2/S3), tab-separated, and do light
    cleanup: strip whitespace, coerce to string, fill any residual NaNs in
    text columns with empty string (empty business_address is expected --
    see dataset_report.txt: ~168-176k rows per source have it)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(REQUIRED_SOURCE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    for col in ["business_name", "business_address", "country"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    if df["entity_id"].duplicated().any():
        n_dup = df["entity_id"].duplicated().sum()
        logger.warning("%s: %d duplicate entity_id values found", path, n_dup)
    return df.reset_index(drop=True)


def read_ground_truth(path: Path) -> pd.DataFrame:
    """Read train_ground_truth.tsv into a long-format DataFrame:
    one row per (source1_entity_id, matched_entity_id) pair, plus a
    column noting whether the S1 entity is a singleton (no matches)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["source1_entity_id"] = df["source1_entity_id"].astype(str).str.strip()
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str)

    records = []
    for s1_id, raw in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        raw = raw.strip()
        if raw == "":
            records.append((s1_id, None))
            continue
        for mid in raw.split(","):
            mid = mid.strip()
            if mid:
                records.append((s1_id, mid))
    long_df = pd.DataFrame(records, columns=["source1_entity_id", "matched_entity_id"])
    return long_df


def parse_id_list(raw: str) -> list[str]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if raw == "":
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def write_pairs_tsv(
    s1_to_ids: dict[str, Iterable[str]],
    all_s1_ids: Iterable[str],
    out_path: Path,
    id_col_name: str,
) -> None:
    """Write a candidate_pairs.tsv / matching_results.tsv style file.
    Every S1 id in `all_s1_ids` gets exactly one row, in that order.
    IDs within each row are de-duplicated but order-preserving.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{id_col_name}\n")
        for s1_id in all_s1_ids:
            ids = s1_to_ids.get(s1_id, [])
            seen = set()
            deduped = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    deduped.append(i)
            f.write(f"{s1_id}\t{','.join(deduped)}\n")
    logger.info("Wrote %s (%d rows)", out_path, len(list(all_s1_ids)) if not isinstance(all_s1_ids, list) else len(all_s1_ids))
