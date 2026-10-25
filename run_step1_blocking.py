"""
STEP 1: Blocking / candidate generation -- run this first.

This does NOT train any model. It just:
  1. Loads your train data (S1, S2, S3, ground truth)
  2. Runs multi-channel blocking to generate candidates for every S1 entity
  3. Reports the two numbers the challenge asks you to optimize:
       - recall ceiling (what fraction of true matches are even reachable
         by your candidate set -- this is your upper bound on final score)
       - average candidates per S1 entity (smaller is better once recall
         is high enough -- graded separately per the challenge PDF)
  4. Prints a recall-vs-K tradeoff table so you can tune
     config.MAX_CANDIDATES_PER_S1 with real numbers instead of guessing
  5. Writes dataset/train candidates to output/candidate_pairs_train_debug.tsv
     (NOT the real submission file -- that only gets written by
     src/predict.py on the TEST set)

Run:
    python3 run_step1_blocking.py

If this is slow on the full ~2.2M S1 / ~10.3M S2+S3 rows, see the
"Known limitations" section of README.md for parallelization tips. As a
reference point: on a single CPU core, indexing ~700k S2+S3 records took
about 45s and scoring 30k S1 entities against them took about 4 minutes
(roughly linear in both dimensions) -- so budget accordingly for the full
dataset, or start by testing on a random subset the way this script's
--sample flag lets you.
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config
from src.blocking import generate_all_candidates
from src.io_utils import read_ground_truth, read_source, write_pairs_tsv
from src.metrics import recall_at_k_blocking

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run blocking on the train set and report quality metrics.")
    parser.add_argument("--sample", type=int, default=None,
                         help="If set, randomly sample this many S1 entities instead of using all of them "
                              "(useful for a quick first run before committing to the full dataset).")
    parser.add_argument("--s23-sample", type=int, default=None,
                         help="If set, ALSO subsample the S2/S3 corpus down to roughly this many records "
                              "each (per source), while always keeping every true-match record for the "
                              "sampled S1 entities so recall stays meaningful. IMPORTANT: index-building "
                              "time is driven by the size of S2+S3, NOT by --sample -- with the full "
                              "~10.3M S2+S3 records, indexing alone can take 15-25+ minutes on a laptop "
                              "EVERY run, regardless of how few S1 entities you're testing. Use --s23-sample "
                              "(e.g. --s23-sample 300000) for fast iteration while developing/tuning, and "
                              "drop it (use the full corpus) only for your final, real run.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    logger.info("Loading train data from %s ...", config.TRAIN_DIR)
    s1_df = read_source(config.TRAIN_S1)
    s2_df = read_source(config.TRAIN_S2)
    s3_df = read_source(config.TRAIN_S3)
    gt_long = read_ground_truth(config.TRAIN_GT)

    if args.sample:
        random.seed(args.seed)
        ids = s1_df["entity_id"].tolist()
        keep = set(random.sample(ids, min(args.sample, len(ids))))
        s1_df = s1_df[s1_df["entity_id"].isin(keep)].reset_index(drop=True)
        logger.info("Sampled down to %d S1 entities (--sample %d)", len(s1_df), args.sample)

    if args.s23_sample:
        random.seed(args.seed)
        # Figure out which S2/S3 ids are true matches for the S1 entities
        # we're actually going to score, so subsampling never accidentally
        # removes a reachable true match (which would understate recall).
        needed_ids = set()
        gt_for_sample = gt_long[gt_long["source1_entity_id"].isin(set(s1_df["entity_id"]))]
        needed_ids.update(gt_for_sample["matched_entity_id"].dropna())

        def _subsample(df, target):
            needed_mask = df["entity_id"].isin(needed_ids)
            kept = df[needed_mask]
            decoy_pool = df[~needed_mask]
            n_decoys_needed = max(0, target - len(kept))
            if n_decoys_needed > 0 and len(decoy_pool) > 0:
                decoy_sample = decoy_pool.sample(
                    n=min(n_decoys_needed, len(decoy_pool)), random_state=args.seed
                )
                return pd.concat([kept, decoy_sample]).reset_index(drop=True)
            return kept.reset_index(drop=True)

        s2_df = _subsample(s2_df, args.s23_sample)
        s3_df = _subsample(s3_df, args.s23_sample)
        logger.info(
            "Subsampled S2/S3 to %d / %d records each (--s23-sample %d, true matches always kept)",
            len(s2_df), len(s3_df), args.s23_sample,
        )
        logger.warning(
            "NOTE: recall numbers below are measured against this SUBSAMPLED S2/S3 pool, not the full "
            "corpus. They're useful for iterating quickly, but re-check on the full corpus (no "
            "--s23-sample) before trusting the numbers for your final tuning decision."
        )

    # Fast single-pass build instead of iterating a pandas groupby object
    # row-group-by-row-group -- with ~2.2M distinct S1 entities in the full
    # ground truth, that groupby-iterate pattern is a well-known pandas
    # performance trap (it materializes a DataFrame slice per group) and
    # can take many minutes on its own, regardless of --sample size.
    truths: dict[str, set] = defaultdict(set)
    for s1_id, mid in zip(gt_long["source1_entity_id"].to_numpy(), gt_long["matched_entity_id"].to_numpy()):
        if pd.notna(mid):
            truths[s1_id].add(mid)
    truths = dict(truths)
    all_s1_ids = s1_df["entity_id"].tolist()
    for s1_id in all_s1_ids:
        truths.setdefault(s1_id, set())

    logger.info("Running blocking: %d S1 entities against %d S2 + %d S3 records ...",
                len(s1_df), len(s2_df), len(s3_df))
    t0 = time.time()
    candidates = generate_all_candidates(s1_df, s2_df, s3_df)
    elapsed = time.time() - t0
    logger.info("Blocking finished in %.1fs (%.2f ms/S1 entity)", elapsed, elapsed / len(s1_df) * 1000)

    avg_c = np.mean([len(v) for v in candidates.values()])
    n_zero = sum(1 for v in candidates.values() if len(v) == 0)
    recall_full = recall_at_k_blocking(candidates, truths, all_s1_ids)

    print("\n" + "=" * 60)
    print("BLOCKING QUALITY REPORT")
    print("=" * 60)
    print(f"S1 entities processed:        {len(s1_df):,}")
    print(f"Avg candidates per S1:        {avg_c:.1f}  (cap = {config.MAX_CANDIDATES_PER_S1})")
    print(f"S1 entities with 0 candidates:{n_zero:,}")
    print(f"RECALL CEILING (at cap):      {recall_full:.4f}")
    print()
    print("Recall vs. candidate-count cutoff (using the already-ranked lists --")
    print("this tells you how much of your recall would survive a SMALLER cap,")
    print("which the challenge explicitly rewards):")
    for K in [5, 10, 25, 50, 75, 100, config.MAX_CANDIDATES_PER_S1]:
        total_true, total_found = 0, 0
        for s1_id, truth in truths.items():
            if not truth or s1_id not in candidates:
                continue
            cand_set = set(candidates[s1_id][:K])
            total_true += len(truth)
            total_found += len(truth & cand_set)
        r = total_found / total_true if total_true else 1.0
        print(f"  K={K:4d}  recall={r:.4f}")
    print("=" * 60)

    debug_path = config.OUTPUT_DIR / "candidate_pairs_train_debug.tsv"
    write_pairs_tsv(candidates, all_s1_ids, debug_path, "candidate_entity_ids")
    print(f"\nWrote debug candidate file to {debug_path}")
    print("(This is for your own inspection only -- the real candidate_pairs.tsv")
    print(" for submission is produced by src/predict.py on the TEST set.)")


if __name__ == "__main__":
    main()
