"""
Train the pairwise matcher.

Pipeline:
  1. Load train S1/S2/S3 + ground truth.
  2. Split S1 entities 80/20 (leakage-free -- entire S1 entities go to one
     side, per dataset_report.txt Section 9).
  3. Run blocking on train S1 to get candidates (or load precomputed ones).
  4. Label each candidate pair: 1 if in ground truth, 0 otherwise. Cap
     negatives per S1 entity at MAX_HARD_NEGATIVES_PER_POSITIVE (keeping
     the highest name-similarity negatives -- these are the informative
     "hard negatives" from dataset_report.txt Section 7) so training
     isn't swamped by millions of easy negatives.
  5. Build features, train LightGBM with early stopping on validation.
  6. Search the decision threshold that maximizes macro F_0.5 on the
     validation split (NOT plain accuracy/F1 -- see the challenge's
     precision-weighted metric).
  7. Save model, feature order, and threshold to models/.

Run:
    python -m src.train
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import config
from .blocking import generate_all_candidates
from .features import PairFeatureBuilder, FEATURE_NAMES
from .io_utils import read_ground_truth, read_source
from .metrics import macro_f_beta, recall_at_k_blocking

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def split_s1_entities(s1_ids: list[str], val_fraction: float, seed: int):
    rng = random.Random(seed)
    ids = list(s1_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def build_labeled_pairs(
    candidates: dict[str, list[str]],
    truths: dict[str, set[str]],
    max_hard_negatives: int,
    name_lookup: dict[str, str],
) -> list[tuple[str, str, int]]:
    """For each S1 entity: keep all true-match candidates as positives, and
    up to `max_hard_negatives` of the highest name-similarity non-matches
    as negatives (informative hard negatives beat random easy negatives)."""
    labeled = []
    for s1_id, cand_ids in candidates.items():
        truth = truths.get(s1_id, set())
        positives = [c for c in cand_ids if c in truth]
        negatives = [c for c in cand_ids if c not in truth]

        for c in positives:
            labeled.append((s1_id, c, 1))

        if negatives:
            s1_name = name_lookup.get(s1_id, "")
            negatives_scored = sorted(
                negatives,
                key=lambda c: fuzz.token_sort_ratio(s1_name, name_lookup.get(c, "")),
                reverse=True,
            )
            for c in negatives_scored[:max_hard_negatives]:
                labeled.append((s1_id, c, 0))
    return labeled


def main():
    parser = argparse.ArgumentParser(description="Train the pairwise entity-resolution matcher.")
    parser.add_argument("--sample", type=int, default=None,
                         help="If set, randomly sample this many S1 entities instead of using all of them "
                              "-- use this to validate the full train pipeline (blocking + features + "
                              "LightGBM + threshold search) in minutes before committing to the real, "
                              "multi-hour full run.")
    parser.add_argument("--s23-sample", type=int, default=None,
                         help="If set, ALSO subsample the S2/S3 corpus down to roughly this many records "
                              "each, keeping every true-match record for the sampled S1 entities so "
                              "recall/training signal stays meaningful. See run_step1_blocking.py's "
                              "--s23-sample for why this matters: index-building time is driven by the "
                              "size of S2+S3, not by --sample, so skipping this leaves the slow part "
                              "untouched.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading train data...")
    s1_df = read_source(config.TRAIN_S1)
    s2_df = read_source(config.TRAIN_S2)
    s3_df = read_source(config.TRAIN_S3)
    gt_long = read_ground_truth(config.TRAIN_GT)

    if args.sample:
        rng = random.Random(args.seed)
        ids = s1_df["entity_id"].tolist()
        keep = set(rng.sample(ids, min(args.sample, len(ids))))
        s1_df = s1_df[s1_df["entity_id"].isin(keep)].reset_index(drop=True)
        logger.info("Sampled down to %d S1 entities (--sample %d)", len(s1_df), args.sample)

    if args.s23_sample:
        truths_for_sample: dict[str, set] = defaultdict(set)
        for s1_id, mid in zip(gt_long["source1_entity_id"].to_numpy(), gt_long["matched_entity_id"].to_numpy()):
            if pd.notna(mid):
                truths_for_sample[s1_id].add(mid)
        needed_ids = set()
        for s1_id in s1_df["entity_id"]:
            needed_ids.update(truths_for_sample.get(s1_id, set()))

        def _subsample(df, target):
            needed_mask = df["entity_id"].isin(needed_ids)
            kept = df[needed_mask]
            decoy_pool = df[~needed_mask]
            n_decoys_needed = max(0, target - len(kept))
            if n_decoys_needed > 0 and len(decoy_pool) > 0:
                decoy_sample = decoy_pool.sample(n=min(n_decoys_needed, len(decoy_pool)), random_state=args.seed)
                return pd.concat([kept, decoy_sample]).reset_index(drop=True)
            return kept.reset_index(drop=True)

        s2_df = _subsample(s2_df, args.s23_sample)
        s3_df = _subsample(s3_df, args.s23_sample)
        logger.info(
            "Subsampled S2/S3 to %d / %d records each (--s23-sample %d, true matches always kept)",
            len(s2_df), len(s3_df), args.s23_sample,
        )
        logger.warning(
            "NOTE: this is a quick-iteration run on subsampled S2/S3. Re-run without --s23-sample "
            "for your real, final model before submitting."
        )

    truths: dict[str, set] = defaultdict(set)
    for s1_id, mid in zip(gt_long["source1_entity_id"].to_numpy(), gt_long["matched_entity_id"].to_numpy()):
        if pd.notna(mid):
            truths[s1_id].add(mid)
    truths = dict(truths)
    all_s1_ids = s1_df["entity_id"].tolist()
    for s1_id in all_s1_ids:
        truths.setdefault(s1_id, set())

    train_ids, val_ids = split_s1_entities(all_s1_ids, config.VAL_FRACTION, config.RANDOM_SEED)
    logger.info("Train S1: %d, Val S1: %d", len(train_ids), len(val_ids))

    logger.info("Running blocking on train S1 set...")
    candidates = generate_all_candidates(s1_df, s2_df, s3_df)

    recall = recall_at_k_blocking(candidates, truths, all_s1_ids)
    avg_candidates = np.mean([len(v) for v in candidates.values()])
    logger.info(
        "Blocking quality: recall_ceiling=%.4f, avg_candidates_per_s1=%.1f",
        recall, avg_candidates,
    )

    from .normalize import normalize_name
    name_lookup = {row.entity_id: normalize_name(row.business_name) for row in s1_df.itertuples(index=False)}
    for df in (s2_df, s3_df):
        for row in df.itertuples(index=False):
            name_lookup[row.entity_id] = normalize_name(row.business_name)

    logger.info("Building labeled pairs...")
    labeled_all = build_labeled_pairs(candidates, truths, config.MAX_HARD_NEGATIVES_PER_POSITIVE, name_lookup)
    labeled_train = [(s1, c, y) for s1, c, y in labeled_all if s1 in train_ids]
    labeled_val = [(s1, c, y) for s1, c, y in labeled_all if s1 in val_ids]
    logger.info("Labeled pairs -- train: %d, val: %d", len(labeled_train), len(labeled_val))

    logger.info("Building feature matrices...")
    fb = PairFeatureBuilder(s1_df, s2_df, s3_df, candidates)

    train_pairs = [(s1, c) for s1, c, _ in labeled_train]
    val_pairs = [(s1, c) for s1, c, _ in labeled_val]
    X_train = fb.build_feature_matrix(train_pairs)
    y_train = np.array([y for _, _, y in labeled_train])
    X_val = fb.build_feature_matrix(val_pairs)
    y_val = np.array([y for _, _, y in labeled_val])

    logger.info("Training LightGBM...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)

    booster = lgb.train(
        config.LGBM_PARAMS,
        dtrain,
        num_boost_round=config.LGBM_NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(50),
        ],
    )

    booster.save_model(str(config.MODEL_PATH))
    with open(config.FEATURE_LIST_PATH, "w") as f:
        json.dump(FEATURE_NAMES, f)

    logger.info("Searching F_0.5-optimal threshold on validation set...")
    val_scores = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_df = pd.DataFrame({
        "s1_id": [s1 for s1, _, _ in labeled_val],
        "cand_id": [c for _, c, _ in labeled_val],
        "score": val_scores,
    })

    best_threshold, best_f = None, -1.0
    val_id_list = list(val_ids)
    for t in config.THRESHOLD_GRID:
        preds: dict[str, set] = {}
        kept = val_df[val_df["score"] >= t]
        for s1_id, group in kept.groupby("s1_id"):
            preds[s1_id] = set(group["cand_id"])
        score = macro_f_beta(preds, {k: v for k, v in truths.items() if k in val_ids}, val_id_list, beta=config.F_BETA)
        if score > best_f:
            best_f = score
            best_threshold = t

    logger.info("Best threshold=%.3f -> validation macro F_0.5=%.4f", best_threshold, best_f)
    with open(config.THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": best_threshold, "val_f0_5": best_f}, f)

    logger.info("Done. Model saved to %s", config.MODEL_PATH)


if __name__ == "__main__":
    main()
