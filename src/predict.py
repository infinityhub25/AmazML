"""
Generate predictions for the test set.

Produces both required output files:
  - output/candidate_pairs.tsv  : the exact candidate set fed to the model
  - output/matching_results.tsv : final matches after thresholding

Run:
    python -m src.predict
"""
from __future__ import annotations

import json
import logging

import lightgbm as lgb
import pandas as pd

from . import config
from .blocking import generate_all_candidates
from .features import PairFeatureBuilder
from .io_utils import read_source, write_pairs_tsv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading test data...")
    s1_df = read_source(config.TEST_S1)
    s2_df = read_source(config.TEST_S2)
    s3_df = read_source(config.TEST_S3)
    all_s1_ids = s1_df["entity_id"].tolist()

    logger.info("Note: test set includes France, which does not appear in "
                "training. Blocking and features are country-agnostic in "
                "their logic (no hardcoded {US, India} branches), so France "
                "rows are handled the same way as any other country.")

    logger.info("Running blocking on test S1 set...")
    candidates = generate_all_candidates(s1_df, s2_df, s3_df)

    logger.info("Writing candidate_pairs.tsv...")
    write_pairs_tsv(candidates, all_s1_ids, config.CANDIDATE_PAIRS_PATH, "candidate_entity_ids")

    logger.info("Loading trained model...")
    booster = lgb.Booster(model_file=str(config.MODEL_PATH))
    with open(config.FEATURE_LIST_PATH) as f:
        feature_names = json.load(f)
    with open(config.THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]
    logger.info("Using decision threshold=%.3f", threshold)

    logger.info("Building features for all candidate pairs...")
    fb = PairFeatureBuilder(s1_df, s2_df, s3_df, candidates)

    all_pairs = []
    for s1_id, cand_ids in candidates.items():
        for c in cand_ids:
            all_pairs.append((s1_id, c))

    matches: dict[str, list[str]] = {}
    chunk_size = 200_000
    for start in range(0, len(all_pairs), chunk_size):
        chunk = all_pairs[start:start + chunk_size]
        X = fb.build_feature_matrix(chunk)
        X = X[feature_names]
        scores = booster.predict(X)
        for (s1_id, cand_id), score in zip(chunk, scores):
            if score >= threshold:
                matches.setdefault(s1_id, []).append(cand_id)
        logger.info("Scored %d / %d candidate pairs", min(start + chunk_size, len(all_pairs)), len(all_pairs))

    logger.info("Writing matching_results.tsv...")
    write_pairs_tsv(matches, all_s1_ids, config.MATCHING_RESULTS_PATH, "matched_entity_ids")

    logger.info("Done. Outputs in %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
