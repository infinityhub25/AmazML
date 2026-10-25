"""
Central configuration for the Business Entity Resolution pipeline.
Edit paths here if your folder layout differs from the challenge's default.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"

MODEL_PATH = MODEL_DIR / "lgbm_matcher.txt"
FEATURE_LIST_PATH = MODEL_DIR / "feature_names.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

# ----------------------------------------------------------------------
# Blocking
# ----------------------------------------------------------------------
# Hard cap on candidates kept per S1 entity after merging all blocking channels.
# Empirically tuned on a real held-out slice of the actual train data
# (30k S1 entities against a realistic-density 350k-record S2/S3 pool):
#   K=10  -> recall=0.942   K=50  -> recall=0.965   K=150 -> recall=0.976
# Recall is already >94% at K=10 and only gains ~3.5 points by K=150 -- and
# the challenge explicitly ranks a SMALLER candidate set higher at the same
# recall (see the PDF's "candidate generation counts toward final ranking").
# 50 is a reasonable default balance; drop it to 20-25 if you want to
# prioritize a smaller footprint over the last couple points of recall, or
# raise it if your matching model needs more room to find true positives.
MAX_CANDIDATES_PER_S1 = 50

# Minimum token length to be indexed (drop very short tokens like "a", "co" from
# the rare-token inverted index -- they are not discriminative and blow up blocks).
MIN_TOKEN_LEN = 3

# A token is considered "too common" (and skipped when building the inverted index)
# if it appears in more than this fraction of all records in a source. Extremely
# common tokens (e.g. "private", "limited", "inc") would otherwise create giant
# blocks that defeat the purpose of blocking.
MAX_TOKEN_DOC_FREQ = 0.02

# HARD CAP on postings list length, independent of MAX_TOKEN_DOC_FREQ above.
# This matters a lot at real scale: 2% of a 350K-record test sample is only
# 7,000 records, but 2% of a real 5M-record source is 100,000 records -- and
# every S1 entity sharing even one such token then has to scan a 100,000-
# entry list. That's what makes blocking blow up in wall-clock time on the
# full dataset even though it looked fast on a small sample. This cap bounds
# per-token lookup cost regardless of corpus size.
MAX_TOKEN_POSTINGS = 3000

# Number of top-scoring records to keep from EACH blocking channel before
# taking the union (keeps the union step itself cheap).
TOP_K_PER_CHANNEL = 100

# Generic / high-collision normalized names (see dataset_report.txt Section 4)
# get an extra address-driven similarity requirement before being kept as
# candidates, since name alone is not discriminative for these.
GENERIC_NAME_MIN_LEN = 4  # names shorter than this (after normalization) are
                          # always treated as "generic" regardless of frequency

# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
RAPIDFUZZ_SCORE_CUTOFF = 0  # keep all scores, let the model decide

# ----------------------------------------------------------------------
# Model / training
# ----------------------------------------------------------------------
RANDOM_SEED = 42
VAL_FRACTION = 0.2  # matches dataset_report.txt Section 9's proposed split

# Negative sampling: for every true-match pair in the training candidate set,
# keep at most this many non-match candidate pairs from the SAME S1 entity's
# candidate list (these are the important "hard negatives" -- pairs the
# blocker found plausible but which are not matches).
MAX_HARD_NEGATIVES_PER_POSITIVE = 6

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_child_samples": 30,
    "verbose": -1,
    "seed": RANDOM_SEED,
}
LGBM_NUM_BOOST_ROUND = 800
LGBM_EARLY_STOPPING_ROUNDS = 50

# ----------------------------------------------------------------------
# Threshold search (optimizing macro F_0.5, precision-weighted)
# ----------------------------------------------------------------------
F_BETA = 0.5
THRESHOLD_GRID = [round(x, 3) for x in
                   [i / 1000 for i in range(300, 991, 5)]]  # 0.300 .. 0.990
