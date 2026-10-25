"""
Feature engineering for candidate (S1, S2/S3) pairs.

Feature groups (grounded in dataset_report.txt):
  1. Name similarity: jaccard, token_sort_ratio, token_set_ratio,
     partial_ratio, jaro_winkler -- Section 3 shows true matches spread
     across all similarity levels (32% near-identical down to hard
     transliteration cases at ~0.05), so multiple metrics are kept and
     left for the model to weight rather than collapsed into one score.
  2. Address similarity: same five metrics.
  3. Numeric agreement / conflict in addresses -- Section 3: 7.9% of true
     matches have a numeric CONFLICT and most true matches show numeric
     AGREEMENT when numbers are present at all. This is one of the
     strongest precision levers against hard negatives (Section 7).
  4. Generic-name interaction -- Section 4 shows names like "physical
     therapy" recur 600-1000+ times; for these, name similarity should
     count for less and address should carry more weight.
  5. Triangulation (Section 8) -- if this is an S1-S2 candidate pair,
     look at the S1 entity's S3 candidates (and vice versa) and take the
     best S2<->S3 similarity found. High co-match similarity is strong
     corroborating evidence; its absence (when both a plausible S2 AND S3
     candidate exist) is a mild red flag.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from . import config
from .normalize import normalize_address, normalize_name, numeric_tokens

FEATURE_NAMES = [
    "name_jaccard", "name_tsr", "name_set", "name_partial", "name_jw",
    "addr_jaccard", "addr_tsr", "addr_set", "addr_partial", "addr_jw",
    "num_agree", "num_conflict", "num_s1_has", "num_cand_has",
    "country_match",
    "is_generic_name", "generic_name_x_name_sim",
    "cand_is_s2",
    "triangulation_best_sim", "triangulation_has_other_side",
    "len_name_s1", "len_name_cand", "len_addr_s1", "len_addr_cand",
]


def _jaccard(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union else 0.0


def _numeric_features(addr_a_raw: str, addr_b_raw: str) -> tuple[float, float, float, float]:
    na = numeric_tokens(addr_a_raw)
    nb = numeric_tokens(addr_b_raw)
    has_a = 1.0 if na else 0.0
    has_b = 1.0 if nb else 0.0
    if not na or not nb:
        return 0.0, 0.0, has_a, has_b
    agree = 1.0 if (na & nb) else 0.0
    conflict = 1.0 if not (na & nb) else 0.0
    return agree, conflict, has_a, has_b


class NameFrequencyTable:
    """Tracks how often each normalized name occurs in a source, used to
    flag 'generic' names (see dataset_report.txt Section 4)."""

    def __init__(self, normalized_names: Iterable[str]):
        self.counter = Counter(normalized_names)
        self.total = sum(self.counter.values()) or 1

    def doc_freq(self, normalized_name: str) -> float:
        return self.counter.get(normalized_name, 0) / self.total

    def is_generic(self, normalized_name: str) -> bool:
        if len(normalized_name) < config.GENERIC_NAME_MIN_LEN:
            return True
        return self.doc_freq(normalized_name) > 0.0005


class PairFeatureBuilder:
    """Computes the full feature vector for (s1_id, candidate_id) pairs.
    Holds pre-normalized lookups for S1 and S2+S3 records plus, per S1
    entity, the full candidate list (needed for the triangulation feature).
    """

    def __init__(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        candidates: dict[str, list[str]],
    ):
        self.s1_records = self._index_records(s1_df)
        s23_records = {}
        s23_records.update(self._index_records(s2_df))
        s23_records.update(self._index_records(s3_df))
        self.s23_records = s23_records
        self.candidates = candidates

        all_norm_names = [r["norm_name"] for r in s23_records.values()]
        self.name_freq = NameFrequencyTable(all_norm_names)

    @staticmethod
    def _index_records(df: pd.DataFrame) -> dict[str, dict]:
        out = {}
        for row in df.itertuples(index=False):
            eid = row.entity_id
            raw_name = row.business_name
            raw_addr = row.business_address
            out[eid] = {
                "raw_name": raw_name,
                "raw_addr": raw_addr,
                "country": row.country,
                "norm_name": normalize_name(raw_name),
                "norm_addr": normalize_address(raw_addr),
            }
        return out

    def _best_cross_source_sim(self, s1_id: str, cand_id: str, cand_is_s2: bool) -> tuple[float, float]:
        """Triangulation: among this S1's OTHER-source candidates, what's
        the best name similarity to `cand_id`? Returns (best_sim,
        has_other_side_candidates)."""
        cand_list = self.candidates.get(s1_id, [])
        other_prefix = "S3-" if cand_is_s2 else "S2-"
        others = [c for c in cand_list if c.startswith(other_prefix)]
        if not others:
            return 0.0, 0.0
        cand_name = self.s23_records[cand_id]["norm_name"]
        best = 0.0
        for other_id in others:
            other_name = self.s23_records[other_id]["norm_name"]
            sim = fuzz.token_sort_ratio(cand_name, other_name) / 100.0
            if sim > best:
                best = sim
        return best, 1.0

    def build_features(self, s1_id: str, cand_id: str) -> dict:
        s1 = self.s1_records[s1_id]
        cand = self.s23_records[cand_id]

        n1, n2 = s1["norm_name"], cand["norm_name"]
        a1, a2 = s1["norm_addr"], cand["norm_addr"]

        name_jaccard = _jaccard(set(n1.split()), set(n2.split()))
        addr_jaccard = _jaccard(set(a1.split()), set(a2.split()))

        name_tsr = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_set = fuzz.token_set_ratio(n1, n2) / 100.0
        name_partial = fuzz.partial_ratio(n1, n2) / 100.0
        name_jw = JaroWinkler.normalized_similarity(n1, n2)

        addr_tsr = fuzz.token_sort_ratio(a1, a2) / 100.0
        addr_set = fuzz.token_set_ratio(a1, a2) / 100.0
        addr_partial = fuzz.partial_ratio(a1, a2) / 100.0
        addr_jw = JaroWinkler.normalized_similarity(a1, a2)

        num_agree, num_conflict, num_s1_has, num_cand_has = _numeric_features(
            s1["raw_addr"], cand["raw_addr"]
        )

        country_match = 1.0 if s1["country"] == cand["country"] else 0.0

        is_generic = 1.0 if self.name_freq.is_generic(n1) else 0.0
        generic_x_sim = is_generic * name_tsr

        cand_is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
        tri_best_sim, tri_has_other = self._best_cross_source_sim(
            s1_id, cand_id, cand_is_s2=bool(cand_is_s2)
        )

        return {
            "name_jaccard": name_jaccard,
            "name_tsr": name_tsr,
            "name_set": name_set,
            "name_partial": name_partial,
            "name_jw": name_jw,
            "addr_jaccard": addr_jaccard,
            "addr_tsr": addr_tsr,
            "addr_set": addr_set,
            "addr_partial": addr_partial,
            "addr_jw": addr_jw,
            "num_agree": num_agree,
            "num_conflict": num_conflict,
            "num_s1_has": num_s1_has,
            "num_cand_has": num_cand_has,
            "country_match": country_match,
            "is_generic_name": is_generic,
            "generic_name_x_name_sim": generic_x_sim,
            "cand_is_s2": cand_is_s2,
            "triangulation_best_sim": tri_best_sim,
            "triangulation_has_other_side": tri_has_other,
            "len_name_s1": float(len(n1)),
            "len_name_cand": float(len(n2)),
            "len_addr_s1": float(len(a1)),
            "len_addr_cand": float(len(a2)),
        }

    def build_feature_matrix(self, pairs: list[tuple[str, str]]) -> pd.DataFrame:
        rows = [self.build_features(s1_id, cand_id) for s1_id, cand_id in pairs]
        df = pd.DataFrame(rows, columns=FEATURE_NAMES)
        return df
