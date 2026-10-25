"""
Blocking / candidate generation.

Why multi-channel (see dataset_report.txt):
  - Country-only blocking gives just 48% pair reduction -- nowhere near
    enough (target is ~200 candidates/S1, i.e. >99.999% reduction).
  - Name-token blocking alone MISSES transliterated true matches (name
    similarity as low as 0.054 for legitimate matches written in a
    different script) -- Section 3's "hardest true-match pairs".
  - Address-driven channels (token overlap + numeric-token overlap)
    recover those cases because the address commonly stays in a shared
    script (city/state names) even when the business name doesn't.

Strategy: build an inverted index per channel (name tokens, address
tokens, address numeric tokens), all scoped to `country` first. For each
S1 record, gather candidates that share at least one index entry with it
from ANY channel, score by (weighted) number of shared postings, then
keep the top MAX_CANDIDATES_PER_S1.

Very common tokens (doc frequency > MAX_TOKEN_DOC_FREQ) are excluded from
the index entirely -- they're not discriminative and would otherwise
produce blocks with tens of thousands of records (see the "physical
therapy" / "primary care" collisions in dataset_report.txt Section 4).
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd

from . import config
from .normalize import (
    char_ngrams, collapsed_name, name_tokens, normalize_address,
    normalize_name, numeric_tokens,
)

logger = logging.getLogger(__name__)


class SourceIndex:
    """Inverted indices (name tokens, address tokens, address numeric
    tokens) for one S2 or S3 dataframe, scoped by country."""

    def __init__(self, df: pd.DataFrame, source_label: str):
        self.source_label = source_label
        self.n = len(df)

        self.name_idx: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.addr_idx: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.num_idx: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        # char-ngram index on the collapsed (space-free) name -- catches
        # domain-style records like "celestialmemorialtrust.com" that have
        # zero token overlap with a spaced-out name.
        self.ngram_idx: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

        # normalized text cache, keyed by entity_id -> used later for scoring
        self.norm_name: dict[str, str] = {}
        self.norm_addr: dict[str, str] = {}

        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        logger.info("Building index for %s (%d rows)...", self.source_label, self.n)

        # ---- pass 1: normalize + compute token doc frequencies (per country) ----
        name_tok_count: dict[str, Counter] = defaultdict(Counter)
        addr_tok_count: dict[str, Counter] = defaultdict(Counter)
        num_tok_count: dict[str, Counter] = defaultdict(Counter)
        country_count: Counter = Counter()

        norm_names, norm_addrs, tok_lists, addr_tok_lists, num_tok_lists = [], [], [], [], []
        collapsed_list, ngram_list = [], []
        for row in df.itertuples(index=False):
            country = row.country
            nname = normalize_name(row.business_name)
            naddr = normalize_address(row.business_address)
            ntoks = [t for t in name_tokens(nname) if len(t) >= config.MIN_TOKEN_LEN]
            atoks = [t for t in naddr.split() if len(t) >= config.MIN_TOKEN_LEN]
            numtoks = numeric_tokens(row.business_address)

            # Domain-style names ("celestialmemorialtrust.com") normalize
            # down to a single long token (or zero tokens if very short) --
            # index those by character n-gram so a spaced-out name like
            # "Celestial Memorial Trust" can still find them.
            cname = collapsed_name(nname)
            ngrams = char_ngrams(cname, n=5) if len(ntoks) <= 1 and len(cname) >= 5 else set()

            norm_names.append(nname)
            norm_addrs.append(naddr)
            tok_lists.append(ntoks)
            addr_tok_lists.append(atoks)
            num_tok_lists.append(numtoks)
            collapsed_list.append(cname)
            ngram_list.append(ngrams)

            country_count[country] += 1
            for t in set(ntoks):
                name_tok_count[country][t] += 1
            for t in set(atoks):
                addr_tok_count[country][t] += 1
            for t in numtoks:
                num_tok_count[country][t] += 1

        # ---- pass 2: build indices, skipping over-common tokens ----
        # (skip tokens whose count exceeds BOTH the max_df fraction AND
        # the absolute MAX_TOKEN_POSTINGS cap -- the cap is what actually
        # protects us at real data scale, see config.py's comment on it)
        for i, row in enumerate(df.itertuples(index=False)):
            eid = row.entity_id
            country = row.country
            self.norm_name[eid] = norm_names[i]
            self.norm_addr[eid] = norm_addrs[i]

            cn = country_count[country] or 1
            max_df = config.MAX_TOKEN_DOC_FREQ
            max_posting = config.MAX_TOKEN_POSTINGS

            for t in set(tok_lists[i]):
                cnt = name_tok_count[country][t]
                if cnt <= max_posting and (cnt / cn) <= max_df:
                    self.name_idx[country][t].append(eid)
            for t in set(addr_tok_lists[i]):
                cnt = addr_tok_count[country][t]
                if cnt <= max_posting and (cnt / cn) <= max_df:
                    self.addr_idx[country][t].append(eid)
            for t in num_tok_lists[i]:
                if num_tok_count[country][t] <= max_posting:
                    self.num_idx[country][t].append(eid)
            for g in ngram_list[i]:
                self.ngram_idx[country][g].append(eid)

        logger.info("Index built for %s.", self.source_label)


def _channel_candidates(
    tokens: Iterable[str], country: str, idx: dict[str, dict[str, list[str]]], weight: float
) -> Counter:
    scores: Counter = Counter()
    bucket = idx.get(country, {})
    for t in tokens:
        postings = bucket.get(t)
        if not postings:
            continue
        for eid in postings:
            scores[eid] += weight
    return scores


def generate_candidates_for_s1(
    s1_row,
    s2_index: SourceIndex,
    s3_index: SourceIndex,
    max_candidates: int = config.MAX_CANDIDATES_PER_S1,
) -> list[str]:
    """Return a ranked, capped list of S2/S3 candidate entity_ids for one
    S1 record, merged across all blocking channels.

    `s1_row` may be a pandas Series OR a namedtuple from itertuples() --
    both support attribute access to entity_id/business_name/
    business_address/country, which is all this function needs.
    """
    country = s1_row.country
    nname = normalize_name(s1_row.business_name)
    naddr = normalize_address(s1_row.business_address)
    ntoks = [t for t in name_tokens(nname) if len(t) >= config.MIN_TOKEN_LEN]
    atoks = [t for t in naddr.split() if len(t) >= config.MIN_TOKEN_LEN]
    numtoks = numeric_tokens(s1_row.business_address)
    ngrams = char_ngrams(collapsed_name(nname), n=5)

    merged: Counter = Counter()
    for src_idx in (s2_index, s3_index):
        name_scores = _channel_candidates(ntoks, country, src_idx.name_idx, weight=2.0)
        addr_scores = _channel_candidates(atoks, country, src_idx.addr_idx, weight=1.0)
        num_scores = _channel_candidates(numtoks, country, src_idx.num_idx, weight=1.5)
        ngram_scores = _channel_candidates(ngrams, country, src_idx.ngram_idx, weight=0.3)

        combined = Counter()
        for c in (name_scores, addr_scores, num_scores, ngram_scores):
            combined.update(c)

        # keep top-K from this source before merging across sources, so one
        # source with many weak hits can't crowd out the other source
        top = combined.most_common(config.TOP_K_PER_CHANNEL)
        for eid, score in top:
            merged[eid] += score

    ranked = [eid for eid, _ in merged.most_common(max_candidates)]
    return ranked


def generate_all_candidates(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame
) -> dict[str, list[str]]:
    """Build indices for S2/S3 and generate a candidate list for every S1
    entity. Returns {s1_entity_id: [candidate_entity_id, ...]}."""
    s2_index = SourceIndex(s2_df, "S2")
    s3_index = SourceIndex(s3_df, "S3")

    result: dict[str, list[str]] = {}
    total = len(s1_df)
    log_every = max(1, min(2000, total // 20))
    for i, row in enumerate(s1_df.itertuples(index=False)):
        result[row.entity_id] = generate_candidates_for_s1(row, s2_index, s3_index)
        if (i + 1) % log_every == 0:
            logger.info("Blocking progress: %d / %d S1 entities", i + 1, total)
    return result
