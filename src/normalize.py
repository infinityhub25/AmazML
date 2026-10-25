"""
Normalization utilities for business_name and business_address.

Design notes (grounded in dataset_report.txt):
- Legal-suffix / abbreviation variants are extremely common (Pvt/Private,
  Ltd/Limited, Corp/Corporation, Rd/Road, St/Street, & vs "and") -> expand
  both sides to a canonical form so fuzzy scores aren't punished for them.
- ~7.9% of true-match address pairs have a NUMERIC CONFLICT (e.g. "12 MG
  Road" vs "45 MG Road") -- this is a strong signal for REJECTING a
  candidate, so numeric tokens are extracted separately and never blurred
  away by generic normalization.
- Some true matches are transliterations into non-Latin scripts (Bengali,
  Tamil, Hindi, Gujarati, Telugu observed in the report) where name
  similarity on raw strings is near zero. We can't reliably transliterate
  without an external service (prohibited by the challenge rules), so we
  flag non-Latin script and let blocking fall back to address-driven
  channels for those rows instead of relying on name tokens.
"""
from __future__ import annotations

import re
import unicodedata

# ----------------------------------------------------------------------
# Legal suffix / common abbreviation canonicalization
# ----------------------------------------------------------------------
_NAME_REPLACEMENTS = [
    (r"\bpvt\.?\b", "private"),
    (r"\bltd\.?\b", "limited"),
    (r"\bcorp\.?\b", "corporation"),
    (r"\bco\.?\b", "company"),
    (r"\binc\.?\b", "incorporated"),
    (r"&", " and "),
    (r"\bllp\b", "limited liability partnership"),
    (r"\bllc\b", "limited liability company"),
]

_ADDR_REPLACEMENTS = [
    (r"\brd\.?\b", "road"),
    (r"\bst\.?\b", "street"),
    (r"\bave\.?\b", "avenue"),
    (r"\bblvd\.?\b", "boulevard"),
    (r"\bdr\.?\b", "drive"),
    (r"\bln\.?\b", "lane"),
    (r"\bapt\.?\b", "apartment"),
    (r"\bfl\.?\b", "floor"),
    (r"\bnr\.?\b", "near"),
    (r"\bopp\.?\b", "opposite"),
    (r"\bp\.?o\.?\b", "post office"),
]

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTISPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+")

_STOPWORDS = {
    "the", "and", "of", "for", "a", "an", "at", "in", "on",
}


def _apply_replacements(text: str, patterns: list[tuple[str, str]]) -> str:
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text


def basic_clean(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_name(raw: str) -> str:
    """Canonical form of a business name for scoring (keeps generic words --
    those are filtered separately at blocking time, not stripped here so
    that similarity scores stay comparable across the whole pipeline)."""
    if not raw:
        return ""
    text = raw.lower()
    text = strip_domain_suffix(text)
    text = strip_accents(text)
    text = _apply_replacements(text, _NAME_REPLACEMENTS)
    text = basic_clean(text)
    return text


def normalize_address(raw: str) -> str:
    if not raw:
        return ""
    text = raw.lower()
    text = translate_native_place_names(text)
    text = strip_accents(text)
    text = _apply_replacements(text, _ADDR_REPLACEMENTS)
    text = basic_clean(text)
    return text


def name_tokens(normalized_name: str, drop_stopwords: bool = True) -> list[str]:
    toks = normalized_name.split()
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def numeric_tokens(text: str) -> set[str]:
    """Extract numeric tokens (house numbers, PIN codes, etc). Used both as
    a blocking channel and as a strong contradiction feature: two addresses
    for the same real business rarely disagree on ALL numbers, and a
    disagreement on every number is a red flag for a false merge."""
    return set(_NUMBER_RE.findall(text))


def is_generic_name(normalized_name: str, doc_freq: float, min_len: int) -> bool:
    """A name is 'generic' (see dataset_report.txt Section 4: 'physical
    therapy', 'primary care', etc. appear 600-1000+ times) if it's short or
    appears very frequently in the corpus -- for these, name similarity
    alone should not drive a match decision."""
    if len(normalized_name) < min_len:
        return True
    return doc_freq is not None and doc_freq > 0.0005  # tuned conservatively;
    # doc_freq is expected as (count of this normalized name) / (source size)


_NON_LATIN_RE = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF"
    r"\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0E00-\u0E7F\u4E00-\u9FFF]"
)


def has_non_latin_script(raw: str) -> bool:
    """Detect Devanagari/Bengali/Gujarati/Tamil/Telugu/Kannada/Malayalam/
    Thai/CJK characters -- covers the transliterated-name cases observed in
    dataset_report.txt (Hindi/Bengali/Tamil/Gujarati/Telugu examples)."""
    if not raw:
        return False
    return bool(_NON_LATIN_RE.search(raw))


def strip_accents(text: str) -> str:
    """Best-effort ASCII folding for Latin-script transliteration variants
    (e.g. accented French characters showing up in the test set)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ----------------------------------------------------------------------
# Non-Latin state/region name translation (small, static, best-effort).
# This is NOT an external lookup of business identity -- it's a fixed
# table of known place-name spellings in scripts that showed up mixed
# into otherwise-Latin Indian addresses (e.g. "..., Indiranagar, Bangalore,
# ಕರ್ನಾಟಕ" where only the state name is in Kannada). Extend as you find
# more examples in your data.
# ----------------------------------------------------------------------
_NATIVE_PLACE_NAMES = {
    # Kannada
    "ಕರ್ನಾಟಕ": "karnataka",
    # Malayalam
    "കേരളം": "kerala",
    # Devanagari (Hindi)
    "दिल्ली": "delhi",
    "महाराष्ट्र": "maharashtra",
    "उत्तर प्रदेश": "uttar pradesh",
    "गुजरात": "gujarat",
    "पंजाब": "punjab",
    "राजस्थान": "rajasthan",
    "बिहार": "bihar",
    "तमिलनाडु": "tamil nadu",
    "पश्चिम बंगाल": "west bengal",
    # Bengali
    "পশ্চিমবঙ্গ": "west bengal",
    "দিল্লি": "delhi",
    # Tamil
    "தமிழ்நாடு": "tamil nadu",
    # Telugu
    "తెలంగాణ": "telangana",
    "ఆంధ్రప్రదేశ్": "andhra pradesh",
    # Gujarati
    "ગુજરાત": "gujarat",
}


def translate_native_place_names(text: str) -> str:
    for native, english in _NATIVE_PLACE_NAMES.items():
        if native in text:
            text = text.replace(native, english)
    return text


# ----------------------------------------------------------------------
# Domain-style name handling (e.g. "celestialmemorialtrust.com",
# "SURGICALHEALTH.COM") -- common in Source 3. These have zero token
# overlap with a spaced-out name like "Celestial Memorial Trust", so we
# strip the TLD and expose a "collapsed" (space-free) form that a
# character-n-gram blocking channel can match against the S1 name with
# its spaces removed.
# ----------------------------------------------------------------------
_TLD_RE = re.compile(r"\.(com|net|org|in|co|biz|info|us)\b")


def strip_domain_suffix(text: str) -> str:
    return _TLD_RE.sub("", text)


def collapsed_name(normalized_name: str) -> str:
    """Space-free form of a normalized name, used for the char-n-gram
    blocking channel that catches domain-style records."""
    return normalized_name.replace(" ", "")


def char_ngrams(text: str, n: int = 5) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}
