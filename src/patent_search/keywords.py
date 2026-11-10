"""Exact keyword parsing and matching. One parser for the CLI and the API.

Rules: AND across keywords, case insensitive, whole words, "double quotes" for
a phrase, a trailing * for word starts, and a hyphen between words matches a
space (and the other way round). No stemming, spelling variants or synonyms.
"""

import re

# Hyphen-minus plus the Unicode hyphens; an en dash is not a word hyphen.
_SEPARATOR = r"[\s\-‐‑]+"
_SPLIT_RE = re.compile(_SEPARATOR)
_TOKEN_RE = re.compile(r'"([^"]*)"?|(\S+)')


def parse_keywords(text: str | None) -> list[str]:
    """Split a keyword string: spaces separate keywords, text in double quotes stays
    one phrase. An unclosed quote runs to the end of the string."""
    if not text:
        return []
    keywords = []
    for phrase, word in _TOKEN_RE.findall(text):
        keyword = " ".join((phrase or word).replace('"', " ").split())
        if keyword:
            keywords.append(keyword)
    return keywords


def normalize_keywords(keywords: str | list[str] | None) -> list[str]:
    """A string goes through parse_keywords; each list item is one keyword (or phrase)."""
    if keywords is None:
        return []
    if isinstance(keywords, str):
        return parse_keywords(keywords)
    result = []
    for item in keywords:
        keyword = " ".join(str(item or "").replace('"', " ").split())
        if keyword:
            result.append(keyword)
    return result


def compile_keyword(keyword: str) -> re.Pattern | None:
    """Regex for one keyword or phrase. None when nothing is left to match (e.g. "*")."""
    parts = []
    for word in _SPLIT_RE.split(keyword.strip()):
        prefix = word.endswith("*")
        word = word.rstrip("*")
        if not word:
            continue
        parts.append(re.escape(word) + (r"\w*" if prefix else ""))
    if not parts:
        return None
    return re.compile(r"(?<!\w)" + _SEPARATOR.join(parts) + r"(?!\w)", re.IGNORECASE)


def compile_keywords(keywords: str | list[str] | None) -> list[re.Pattern]:
    patterns = [compile_keyword(k) for k in normalize_keywords(keywords)]
    return [p for p in patterns if p is not None]


def matches_all(text: str, patterns: list[re.Pattern]) -> bool:
    """True when every pattern is found in the text (the claim's own display text)."""
    return all(p.search(text) for p in patterns)
