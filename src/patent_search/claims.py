"""Turn a patent's claim items into numbered claims with parent and root links."""

import logging
import re
from dataclasses import dataclass

from patent_search import config

log = logging.getLogger(__name__)

# Leading claim number on a cleaned item. The period (or the ":" or ")" some
# sources use) must not be followed by a digit, so a recovered item starting
# "0.5 phr of ..." is not claim 0.
#   single: "17. The", "18 : The", "2 ) The"
#   range: "6 - 9. (canceled)", "1.- 15. (canceled)",
#          "1 - 15 : (canceled)", "1 - 10 (canceled)"
RANGE_RE = re.compile(r"^(\d+)\s*\.?\s*[-–]\s*(\d+)\s*(?:\.(?!\d)|:|(?=\s*\(cancel))\s*")
SINGLE_RE = re.compile(r"^(\d+)\s*[.:)](?!\d)\s*")
CANCELLED_RE = re.compile(r"\(?\s*cancell?ed\s*\)?[.\s]*", re.IGNORECASE)
PARENT_RE = re.compile(r"\bclaims?\s+(\d+)", re.IGNORECASE)


@dataclass
class Claim:
    claim_id: str
    doc_number: str
    claim_number: int
    text: str                       # display text without the leading number
    parent_number: int | None
    root_number: int
    is_independent: bool | None
    opening_missing: bool


@dataclass
class ParsedClaims:
    claims: list[Claim]
    cancelled_numbers: set[int]


def split_number(item: str) -> tuple[int, int, str] | None:
    """Return (first, last, rest) if the item starts with a claim number, else None."""
    match = RANGE_RE.match(item) or SINGLE_RE.match(item)
    if not match:
        return None
    a = int(match.group(1))
    b = int(match.group(2)) if match.re is RANGE_RE else a
    return a, b, item[match.end():].strip()


def is_cancelled(text: str) -> bool:
    return bool(CANCELLED_RE.fullmatch(text))


def find_parent(text: str) -> int | None:
    match = PARENT_RE.search(text[: config.PARENT_SEARCH_CHARS])
    return int(match.group(1)) if match else None


def parse_claims(doc_number: str, items: list[str]) -> ParsedClaims:
    used: set[int] = set()
    numbered: list[tuple[int, str, bool]] = []    # (number, text, opening_missing), real claims only
    cancelled: set[int] = set()

    def take(number: int) -> int:
        """Reserve a claim number; on a clash use the next free one above the highest."""
        if number < 1 or number in used:
            new = max(used, default=0) + 1
            log.warning("%s: claim number %d already used, using %d", doc_number, number, new)
            number = new
        used.add(number)
        return number

    def place(item: str, number: int, opening_missing: bool):
        if is_cancelled(item):
            cancelled.add(number)
        else:
            numbered.append((number, item, opening_missing))

    n = 0
    pending: list[str] = []         # unnumbered items waiting for their numbers
    for item in items:
        parts = split_number(item)
        if parts is None:
            pending.append(item)
            continue
        a, b, rest = parts
        k = len(pending)
        backward = list(range(a - k, a))
        if k and a != n + k + 1 and a - k >= 1 and not used.intersection(backward):
            log.warning("%s: claim %d follows claim %d after %d unnumbered item(s); "
                        "numbering them backward as %s", doc_number, a, n, k, backward)
            for number, text in zip(backward, pending):
                place(text, take(number), True)
        else:
            for text in pending:
                n = take(n + 1)
                place(text, n, True)
            if a != n + 1:
                log.warning("%s: expected claim %d, found %d", doc_number, n + 1, a)
        pending = []
        if is_cancelled(rest):
            for number in range(a, b + 1):
                cancelled.add(take(number))
        else:
            if b != a:
                log.warning("%s: claim range %d-%d with content, using %d", doc_number, a, b, a)
            place(rest, take(a), False)
        n = b
    for text in pending:
        n = take(n + 1)
        place(text, n, True)

    parents = {num: (None if missing else find_parent(text)) for num, text, missing in numbered}

    def root_of(number: int) -> int:
        seen = {number}
        while (parent := parents.get(number)) is not None and parent in parents:
            if parent in seen:
                if parent == number:
                    log.warning("%s: claim %d refers to itself", doc_number, number)
                else:
                    log.warning("%s: claim reference loop at claim %d", doc_number, parent)
                break
            seen.add(parent)
            number = parent
        return number

    claims = []
    for number, text, missing in sorted(numbered):
        parent = parents[number]
        claims.append(Claim(
            claim_id=f"{doc_number}:{number}",
            doc_number=doc_number,
            claim_number=number,
            text=text,
            parent_number=parent,
            root_number=root_of(number),
            is_independent=None if missing else parent is None,
            opening_missing=missing,
        ))
    return ParsedClaims(claims=claims, cancelled_numbers=cancelled)
