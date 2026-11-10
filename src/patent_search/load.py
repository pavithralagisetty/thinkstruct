"""Read raw patent JSON files, handle empty values, clean text, drop duplicates."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from patent_search import config

log = logging.getLogger(__name__)

# "US20240051333A1-20240215.XML" -> kind code "A1", date "20240215"
FILENAME_RE = re.compile(r"([A-Z]\d)-(\d{8})\.XML$", re.IGNORECASE)

# Bracketed reference numerals as the source writes them, always with a space
# after "(": "( 1 )", "( 2a )", "( 1 , 1 a )", "( 56 A, 56 B; 57 A)", "( 12 ″)".
# Unspaced brackets such as "formula (1)", "(2018)" or "(i)" are real text.
_NUMERAL = r"\d+\s*[A-Za-z]?\s*[′″‴']*"
REFERENCE_NUMERAL_RE = re.compile(
    rf"\s*\(\s+{_NUMERAL}(?:\s*[,;]\s*{_NUMERAL})*\s*\)"
)


@dataclass
class Patent:
    doc_number: str
    title: str
    abstract: str
    claims: list[str]           # cleaned, one item per claim, in source order
    description: str            # non blank paragraphs joined by a blank line, or ""
    has_description: bool
    classification: str
    kind_code: str | None
    pub_date: str | None        # "YYYY-MM-DD"
    source_file: str


def is_empty(value) -> bool:
    """Missing, None, blank string, [] and lists whose items are all blank."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return all(is_empty(item) for item in value)
    return False


def clean_text(text) -> str:
    """Collapse whitespace and fix spacing before commas and periods."""
    if is_empty(text):
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"(\d) \. (?=\d)", r"\1.", text)    # "0 . 75" -> "0.75"
    text = re.sub(r" +([,.])", r"\1", text)           # "claim 1 , wherein" -> "claim 1, wherein"
    return text


def clean_list(items) -> list[str]:
    """Clean each item and drop blank ones."""
    if is_empty(items):
        return []
    if not isinstance(items, list):
        items = [items]
    return [t for t in (clean_text(item) for item in items) if t]


def remove_reference_numerals(text: str) -> str:
    """Remove bracketed reference numerals. For embedding text only."""
    return re.sub(r"\s+([,.;:])", r"\1", REFERENCE_NUMERAL_RE.sub("", text)).strip()


def parse_filename(filename: str) -> tuple[str | None, str | None]:
    """Return (kind_code, pub_date as YYYY-MM-DD) from the source filename."""
    match = FILENAME_RE.search(filename or "")
    if not match:
        return None, None
    kind, d = match.groups()
    return kind.upper(), f"{d[:4]}-{d[4:6]}-{d[6:]}"


def parse_record(raw: dict, source_file: str) -> Patent | None:
    doc_number = clean_text(raw.get("doc_number"))
    if not doc_number:
        log.warning("%s: record without doc_number skipped", source_file)
        return None

    filename = clean_text(raw.get("filename"))
    kind_code, pub_date = parse_filename(filename)
    if filename and pub_date is None:
        log.warning("%s: cannot parse kind code and date from filename %r", doc_number, filename)

    raw_claims = raw.get("claims")
    claims = clean_list(raw_claims)
    if isinstance(raw_claims, list) and len(claims) != len(raw_claims) and claims:
        log.warning("%s: %d blank claim items dropped", doc_number, len(raw_claims) - len(claims))

    paragraphs = clean_list(raw.get("detailed_description"))
    return Patent(
        doc_number=doc_number,
        title=clean_text(raw.get("title")),
        abstract=clean_text(raw.get("abstract")),
        claims=claims,
        description="\n\n".join(paragraphs),
        has_description=bool(paragraphs),
        classification=clean_text(raw.get("classification")),
        kind_code=kind_code,
        pub_date=pub_date,
        source_file=source_file,
    )


def load_patents(raw_dir: Path | None = None) -> list[Patent]:
    """Load every patents_*.json file. Same doc_number twice keeps the later publication."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    files = sorted(raw_dir.glob(config.RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"No {config.RAW_GLOB} files in {raw_dir}")

    by_number: dict[str, Patent] = {}
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            log.warning("%s: not a JSON list, skipped", path.name)
            continue
        for raw in records:
            if not isinstance(raw, dict):
                continue
            patent = parse_record(raw, path.name)
            if patent is None:
                continue
            old = by_number.get(patent.doc_number)
            if old is not None:
                # On equal or unknown dates, the record read later wins.
                if (old.pub_date or "") > (patent.pub_date or ""):
                    keep = old
                else:
                    keep = patent
                log.warning("%s: duplicate, keeping %s (%s)", patent.doc_number,
                            keep.source_file, keep.pub_date)
                patent = keep
            by_number[patent.doc_number] = patent
    return sorted(by_number.values(), key=lambda p: p.doc_number)
