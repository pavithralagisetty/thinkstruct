"""Load, clean, parse and embed everything, then write data/index."""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from patent_search import config
from patent_search.claims import Claim, parse_claims
from patent_search.load import Patent, load_patents, remove_reference_numerals

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE patents (
    doc_number TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    description TEXT NOT NULL,
    has_description INTEGER NOT NULL,
    classification TEXT NOT NULL,
    kind_code TEXT,
    pub_date TEXT,
    source_file TEXT NOT NULL,
    vec_row INTEGER
);
CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY,
    doc_number TEXT NOT NULL REFERENCES patents(doc_number),
    claim_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    embed_text TEXT NOT NULL,
    parent_number INTEGER,
    root_number INTEGER NOT NULL,
    is_independent INTEGER,
    opening_missing INTEGER NOT NULL,
    vec_row INTEGER NOT NULL
);
CREATE INDEX claims_doc ON claims(doc_number, claim_number);
"""


@dataclass
class BuildResult:
    counts: dict


def claim_embed_text(title: str, claim: Claim, by_number: dict[int, Claim]) -> str:
    """Patent / Main claim / Claim lines. Main claim only when the claim depends on another."""
    lines = []
    if title:
        lines.append(f"Patent: {title}")
    root = by_number.get(claim.root_number)
    if claim.parent_number is not None and root is not None and root.claim_number != claim.claim_number:
        lines.append(f"Main claim: {remove_reference_numerals(root.text)}")
    lines.append(f"Claim: {remove_reference_numerals(claim.text)}")
    return "\n".join(lines)


def patent_embed_text(patent: Patent) -> str | None:
    """'{title}. {abstract}', the title alone without abstract, None if both are empty."""
    if patent.title and patent.abstract:
        return f"{patent.title}. {patent.abstract}"
    return patent.title or patent.abstract or None


def _bool(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _save_npy(path: Path, array: np.ndarray):
    with open(path, "wb") as f:
        np.save(f, array)


def build(raw_dir: Path | None = None, index_dir: Path | None = None, embedder=None) -> BuildResult:
    index_dir = Path(index_dir or config.INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)

    patents = load_patents(raw_dir)
    claim_rows, claim_texts, cancelled = [], [], 0
    for patent in patents:
        parsed = parse_claims(patent.doc_number, patent.claims)
        cancelled += len(parsed.cancelled_numbers)
        by_number = {c.claim_number: c for c in parsed.claims}
        for claim in parsed.claims:
            claim_rows.append(claim)
            claim_texts.append(claim_embed_text(patent.title, claim, by_number))

    patent_texts = [(p, patent_embed_text(p)) for p in patents]
    embedded_patents = [(p, t) for p, t in patent_texts if t is not None]

    if embedder is None:
        from patent_search.embedder import Embedder
        embedder = Embedder()
    log.info("Embedding %d patents and %d claims", len(embedded_patents), len(claim_texts))
    patent_vectors = embedder.embed_documents([t for _, t in embedded_patents], show_progress=True)
    claim_vectors = embedder.embed_documents(claim_texts, show_progress=True)
    dim = embedder.dimension
    patent_vectors = patent_vectors.reshape(len(embedded_patents), dim)
    claim_vectors = claim_vectors.reshape(len(claim_texts), dim)

    patent_vec_row = {p.doc_number: i for i, (p, _) in enumerate(embedded_patents)}
    counts = {
        "patents": len(patents),
        "claims": len(claim_rows),
        "patents_without_description": sum(not p.has_description for p in patents),
        "claims_opening_missing": sum(c.opening_missing for c in claim_rows),
        "cancelled_claim_numbers": cancelled,
        "patent_vectors": len(embedded_patents),
    }

    # Write everything to temp files, then swap them in.
    tmp = {name: index_dir / f"{name}.tmp" for name in
           ("search.db", "claim_vectors.npy", "patent_vectors.npy", "meta.json")}
    for path in tmp.values():
        path.unlink(missing_ok=True)

    con = sqlite3.connect(tmp["search.db"])
    try:
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT INTO patents VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(p.doc_number, p.title, p.abstract, p.description, int(p.has_description),
              p.classification, p.kind_code, p.pub_date, p.source_file,
              patent_vec_row.get(p.doc_number)) for p in patents])
        con.executemany(
            "INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(c.claim_id, c.doc_number, c.claim_number, c.text, text, c.parent_number,
              c.root_number, _bool(c.is_independent), int(c.opening_missing), row)
             for row, (c, text) in enumerate(zip(claim_rows, claim_texts))])
        con.commit()
    finally:
        con.close()

    _save_npy(tmp["claim_vectors.npy"], claim_vectors)
    _save_npy(tmp["patent_vectors.npy"], patent_vectors)
    meta = {
        "model_name": embedder.model_name,
        "model_revision": embedder.revision,
        "model_code_revision": embedder.code_revision,
        "vector_size": dim,
        "build_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
    }
    tmp["meta.json"].write_text(json.dumps(meta, indent=2))

    # meta.json last, so a complete meta always describes a complete index.
    for name in ("search.db", "claim_vectors.npy", "patent_vectors.npy", "meta.json"):
        os.replace(tmp[name], index_dir / name)
    return BuildResult(counts=counts)
