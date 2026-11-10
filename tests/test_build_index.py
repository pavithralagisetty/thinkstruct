import json
import sqlite3

import numpy as np

from patent_search.build_index import build, claim_embed_text, patent_embed_text
from patent_search.claims import parse_claims


class FakeEmbedder:
    model_name, revision, code_revision, dimension = "fake", "r1", None, 4

    def embed_documents(self, texts, show_progress=False):
        v = np.ones((len(texts), 4), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_claim_embed_text():
    parsed = parse_claims("1", ["1. A spoke star ( 1 ) comprising a hub.",
                                "2. The spoke star of claim 1, wherein a loop ( 2a ).",
                                "3. The spoke star of claim 3, wherein."])
    by = {c.claim_number: c for c in parsed.claims}
    assert claim_embed_text("SPOKE", by[1], by) == "Patent: SPOKE\nClaim: A spoke star comprising a hub."
    assert claim_embed_text("SPOKE", by[2], by) == (
        "Patent: SPOKE\nMain claim: A spoke star comprising a hub.\n"
        "Claim: The spoke star of claim 1, wherein a loop.")
    # a claim that refers to itself has no other main claim
    assert "Main claim" not in claim_embed_text("SPOKE", by[3], by)


def test_patent_embed_text():
    assert patent_embed_text("T", "A b.") == "T. A b."
    assert patent_embed_text("T", "") == "T"
    assert patent_embed_text("", "A b.") == "A b."
    assert patent_embed_text("", "") is None


def test_build_writes_index(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "patents_1.json").write_text(json.dumps([
        {"doc_number": "1", "title": "T", "abstract": "A", "claims": ["1. A.", "2. (canceled)", "a b"],
         "filename": "US1A1-20240101.XML"},
        {"doc_number": "2", "claims": ["1. B."]},
    ]))
    index = tmp_path / "index"
    counts = build(raw, index, FakeEmbedder()).counts
    assert counts["claims"] == 3 and counts["patent_vectors"] == 1 and counts["cancelled_claim_numbers"] == 1
    assert np.load(index / "claim_vectors.npy").shape == (3, 4)
    assert np.load(index / "patent_vectors.npy").shape == (1, 4)
    con = sqlite3.connect(index / "search.db")
    assert con.execute("SELECT vec_row FROM patents ORDER BY doc_number").fetchall() == [(0,), (None,)]
    assert con.execute("SELECT count(*) FROM claims").fetchone() == (3,)
    assert json.loads((index / "meta.json").read_text())["model_name"] == "fake"
    assert not list(index.glob("*.tmp"))
