import json

import numpy as np
import pytest

from patent_search.build_index import build
from patent_search.search import Index

WORDS = ["spoke", "tire", "bassinet", "sensor"]


class WordEmbedder:
    """Deterministic fake: one dimension per word in WORDS, plus a constant."""
    model_name, revision, code_revision, dimension = "fake", "r1", None, 5

    def _vec(self, text):
        v = np.array([text.lower().count(w) for w in WORDS] + [1], dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts, show_progress=False):
        return np.array([self._vec(t) for t in texts]).reshape(len(texts), 5)

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("idx")
    raw = tmp / "raw"
    raw.mkdir()
    (raw / "patents_1.json").write_text(json.dumps([
        {"doc_number": "B", "title": "SPOKE", "abstract": "A spoke.",
         "classification": "B60B100FI", "filename": "USBA1-20250110.XML",
         "claims": ["1. A spoke.", "2. The spoke of claim 1, with a tire."]},
        {"doc_number": "A", "title": "TIRE", "abstract": "A tire.",
         "classification": "B60C1112FI", "filename": "USAA1-20240615.XML",
         "claims": ["1. A tire.", "2. The tire of claim 1, with a sensor.", "a tire"]},
        {"doc_number": "C", "claims": ["1. A spoke."]},
    ]))
    build(raw, tmp / "index", WordEmbedder())
    return Index(tmp / "index", embedder=WordEmbedder())
