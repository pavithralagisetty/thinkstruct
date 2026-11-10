"""Search over the built index: exact keyword filter, then semantic ranking."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from patent_search import config
from patent_search.keywords import compile_keywords, matches_all

REBUILD = "Rebuild it with: python -m patent_search build"


class IndexNotReady(RuntimeError):
    """The index is missing or does not match the current config."""


@dataclass
class SearchHit:
    doc_number: str
    title: str
    claim_number: int
    text: str
    score: float | None
    is_independent: bool | None
    opening_missing: bool
    pub_date: str | None
    classification: str
    source_file: str


@dataclass
class SearchResponse:
    total_matches: int
    hits: list[SearchHit]

    def to_dict(self) -> dict:
        return asdict(self)


class VectorStore:
    """The only code that touches vectors, so it can move to FAISS or Qdrant later."""

    def __init__(self, claim_vectors: np.ndarray, patent_vectors: np.ndarray):
        self.claim_vectors = claim_vectors
        self.patent_vectors = patent_vectors

    def similarities(self, query_vector: np.ndarray, claim_rows: np.ndarray,
                     patent_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cosine similarity of the query to the given claim and patent rows.
        A patent row of -1 means the patent has no vector and scores 0."""
        q = query_vector.astype(np.float32)
        claim_sim = self.claim_vectors[claim_rows] @ q
        has_vec = patent_rows >= 0
        patent_sim = np.zeros(len(patent_rows), dtype=np.float32)
        patent_sim[has_vec] = self.patent_vectors[patent_rows[has_vec]] @ q
        return claim_sim, patent_sim


class Index:
    """Everything needed at search time, loaded once into memory."""

    def __init__(self, index_dir: Path | None = None, embedder=None):
        index_dir = Path(index_dir or config.INDEX_DIR)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            raise IndexNotReady(f"No index found in {index_dir}. {REBUILD}")
        meta = json.loads(meta_path.read_text())
        wanted = (config.MODEL_NAME, config.MODEL_REVISION, config.MODEL_CODE_REVISION)
        built = (meta.get("model_name"), meta.get("model_revision"), meta.get("model_code_revision"))
        if embedder is None and built != wanted:
            raise IndexNotReady(
                f"The index was built with model {built[0]} (revision {built[1]}, code {built[2]}), "
                f"but config uses {wanted[0]} (revision {wanted[1]}, code {wanted[2]}). {REBUILD}")
        self.meta = meta
        self._embedder = embedder

        con = sqlite3.connect(index_dir / "search.db")
        con.row_factory = sqlite3.Row
        try:
            self.patents = {r["doc_number"]: dict(r) for r in con.execute(
                "SELECT doc_number, title, classification, pub_date, source_file, vec_row FROM patents")}
            self.claims = [dict(r) for r in con.execute(
                "SELECT doc_number, claim_number, text, is_independent, opening_missing, vec_row "
                "FROM claims ORDER BY doc_number, claim_number")]
        finally:
            con.close()

        self.vectors = VectorStore(np.load(index_dir / "claim_vectors.npy"),
                                   np.load(index_dir / "patent_vectors.npy"))
        if len(self.vectors.claim_vectors) != len(self.claims):
            raise IndexNotReady(f"claim_vectors.npy does not match search.db. {REBUILD}")
        self.claim_rows = np.array([c["vec_row"] for c in self.claims], dtype=np.int64)
        self.patent_rows = np.array(
            [v if (v := self.patents[c["doc_number"]]["vec_row"]) is not None else -1 for c in self.claims],
            dtype=np.int64)

    @property
    def embedder(self):
        if self._embedder is None:
            from patent_search.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def hit(self, i: int, score: float | None) -> SearchHit:
        c = self.claims[i]
        p = self.patents[c["doc_number"]]
        return SearchHit(
            doc_number=c["doc_number"], title=p["title"], claim_number=c["claim_number"],
            text=c["text"], score=score,
            is_independent=None if c["is_independent"] is None else bool(c["is_independent"]),
            opening_missing=bool(c["opening_missing"]), pub_date=p["pub_date"],
            classification=p["classification"], source_file=p["source_file"],
        )

    def search(self, query: str | None = None, keywords: str | list[str] | None = None,
               top_n: int = config.TOP_N_DEFAULT) -> SearchResponse:
        query = query.strip() if query else None
        patterns = compile_keywords(keywords)
        if not query and not patterns:
            raise ValueError("Give a query or at least one keyword.")
        if not isinstance(top_n, int) or isinstance(top_n, bool) or not 1 <= top_n <= config.TOP_N_MAX:
            raise ValueError(f"top_n must be a whole number from 1 to {config.TOP_N_MAX}.")

        # Step 1: exact keyword filter on the claim's own display text.
        if patterns:
            matched = [i for i, c in enumerate(self.claims) if matches_all(c["text"], patterns)]
        else:
            matched = list(range(len(self.claims)))
        total = len(matched)

        # Step 2b: keywords only. self.claims is already in doc_number, claim_number order.
        if not query:
            return SearchResponse(total, [self.hit(i, None) for i in matched[:top_n]])

        # Step 2a: weighted claim and patent similarity.
        if not matched:
            return SearchResponse(0, [])
        rows = np.array(matched, dtype=np.int64)
        q = self.embedder.embed_query(query)
        claim_sim, patent_sim = self.vectors.similarities(q, self.claim_rows[rows], self.patent_rows[rows])
        scores = config.W_CLAIM * claim_sim.astype(np.float64) + config.W_PATENT * patent_sim.astype(np.float64)
        scored = [(round(float(s), 6), i) for s, i in zip(scores, matched)]
        # Ties: doc_number then claim_number, which is the order of i.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return SearchResponse(total, [self.hit(i, s) for s, i in scored[:top_n]])


@lru_cache(maxsize=1)
def get_index() -> Index:
    return Index()


def search(query: str | None = None, keywords: str | list[str] | None = None,
           top_n: int = config.TOP_N_DEFAULT) -> SearchResponse:
    return get_index().search(query, keywords, top_n)
