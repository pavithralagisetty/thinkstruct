"""Model comparison: run fixed questions through the production search for each
candidate (BM25 or an embedding model) and save the top claim ids. Only the
vectors change between candidates; weights, filtering and tie order are the
same search code. Grading (and so the metrics) comes from labels added later."""

import hashlib
import io
import json
import logging
import math
import os
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from patent_search import config
from patent_search.build_index import patent_embed_text
from patent_search.search import Index, VectorStore

log = logging.getLogger(__name__)

BM25 = "bm25"


# ---------- BM25 baseline ----------

def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. No stemming, same spirit as the keyword filter."""
    return re.findall(r"\w+", text.lower())


class _BM25Corpus:
    def __init__(self, texts: list[str], k1: float, b: float):
        docs = [Counter(tokenize(t)) for t in texts]
        lengths = np.array([sum(d.values()) for d in docs], dtype=np.float64)
        avg = lengths.mean() if len(lengths) and lengths.mean() > 0 else 1.0
        self.size = len(docs)
        self.k1 = k1
        self.norm = k1 * (1 - b + b * lengths / avg)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in doc.items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(i)
                tfs.append(tf)
        self.postings = {t: (np.array(ids), np.array(tfs, dtype=np.float64)) for t, (ids, tfs) in postings.items()}

    def scores(self, terms: list[str]) -> np.ndarray:
        """Okapi BM25 per document, divided by the best document's score so it lies in [0, 1]
        like a cosine similarity and the production weights mean the same thing."""
        s = np.zeros(self.size, dtype=np.float64)
        for term in dict.fromkeys(terms):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            df = len(ids)
            idf = math.log(1 + (self.size - df + 0.5) / (df + 0.5))
            s[ids] += idf * tf * (self.k1 + 1) / (tf + self.norm[ids])
        best = s.max() if self.size else 0.0
        return s / best if best > 0 else s


class BM25Store:
    """Word-matching stand-in for VectorStore: same similarities() shape, so the
    production Index.search can rank with it. The "query vector" is a token list."""

    def __init__(self, claim_texts: list[str], patent_texts: list[str],
                 k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.claims = _BM25Corpus(claim_texts, k1, b)
        self.patents = _BM25Corpus(patent_texts, k1, b)

    def similarities(self, query_terms: list[str], claim_rows: np.ndarray,
                     patent_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        claim_sim = self.claims.scores(query_terms)[claim_rows]
        patent_sim = np.zeros(len(patent_rows), dtype=np.float64)
        has_vec = patent_rows >= 0
        patent_sim[has_vec] = self.patents.scores(query_terms)[patent_rows[has_vec]]
        return claim_sim, patent_sim


class BM25Query:
    """Embedder-shaped query side of BM25."""
    model_name, revision = BM25, None

    def embed_query(self, text: str) -> list[str]:
        return tokenize(text)


# ---------- metrics (grades: {claim_id: 0 | 1 | 2}) ----------

def _grade(grades: dict[str, int], claim_id: str) -> int:
    return grades.get(claim_id, 0)


def hit_at_k(ranked: list[str], grades: dict[str, int], k: int = 5) -> float:
    return float(any(_grade(grades, c) >= config.EVAL_RELEVANT_GRADE for c in ranked[:k]))


def precision_at_k(ranked: list[str], grades: dict[str, int], k: int = 5) -> float:
    return sum(_grade(grades, c) >= config.EVAL_RELEVANT_GRADE for c in ranked[:k]) / k


def recall_at_k(ranked: list[str], grades: dict[str, int], k: int = 10) -> float | None:
    """None when the question has no relevant claim at all."""
    relevant = {c for c, g in grades.items() if g >= config.EVAL_RELEVANT_GRADE}
    if not relevant:
        return None
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: list[str], grades: dict[str, int]) -> float:
    for rank, c in enumerate(ranked, 1):
        if _grade(grades, c) >= config.EVAL_RELEVANT_GRADE:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int = 10) -> float | None:
    """Graded gain 2^grade - 1. None when no claim has a positive grade."""
    def dcg(gs):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs))
    ideal = dcg(sorted((g for g in grades.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([_grade(grades, c) for c in ranked[:k]]) / ideal


def first_full_rank(ranked: list[str], grades: dict[str, int]) -> int | None:
    """1-based rank of the first fully relevant claim, None if none is in the list."""
    for rank, c in enumerate(ranked, 1):
        if _grade(grades, c) >= config.EVAL_FULL_GRADE:
            return rank
    return None


def question_metrics(ranked: list[str], grades: dict[str, int]) -> dict:
    return {
        "hit@5": hit_at_k(ranked, grades, 5),
        "p@5": precision_at_k(ranked, grades, 5),
        "recall@10": recall_at_k(ranked, grades, 10),
        "mrr": reciprocal_rank(ranked, grades),
        "ndcg@10": ndcg_at_k(ranked, grades, 10),
        "first_full_rank": first_full_rank(ranked, grades),
    }


# ---------- candidates ----------

def default_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def make_embedder(spec: dict, device: str):
    from patent_search.embedder import Embedder
    return Embedder(
        model_name=spec["model_name"], revision=spec["revision"], code_revision=spec.get("code_revision"),
        document_prefix=spec.get("document_prefix", ""), query_prefix=spec.get("query_prefix", ""),
        batch_size=spec.get("batch_size", config.EVAL_BATCH_SIZE),
        trust_remote_code=spec.get("trust_remote_code", False),
        device=device, max_seq_length=spec.get("max_seq_length"),
    )


def load_documents(index_dir: Path) -> tuple[list[str], list[str]]:
    """Claim and patent embedding texts in vec_row order, exactly as the build embedded them."""
    con = sqlite3.connect(index_dir / "search.db")
    try:
        claims = [t for _, t in con.execute("SELECT vec_row, embed_text FROM claims ORDER BY vec_row")]
        patents = [patent_embed_text(title, abstract) for _, title, abstract in con.execute(
            "SELECT vec_row, title, abstract FROM patents WHERE vec_row IS NOT NULL ORDER BY vec_row")]
    finally:
        con.close()
    return claims, patents


def texts_hash(claim_texts: list[str], patent_texts: list[str]) -> str:
    h = hashlib.sha256()
    for part in ("claims", *claim_texts, "patents", *patent_texts):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def cache_key(spec: dict, claim_texts: list[str], patent_texts: list[str]) -> dict:
    key = {
        "model_name": spec["model_name"], "revision": spec["revision"],
        "code_revision": spec.get("code_revision"), "document_prefix": spec.get("document_prefix", ""),
        "texts_sha256": texts_hash(claim_texts, patent_texts),
    }
    if spec.get("max_seq_length") is not None:     # only when set, so existing caches stay valid
        key["max_seq_length"] = spec["max_seq_length"]
    if Path(spec["model_name"]).is_dir():          # a local model: no revision, so hash its weights
        key["weights_sha256"] = weights_hash(Path(spec["model_name"]))
    return key


def weights_hash(folder: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(folder.glob("*.safetensors")) + sorted(folder.glob("*.bin")):
        h.update(path.name.encode())
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def cached_vectors(name: str, spec: dict, embedder, claim_texts: list[str], patent_texts: list[str],
                   cache_dir: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load vectors from data/eval_cache/<name>/ when model, revision, prefix and texts
    all match; otherwise embed and save. Returns (claims, patents, was_cached)."""
    key = cache_key(spec, claim_texts, patent_texts)
    folder = Path(cache_dir) / name
    meta_path = folder / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("key") == key:
        return np.load(folder / "claim_vectors.npy"), np.load(folder / "patent_vectors.npy"), True

    log.info("%s: embedding %d claims and %d patents", name, len(claim_texts), len(patent_texts))
    claims = embedder.embed_documents(claim_texts, show_progress=True).reshape(len(claim_texts), -1)
    patents = embedder.embed_documents(patent_texts, show_progress=True).reshape(len(patent_texts), claims.shape[1])
    folder.mkdir(parents=True, exist_ok=True)
    for file, array in (("claim_vectors.npy", claims), ("patent_vectors.npy", patents)):
        with open(folder / f"{file}.tmp", "wb") as f:
            np.save(f, array)
        os.replace(folder / f"{file}.tmp", folder / file)
    meta_path.write_text(json.dumps({"key": key, "vector_size": int(claims.shape[1]),
                                     "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2))
    return claims, patents, False


# ---------- precomputed vectors (models too large to run here) ----------

class PrecomputedQuery:
    """embed_query() for a model whose question vectors were made elsewhere: it looks the
    question up by its exact text and refuses any other question."""

    def __init__(self, questions: list[dict], vectors: np.ndarray):
        self.by_text = {q["text"]: vectors[i] for i, q in enumerate(questions)}

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self.by_text:
            raise ValueError(f"No precomputed vector for the question {text!r}; re-embed the questions")
        return self.by_text[text]


def import_vectors(name: str, zip_path: Path, spec: dict, claim_texts: list[str], patent_texts: list[str],
                   questions: list[dict], cache_dir: Path) -> Path:
    """Check the zip from colab_qwen4b.py against the local texts, questions and config, then
    store it in data/eval_cache/<name>/ under the same key the cache uses."""
    import zipfile
    key = cache_key(spec, claim_texts, patent_texts)
    with zipfile.ZipFile(zip_path) as z:
        meta_name = next((n for n in z.namelist() if n.endswith("meta.json")), None)
        if meta_name is None:
            raise ValueError(f"{zip_path}: no *meta.json inside")
        meta = json.loads(z.read(meta_name))
        arrays = {}
        for part in ("claim", "patent", "query"):
            with z.open(f"{part}_vectors.npy") as f:
                arrays[part] = np.load(io.BytesIO(f.read()))
    expected = {"model_name": key["model_name"], "revision": key["revision"],
                "document_prefix": key["document_prefix"], "query_prefix": spec.get("query_prefix", ""),
                "max_seq_length": spec.get("max_seq_length"), "texts_sha256": key["texts_sha256"],
                "question_ids": [q["id"] for q in questions]}
    wrong = [k for k, v in expected.items() if meta.get(k) != v]
    if wrong:
        raise ValueError(f"{zip_path} does not match the local {', '.join(wrong)}; re-embed on Colab")
    sizes = {"claim": len(claim_texts), "patent": len(patent_texts), "query": len(questions)}
    dims = {a.shape[1] for a in arrays.values() if a.ndim == 2}
    for part, a in arrays.items():
        if a.ndim != 2 or a.shape[0] != sizes[part] or len(dims) != 1 or not np.isfinite(a).all():
            raise ValueError(f"{zip_path}: {part}_vectors has shape {a.shape} or bad values, "
                             f"expected {sizes[part]} finite rows of one size")
        norms = np.linalg.norm(a, axis=1)
        if np.abs(norms - 1).max() > 0.01:
            raise ValueError(f"{zip_path}: {part}_vectors are not normalized")

    folder = Path(cache_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    for part, a in arrays.items():
        with open(folder / f"{part}_vectors.npy.tmp", "wb") as f:
            np.save(f, a.astype(np.float32))
        os.replace(folder / f"{part}_vectors.npy.tmp", folder / f"{part}_vectors.npy")
    (folder / "meta.json").write_text(json.dumps({
        "key": key, "vector_size": dims.pop(), "precomputed": True, "dtype": meta.get("dtype"),
        "query_prefix": expected["query_prefix"], "questions": [{"id": q["id"], "text": q["text"]} for q in questions],
        "source": Path(zip_path).name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False))
    return folder


def precomputed_vectors(name: str, spec: dict, claim_texts: list[str], patent_texts: list[str],
                        cache_dir: Path) -> tuple[VectorStore, PrecomputedQuery]:
    folder = Path(cache_dir) / name
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"{name}: no vectors in {folder}; import them with "
                         f"`python -m patent_search eval --import-vectors ZIP --models {name}`")
    meta = json.loads(meta_path.read_text())
    if meta.get("key") != cache_key(spec, claim_texts, patent_texts):
        raise ValueError(f"{name}: the vectors in {folder} were made from other texts or settings; "
                         "re-embed on Colab and import again")
    store = VectorStore(np.load(folder / "claim_vectors.npy"), np.load(folder / "patent_vectors.npy"))
    return store, PrecomputedQuery(meta["questions"], np.load(folder / "query_vectors.npy"))


def load_questions(path: Path = config.EVAL_QUESTIONS_PATH) -> list[dict]:
    questions = json.loads(Path(path).read_text())
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question ids in {path}")
    return questions


# ---------- run ----------

def run(candidates: list[str] | None = None, questions: list[dict] | None = None,
        index: Index | None = None, models: dict | None = None, device: str | None = None,
        cache_dir: Path | None = None, top_n: int = config.EVAL_TOP_N,
        embedder_factory=make_embedder) -> dict:
    """Search every question with every candidate and return the report."""
    models = config.EVAL_MODELS if models is None else models
    candidates = [BM25, *models] if candidates is None else candidates
    unknown = [c for c in candidates if c != BM25 and c not in models]
    if unknown:
        raise ValueError(f"Unknown candidate(s): {', '.join(unknown)}. Known: {', '.join([BM25, *models])}")
    questions = load_questions() if questions is None else questions
    index = index or Index()
    device = device or default_device()
    cache_dir = Path(cache_dir or config.EVAL_CACHE_DIR)
    claim_texts, patent_texts = load_documents(index.index_dir)
    if len(claim_texts) != len(index.claims):
        raise RuntimeError("search.db and the loaded index disagree; rebuild the index")

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index": {k: index.meta.get(k) for k in ("model_name", "model_revision", "build_time", "counts")},
        "settings": {"top_n": top_n, "w_claim": config.W_CLAIM, "w_patent": config.W_PATENT,
                     "bm25_k1": config.BM25_K1, "bm25_b": config.BM25_B, "device": device},
        "questions": questions,
        "candidates": {},
        "results": {},
    }
    for name in candidates:
        start = time.perf_counter()
        info = {"kind": "bm25" if name == BM25 else "embedding"}
        if name == BM25:
            searcher = index.with_store(BM25Store(claim_texts, patent_texts), BM25Query())
        elif models[name].get("precomputed"):
            spec = models[name]
            vectors, embedder = precomputed_vectors(name, spec, claim_texts, patent_texts, cache_dir)
            info.update({k: spec.get(k) for k in ("model_name", "revision", "code_revision", "query_prefix",
                                                  "document_prefix", "max_seq_length", "pooling")})
            info.update({"batch_size": None, "vectors": "precomputed"})
            searcher = index.with_store(vectors, embedder)
        else:
            spec = models[name]
            embedder = embedder_factory(spec, device)
            if spec.get("query_adapter"):
                from patent_search.adapter import AdaptedQuery, QueryAdapter, check_adapter
                adapter = QueryAdapter.load(spec["query_adapter"])
                check_adapter(adapter, index.meta)
                embedder = AdaptedQuery(embedder, adapter)
                info["query_adapter"] = {"path": str(spec["query_adapter"]), "trained": adapter.meta.get("trained")}
            info.update({k: spec.get(k) for k in ("model_name", "revision", "code_revision",
                                                  "query_prefix", "document_prefix")})
            info["batch_size"] = getattr(embedder, "batch_size", None)
            info["max_seq_length"] = getattr(getattr(embedder, "model", None), "max_seq_length", None)
            info["pooling"] = getattr(embedder, "pooling", None)
            if spec.get("reuse_index"):
                vectors, info["vectors"] = index.vectors, "data/index"
            else:
                claims, patents, cached = cached_vectors(name, spec, embedder, claim_texts, patent_texts, cache_dir)
                vectors, info["vectors"] = VectorStore(claims, patents), "cache" if cached else "embedded"
            searcher = index.with_store(vectors, embedder)
        report["results"][name] = {}
        for q in questions:
            response = searcher.search(q["text"], None, top_n)
            report["results"][name][q["id"]] = [
                {"claim_id": f"{h.doc_number}:{h.claim_number}", "score": h.score} for h in response.hits]
        info["seconds"] = round(time.perf_counter() - start, 1)
        report["candidates"][name] = info
        log.info("%s done in %.1fs", name, info["seconds"])
    return report


def save_report(report: dict, results_dir: Path | None = None) -> Path:
    results_dir = Path(results_dir or config.EVAL_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"run_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return path
