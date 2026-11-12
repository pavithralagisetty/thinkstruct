"""Part 3 query adapter: one linear layer on the question vector, trained on abstracts or on
short summaries of them (config.ADAPTER_SOURCES).

* Vectors, once. Claim vectors are the index's own (data/index/claim_vectors.npy, the same
  embed_text search uses). Each abstract is embedded once as a query (`search_query: ` + abstract)
  and cached in config.ADAPTER_ABSTRACT_CACHE. With source "summary" the query is instead the
  patent's 1 to 2 sentence summary (config.ADAPTER_SUMMARIES_PATH), cached separately.
* Pairs. An abstract's positives are all claims of its own patent; every other patent's claims
  are its wrong answers.
* Layer. q' = normalize(W q + b), W starting as the identity and b as zero, so before training it
  returns q unchanged and search starts exactly where it is. The loss is a softmax over the claims
  (temperature ADAPTER_TEMPERATURE) that puts the probability on the patent's own claims, plus
  ADAPTER_IDENTITY_PENALTY * (||W - I||^2 + ||b||^2) so the layer only moves as far as the data
  pushes it.
* Split, by week like training.py: the oldest PART3_TRAIN_FRACTION of weeks train (their newest
  ADAPTER_VAL_FRACTION pick the epoch), the newest weeks test. Training only ever sees the claims
  of training patents; validation and test rank every claim in the index.
* Search. With PATENT_SEARCH_QUERY_ADAPTER=<file> the question goes through the model, then this
  layer, then is normalized and searched as usual. Claim vectors and the index do not change.

    python -m patent_search adapter [--source abstract|summary] [--device mps]
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from patent_search import config
from patent_search.training import week_of

log = logging.getLogger(__name__)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.maximum(norm, 1e-12)).astype(np.float32)


class QueryAdapter:
    """q -> normalize(W q + b)."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray, meta: dict | None = None):
        self.weight = np.asarray(weight, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.meta = meta or {}

    @classmethod
    def identity(cls, dim: int) -> "QueryAdapter":
        return cls(np.eye(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))

    def apply(self, q: np.ndarray) -> np.ndarray:
        return _normalize(q @ self.weight.T + self.bias)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, weight=self.weight, bias=self.bias, meta=json.dumps(self.meta))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "QueryAdapter":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No query adapter at {path}. Train one: python -m patent_search adapter")
        with np.load(path) as f:
            return cls(f["weight"], f["bias"], json.loads(str(f["meta"])))


class AdaptedQuery:
    """embed_query() through the model, then the adapter. Everything else passes through."""

    def __init__(self, embedder, adapter: QueryAdapter):
        self._embedder = embedder
        self.adapter = adapter

    def embed_query(self, text: str) -> np.ndarray:
        return self.adapter.apply(self._embedder.embed_query(text))

    def __getattr__(self, name):
        return getattr(self._embedder, name)


def check_adapter(adapter: QueryAdapter, index_meta: dict) -> None:
    """An adapter only fits the vectors of the model it was trained on."""
    want = (index_meta.get("model_name"), index_meta.get("model_revision"))
    got = (adapter.meta.get("model_name"), adapter.meta.get("model_revision"))
    if got != want:
        raise ValueError(f"The query adapter was trained for {got[0]} (revision {got[1]}), but the index "
                         f"uses {want[0]} (revision {want[1]}). Retrain it: python -m patent_search adapter")


# ---------- data ----------

@dataclass
class Data:
    docs: list[str]                 # one per patent with an abstract, in split order
    weeks: list[str]
    abstracts: np.ndarray           # (patents, dim) query vectors (abstracts or summaries), normalized
    claims: np.ndarray              # (claims, dim) the index's claim vectors
    claim_patent: np.ndarray        # (claims,) row in docs of each claim's patent, -1 if none


def load_summaries(path: Path, docs: list[str]) -> list[str]:
    """The summary of each patent in docs. Every patent needs one; a missing or blank one is an error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No summaries at {path}.")
    summaries = json.loads(path.read_text())
    missing = [d for d in docs if not str(summaries.get(d) or "").strip()]
    if missing:
        raise ValueError(f"{len(missing)} patents have no summary in {path}, for example {missing[:3]}.")
    return [summaries[d].strip() for d in docs]


def load_data(index_dir: Path, embedder, cache_path: Path = config.ADAPTER_ABSTRACT_CACHE,
              summaries_path: Path | None = None) -> Data:
    """Query vectors come from the abstracts, or from the summaries when summaries_path is given."""
    index_dir = Path(index_dir)
    meta = json.loads((index_dir / "meta.json").read_text())
    with sqlite3.connect(index_dir / "search.db") as db:
        patents = db.execute("SELECT doc_number, abstract, source_file FROM patents "
                             "WHERE trim(coalesce(abstract, '')) != '' ORDER BY doc_number").fetchall()
        claim_rows = db.execute("SELECT doc_number, vec_row FROM claims ORDER BY vec_row").fetchall()
    docs = [p[0] for p in patents]
    texts = [p[1] for p in patents] if summaries_path is None else load_summaries(summaries_path, docs)
    weeks = [week_of(Path(p[2])) for p in patents]

    key = {"model_name": meta.get("model_name"), "model_revision": meta.get("model_revision"),
           "query_prefix": config.QUERY_PREFIX,
           "texts_sha256": hashlib.sha256("\0".join(docs + texts).encode("utf-8")).hexdigest()}
    abstracts = None
    if Path(cache_path).exists():
        with np.load(cache_path) as f:
            if json.loads(str(f["key"])) == key:
                abstracts = f["vectors"]
                log.info("query vectors from %s", cache_path)
    if abstracts is None:
        log.info("embedding %d %s as queries", len(texts), "abstracts" if summaries_path is None else "summaries")
        abstracts = np.array([embedder.embed_query(t) for t in texts], dtype=np.float32)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(cache_path).with_name(Path(cache_path).stem + ".tmp.npz")
        np.savez(tmp, vectors=abstracts, key=json.dumps(key))
        tmp.replace(cache_path)

    claims = np.load(index_dir / "claim_vectors.npy").astype(np.float32)
    row_of = {d: i for i, d in enumerate(docs)}
    claim_patent = np.full(len(claims), -1, dtype=np.int64)
    for doc, vec_row in claim_rows:
        claim_patent[vec_row] = row_of.get(doc, -1)
    return Data(docs, weeks, abstracts, claims, claim_patent)


def split_weeks(weeks: list[str], train_fraction: float = config.PART3_TRAIN_FRACTION,
                val_fraction: float = config.ADAPTER_VAL_FRACTION) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patent rows for train, validation and test, by week: oldest train, newest test."""
    order = sorted(set(weeks))
    n_train = round(len(order) * train_fraction)
    n_val = max(1, round(n_train * val_fraction))
    train_w, val_w = set(order[:n_train - n_val]), set(order[n_train - n_val:n_train])
    rows = np.arange(len(weeks))
    w = np.array(weeks)
    return (rows[np.isin(w, list(train_w))], rows[np.isin(w, list(val_w))],
            rows[~np.isin(w, list(train_w | val_w))])


# ---------- metrics ----------

def retrieval_metrics(queries: np.ndarray, patent_rows: np.ndarray, claims: np.ndarray,
                      claim_patent: np.ndarray, k: int = 10) -> dict:
    """Each query ranks every claim; its own patent's claims are the right answers."""
    sims = queries @ claims.T
    hit1 = hitk = mrr = ndcg = own_sim = other_sim = 0.0
    for i, p in enumerate(patent_rows):
        own = claim_patent == p
        order = np.argsort(-sims[i], kind="stable")[:k]
        rel = own[order]
        hit1 += float(rel[0])
        hitk += float(rel.any())
        mrr += 1.0 / (int(np.argmax(rel)) + 1) if rel.any() else 0.0
        dcg = float(np.sum(rel / np.log2(np.arange(2, k + 2))))
        ideal = float(np.sum(1.0 / np.log2(np.arange(2, min(own.sum(), k) + 2))))
        ndcg += dcg / ideal if ideal else 0.0
        own_sim += float(sims[i, own].mean())
        other_sim += float(sims[i, ~own].mean())
    n = len(patent_rows)
    return {"hit@1": hit1 / n, f"hit@{k}": hitk / n, f"mrr@{k}": mrr / n, f"ndcg@{k}": ndcg / n,
            "mean_sim_own_claims": own_sim / n, "mean_sim_other_claims": other_sim / n,
            "gap": (own_sim - other_sim) / n}


# ---------- training ----------

def train(data: Data, train_rows: np.ndarray, val_rows: np.ndarray, epochs: int = config.ADAPTER_EPOCHS,
          lr: float = config.ADAPTER_LEARNING_RATE, temperature: float = config.ADAPTER_TEMPERATURE,
          penalty: float = config.ADAPTER_IDENTITY_PENALTY, seed: int = config.ADAPTER_SEED,
          ) -> tuple[QueryAdapter, list[dict]]:
    """Full batch Adam. Keeps the epoch with the best validation MRR (epoch 0 = identity)."""
    import torch

    torch.manual_seed(seed)
    dim = data.abstracts.shape[1]
    train_claims = np.isin(data.claim_patent, train_rows)
    C = torch.from_numpy(data.claims[train_claims])
    owner = torch.from_numpy(data.claim_patent[train_claims])
    A = torch.from_numpy(data.abstracts[train_rows])
    positive = owner[None, :] == torch.from_numpy(train_rows)[:, None]     # (abstracts, claims)

    W = torch.nn.Parameter(torch.eye(dim))
    b = torch.nn.Parameter(torch.zeros(dim))
    eye = torch.eye(dim)
    opt = torch.optim.Adam([W, b], lr=lr)

    def current() -> QueryAdapter:
        return QueryAdapter(W.detach().numpy().copy(), b.detach().numpy().copy())

    def val_mrr(adapter):
        return retrieval_metrics(adapter.apply(data.abstracts[val_rows]), val_rows, data.claims,
                                 data.claim_patent)["mrr@10"]

    best, best_epoch = val_mrr(current()), 0
    best_adapter = current()
    history = [{"epoch": 0, "loss": None, "val_mrr@10": best}]
    for epoch in range(1, epochs + 1):
        q = torch.nn.functional.normalize(A @ W.T + b, dim=-1)
        logits = q @ C.T / temperature
        # Softmax over all train claims; the probability mass should sit on the own patent's claims.
        nce = (torch.logsumexp(logits, dim=1) -
               torch.logsumexp(logits.masked_fill(~positive, float("-inf")), dim=1)).mean()
        loss = nce + penalty * (((W - eye) ** 2).sum() + (b ** 2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == epochs:
            m = val_mrr(current())
            history.append({"epoch": epoch, "loss": round(nce.item(), 5), "val_mrr@10": m})
            if m > best:
                best, best_epoch, best_adapter = m, epoch, current()
    best_adapter.meta["best_epoch"] = best_epoch
    return best_adapter, history


def run(index_dir: Path | None = None, embedder=None, out_path: Path | None = None,
        results_dir: Path = config.PART3_RESULTS_DIR, source: str = "abstract") -> dict:
    if source not in config.ADAPTER_SOURCES:
        raise ValueError(f"source must be one of: {', '.join(config.ADAPTER_SOURCES)}")
    paths = config.ADAPTER_SOURCES[source]
    out_path = Path(out_path or paths["out"])
    index_dir = Path(index_dir or config.INDEX_DIR)
    meta = json.loads((index_dir / "meta.json").read_text())
    if embedder is None:
        from patent_search.embedder import Embedder
        embedder = Embedder()
    data = load_data(index_dir, embedder, paths["cache"],
                     config.ADAPTER_SUMMARIES_PATH if source == "summary" else None)
    has_claims = set(data.claim_patent[data.claim_patent >= 0].tolist())
    train_rows, val_rows, test_rows = (np.array([r for r in rows if r in has_claims], dtype=np.int64)
                                       for rows in split_weeks(data.weeks))
    log.info("patents: %d train, %d validation, %d test", len(train_rows), len(val_rows), len(test_rows))

    identity = QueryAdapter.identity(data.abstracts.shape[1])
    adapter, history = train(data, train_rows, val_rows)

    def scores(a, rows):
        return retrieval_metrics(a.apply(data.abstracts[rows]), rows, data.claims, data.claim_patent)
    result = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "split": {name: {"patents": len(rows), "weeks": [min(data.weeks[r] for r in rows),
                                                         max(data.weeks[r] for r in rows)]}
                  for name, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows))},
        "settings": {k: getattr(config, k) for k in ("ADAPTER_TEMPERATURE", "ADAPTER_LEARNING_RATE",
                                                    "ADAPTER_EPOCHS", "ADAPTER_IDENTITY_PENALTY",
                                                    "ADAPTER_VAL_FRACTION", "PART3_TRAIN_FRACTION")},
        "best_epoch": adapter.meta["best_epoch"],
        "change": {"weight_minus_identity_norm": float(np.linalg.norm(adapter.weight - identity.weight)),
                   "bias_norm": float(np.linalg.norm(adapter.bias))},
        "before": {"validation": scores(identity, val_rows), "test": scores(identity, test_rows)},
        "after": {"validation": scores(adapter, val_rows), "test": scores(adapter, test_rows)},
        "history": history,
    }
    adapter.meta.update({"model_name": meta.get("model_name"), "model_revision": meta.get("model_revision"),
                         "index_build_time": meta.get("build_time"), "trained": result["time"], "source": source,
                         "test": result["after"]["test"]})
    adapter.save(out_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{paths['results']}.json").write_text(json.dumps(result, indent=2) + "\n")
    (results_dir / f"{paths['results']}.md").write_text(markdown(result))
    return result


LABELS = [("hit@1", "Own claim ranked 1st"), ("hit@10", "Own claim in top 10"), ("mrr@10", "MRR@10"),
          ("ndcg@10", "nDCG@10"), ("mean_sim_own_claims", "Mean similarity, own claims"),
          ("mean_sim_other_claims", "Mean similarity, other patents' claims"), ("gap", "Gap (own - other)")]


def markdown(r: dict) -> str:
    s = r["split"]
    query = r.get("source", "abstract")
    lines = [f"# Part 3 query adapter (trained on {query} queries)", "",
             f"Train {s['train']['patents']} patents ({s['train']['weeks'][0]} to {s['train']['weeks'][1]}), "
             f"validation {s['validation']['patents']} ({s['validation']['weeks'][0]} to "
             f"{s['validation']['weeks'][1]}), test {s['test']['patents']} ({s['test']['weeks'][0]} to "
             f"{s['test']['weeks'][1]}). Each {query} ranks all claims in the index; its own patent's "
             f"claims are the right answers. Best epoch {r['best_epoch']} of {r['settings']['ADAPTER_EPOCHS']}; "
             f"||W - I|| = {r['change']['weight_minus_identity_norm']:.3f}, ||b|| = {r['change']['bias_norm']:.3f}.",
             "", "| Test metric | Before (identity) | After | Change |", "|---|---|---|---|"]
    for key, label in LABELS:
        b, a = r["before"]["test"][key], r["after"]["test"][key]
        lines.append(f"| {label} | {b:.4f} | {a:.4f} | {a - b:+.4f} |")
    return "\n".join(lines) + "\n"
