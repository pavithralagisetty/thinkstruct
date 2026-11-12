"""Part 2 proof of concept: approximate nearest neighbour search with FAISS.

At 10M patents (~160M claims) the production search, which compares the question with
every claim vector, is too slow and too big for memory. This script checks what an ANN
index buys and what it costs, on our real claim vectors and on a larger synthetic set:

  1. Real data (10,411 claims): exact vs HNSW vs int8 vs IVF-PQ, recall@10 and latency.
  2. Two-stage: ANN top-K candidates, then the exact production score
     (W_CLAIM * claim + W_PATENT * patent) on just those. Does it keep the production top 10?
  3. Filters: post-filtering an ANN result (returns too few hits) vs filtering inside HNSW
     vs exact scoring of the filtered rows.
  4. Scale: the same indexes on N synthetic vectors (real vectors + noise).

Fast and sloppy on purpose. Run: .venv/bin/python part2/faiss_ann.py [--scale 200000]
Writes part2/results.md.
"""

import argparse
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from patent_search import config  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
QUERY_CACHE = ROOT / "part2" / "query_vectors.npy"
OUT = ROOT / "part2" / "results.md"
K = 10
BUILD_THREADS = faiss.omp_get_max_threads()

lines: list[str] = []


def out(s=""):
    print(s)
    lines.append(s)


def load_queries() -> np.ndarray:
    questions = json.loads((ROOT / "tests" / "eval" / "questions.json").read_text())
    if QUERY_CACHE.exists():
        q = np.load(QUERY_CACHE)
        if len(q) == len(questions):
            return q
    # torch and faiss each bring their own OpenMP and segfault in one process, so embed apart.
    import subprocess
    code = ("import json, numpy as np; from patent_search.embedder import Embedder; e = Embedder(); "
            f"qs = json.load(open({str(ROOT / 'tests/eval/questions.json')!r})); "
            f"np.save({str(QUERY_CACHE)!r}, np.stack([e.embed_query(x['text']) for x in qs]).astype(np.float32))")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT,
                   env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    return np.load(QUERY_CACHE)


def exact_topk(base: np.ndarray, queries: np.ndarray, k=K) -> np.ndarray:
    sims = queries @ base.T
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    order = np.take_along_axis(sims, top, 1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, 1)


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)]))


def timed_search(fn, queries, repeats=5):
    """Median and p95 ms per single query (one user, one question at a time)."""
    times, results = [], []
    for q in queries:                                   # warm up
        fn(q[None, :])
    for _ in range(repeats):
        for q in queries:
            t = time.perf_counter()
            r = fn(q[None, :])
            times.append((time.perf_counter() - t) * 1000)
    for q in queries:
        results.append(fn(q[None, :])[0])
    times.sort()
    return statistics.median(times), times[int(len(times) * 0.95)], np.array(results)


def numpy_exact(base):
    def fn(q):
        return exact_topk(base, q)
    return fn


def faiss_fn(index, **params):
    def fn(q):
        for name, value in params.items():
            if name == "efSearch":
                index.hnsw.efSearch = value
            elif name == "nprobe":
                faiss.extract_index_ivf(index).nprobe = value
            elif name == "k_factor":
                faiss.downcast_index(index).k_factor = value
        return index.search(q, K)[1]
    return fn


def index_bytes(index) -> int:
    return len(faiss.serialize_index(index))


def build(kind: str, base: np.ndarray):
    d = base.shape[1]
    t = time.perf_counter()
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
    elif kind == "sq8":
        index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    elif kind in ("ivfpq", "ivfpq_rf"):
        # PQ96x4fs: 96 sub-vectors x 4 bits = 48 bytes per vector (8-bit PQ trains very slowly here).
        # ",RFlat" keeps the full vectors and re-scores the PQ shortlist exactly.
        nlist = max(16, int(4 * np.sqrt(len(base))))
        refine = ",RFlat" if kind == "ivfpq_rf" else ""
        index = faiss.index_factory(d, f"IVF{nlist},PQ96x4fs{refine}", faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        sample = base[np.random.default_rng(0).choice(len(base), min(len(base), 100_000), replace=False)]
        index.train(sample)
    index.add(base)
    return index, time.perf_counter() - t


def compare(base, queries, truth, title, repeats):
    out(f"### {title}")
    out()
    out("| Method | Setting | Build s | Size MB | Bytes / vector | Recall@10 | Median ms | p95 ms |")
    out("|---|---|---|---|---|---|---|---|")
    n = len(base)
    med, p95, res = timed_search(numpy_exact(base), queries, repeats)
    out(f"| numpy exact (production today) | - | 0 | {base.nbytes / 1e6:.0f} | {base.nbytes // n} "
        f"| {recall(res, truth):.3f} | {med:.2f} | {p95:.2f} |")
    indexes = {}
    for kind, settings in [("flat", [{}]),
                           ("hnsw", [{"efSearch": e} for e in (16, 32, 64, 128, 256)]),
                           ("sq8", [{}]),
                           ("ivfpq", [{"nprobe": p} for p in (4, 16, 64)]),
                           ("ivfpq_rf", [{"nprobe": 16, "k_factor": f} for f in (4, 16)])]:
        faiss.omp_set_num_threads(BUILD_THREADS)
        index, secs = build(kind, base)
        indexes[kind] = index
        faiss.omp_set_num_threads(1)                    # one query, one core: fair latency
        size = index_bytes(index)
        for s in settings:
            med, p95, res = timed_search(faiss_fn(index, **s), queries, repeats)
            label = {"flat": "FAISS exact (IndexFlatIP)", "hnsw": "HNSW M=32",
                     "sq8": "int8 exact (SQ8)"}.get(kind)
            if kind.startswith("ivfpq"):
                label = f"IVF-PQ 48 B ({faiss.extract_index_ivf(index).nlist} lists)"
                if kind == "ivfpq_rf":
                    label += " + exact re-score"
            setting = ", ".join(f"{k}={v}" for k, v in s.items()) or "-"
            out(f"| {label} | {setting} | {secs:.1f} | {size / 1e6:.0f} | {size // n} "
                f"| {recall(res, truth):.3f} | {med:.2f} | {p95:.2f} |")
    out()
    return indexes


def two_stage(claims, patents, claim_patent_row, queries, hnsw, repeats):
    """ANN on claim vectors gives candidates, the exact production score reranks them."""
    out("### Two-stage: HNSW candidates, then the production score")
    out()
    out(f"Production score = {config.W_CLAIM} x claim similarity + {config.W_PATENT} x patent similarity, "
        "over every claim. The ANN index only knows claim vectors, so it fetches K candidates "
        "and the full score is computed on those.")
    out()
    has = claim_patent_row >= 0
    def prod_scores(q, rows):
        cs = claims[rows] @ q
        ps = np.zeros(len(rows), np.float32)
        h = has[rows]
        ps[h] = patents[claim_patent_row[rows][h]] @ q
        return np.round(config.W_CLAIM * cs + config.W_PATENT * ps, 6)
    all_rows = np.arange(len(claims))
    truth = [all_rows[np.argsort(-prod_scores(q, all_rows), kind="stable")[:K]] for q in queries]
    out("| Candidates K | efSearch | Overlap with production top 10 | Median ms |")
    out("|---|---|---|---|")
    faiss.omp_set_num_threads(1)
    for cand in (10, 50, 100, 200):
        hnsw.hnsw.efSearch = max(64, cand)
        times, found = [], []
        for _ in range(repeats):
            for q in queries:
                t = time.perf_counter()
                rows = hnsw.search(q[None, :], cand)[1][0]
                rows = rows[rows >= 0]
                top = rows[np.argsort(-prod_scores(q, rows), kind="stable")[:K]]
                times.append((time.perf_counter() - t) * 1000)
        for q in queries:
            rows = hnsw.search(q[None, :], cand)[1][0]
            found.append(rows[np.argsort(-prod_scores(q, rows), kind="stable")[:K]])
        out(f"| {cand} | {hnsw.hnsw.efSearch} | {recall(found, truth):.3f} | {statistics.median(times):.2f} |")
    out()


def filtered(claims, queries, hnsw, filters, repeats):
    out("### Filters with an ANN index")
    out()
    out("Truth is the exact top 10 among the claims that pass the filter. "
        "*Post-filter*: ask HNSW for k results, drop those that fail the filter. "
        "*In-index*: FAISS `IDSelectorBatch`, HNSW only returns rows that pass. "
        "*Exact on subset*: what production does today (filter first, score what passed).")
    out()
    out("| Filter | Claims passing | Method | Avg hits returned (of 10) | Recall@10 | Median ms |")
    out("|---|---|---|---|---|---|")
    faiss.omp_set_num_threads(1)
    for name, rows in filters.items():
        rows = np.asarray(rows, dtype=np.int64)
        allowed = np.zeros(len(claims), bool)
        allowed[rows] = True
        sub = claims[rows]
        truth = [rows[t] for t in exact_topk(sub, queries)]
        sel = faiss.IDSelectorBatch(rows)
        methods = {}
        for k in (10, 100):
            def post(q, k=k):
                hnsw.hnsw.efSearch = max(64, k)
                ids = hnsw.search(q, k)[1][0]
                return [ids[(ids >= 0) & allowed[np.maximum(ids, 0)]][:K]]
            methods[f"post-filter, k={k}"] = post
        for ef in (64, 512):
            def inside(q, ef=ef):
                params = faiss.SearchParametersHNSW(sel=sel, efSearch=ef)
                ids = hnsw.search(q, K, params=params)[1][0]
                return [ids[ids >= 0]]
            methods[f"in-index (IDSelector), efSearch={ef}"] = inside
        def exact_sub(q):
            return [rows[exact_topk(sub, q)[0]]]
        methods["exact on subset"] = exact_sub
        for mname, fn in methods.items():
            times = []
            for _ in range(repeats):
                for q in queries:
                    t = time.perf_counter()
                    fn(q[None, :])
                    times.append((time.perf_counter() - t) * 1000)
            res = [fn(q[None, :])[0] for q in queries]
            hits = np.mean([len(r) for r in res])
            rec = np.mean([len(set(r) & set(t)) / K for r, t in zip(res, truth)])
            out(f"| {name} | {len(rows):,} ({len(rows) / len(claims):.1%}) | {mname} "
                f"| {hits:.1f} | {rec:.3f} | {statistics.median(times):.2f} |")
    out()


def synthetic(real: np.ndarray, n: int, sigma: float, seed=0) -> np.ndarray:
    """n vectors: each a real claim vector plus Gaussian noise, renormalized.
    Keeps the real data's topic structure but invents the fine detail."""
    rng = np.random.default_rng(seed)
    base = np.empty((n, real.shape[1]), np.float32)
    for start in range(0, n, 100_000):
        stop = min(n, start + 100_000)
        idx = rng.integers(0, len(real), stop - start)
        chunk = real[idx] + rng.normal(0, sigma, (stop - start, real.shape[1])).astype(np.float32)
        base[start:stop] = chunk / np.linalg.norm(chunk, axis=1, keepdims=True)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=200_000, help="synthetic vectors (0 to skip)")
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    claims = np.load(INDEX_DIR / "claim_vectors.npy").astype(np.float32)
    patents = np.load(INDEX_DIR / "patent_vectors.npy").astype(np.float32)
    queries = load_queries()
    con = sqlite3.connect(INDEX_DIR / "search.db")
    rows = con.execute("""SELECT c.vec_row, COALESCE(p.vec_row, -1), p.cpc_subclass, p.cpc_group,
                                 c.is_independent
                          FROM claims c JOIN patents p USING (doc_number) ORDER BY c.vec_row""").fetchall()
    claim_patent_row = np.array([r[1] for r in rows], dtype=np.int64)

    out("# Part 2 proof of concept: approximate vector search with FAISS")
    out()
    out(f"FAISS {faiss.__version__}, {len(queries)} evaluation questions embedded with "
        f"{config.MODEL_NAME}, {claims.shape[1]} dims, k = {K}. Latency is one question at a time "
        f"on one CPU core (FAISS) after a warm-up, {args.repeats} rounds. numpy exact uses Accelerate (all cores). "
        "Recall@10 = share of the exact top 10 (by claim similarity) that the method returns.")
    out()

    truth = exact_topk(claims, queries)
    idx = compare(claims, queries, truth, f"Real claim vectors ({len(claims):,})", args.repeats)
    two_stage(claims, patents, claim_patent_row, queries, idx["hnsw"], args.repeats)
    filtered(claims, queries, idx["hnsw"], {
        "CPC B60C 11": [r[0] for r in rows if r[2] == "B60C" and r[3] == 11],
        "Main claims only": [r[0] for r in rows if r[4] == 1],
    }, args.repeats)

    if args.scale:
        base = synthetic(claims, args.scale, args.sigma)
        truth = exact_topk(base, queries)
        compare(base, queries, truth,
                f"Synthetic vectors ({args.scale:,}, real vectors + noise sigma={args.sigma})", args.repeats)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
