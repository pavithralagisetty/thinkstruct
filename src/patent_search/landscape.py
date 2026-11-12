"""Patent map: every patent vector placed in 2D so similar patents sit close together.

The layout is t-SNE (scikit-learn, installed with sentence-transformers) on the patent vectors
with cosine distance and a fixed seed, so the same index always gives the same map. It takes a
few seconds, so it is computed once per index build and cached next to the index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from patent_search import config

CACHE_NAME = "patent_map.json"
_VERSION = 1


def _layout(vectors: np.ndarray) -> np.ndarray:
    """(n, 2) positions scaled to 0..1 on each axis."""
    n = len(vectors)
    if n < 3:   # too few for t-SNE: a line in the middle
        xy = np.column_stack([np.linspace(0, 1, n) if n > 1 else np.full(n, 0.5), np.full(n, 0.5)])
        return xy
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, metric="cosine", init="pca", random_state=config.MAP_SEED,
              perplexity=min(config.MAP_PERPLEXITY, (n - 1) / 3)).fit_transform(vectors).astype(np.float64)
    lo, span = xy.min(axis=0), xy.max(axis=0) - xy.min(axis=0)
    return (xy - lo) / np.where(span > 0, span, 1)


def patent_map(index) -> dict[str, tuple[float, float]]:
    """{doc_number: (x, y)} for every patent that has a vector. Cached per index build."""
    path = Path(index.index_dir) / CACHE_NAME
    key = {"version": _VERSION, "built_at": index.meta.get("build_time"), "model": index.meta.get("model_name"),
           "seed": config.MAP_SEED, "perplexity": config.MAP_PERPLEXITY}
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("key") == key:
            return {d: tuple(xy) for d, xy in cached["points"].items()}
    docs = sorted(d for d, p in index.patents.items() if p["vec_row"] is not None)
    vectors = index.vectors.patent_vectors[[index.patents[d]["vec_row"] for d in docs]] if docs \
        else np.zeros((0, 2))
    points = {d: (round(float(x), 5), round(float(y), 5)) for d, (x, y) in zip(docs, _layout(vectors))}
    try:
        tmp = path.with_name(CACHE_NAME + ".tmp")
        tmp.write_text(json.dumps({"key": key, "points": points}))
        tmp.replace(path)
    except OSError:
        pass   # a read-only index still gets a map, just recomputed per process
    return points
