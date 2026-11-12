import json

import numpy as np

from fastapi.testclient import TestClient

from patent_search import api, landscape, search as search_module


def test_map_covers_patents_with_a_vector_and_is_cached(index):
    xy = landscape.patent_map(index)
    assert set(xy) == {"A", "B"}                      # C has no title or abstract, so no vector
    assert all(0 <= v <= 1 for p in xy.values() for v in p)
    cache = json.loads((index.index_dir / landscape.CACHE_NAME).read_text())
    assert cache["key"]["built_at"] == index.meta["build_time"]
    assert landscape.patent_map(index) == xy


def test_tsne_layout_is_repeatable_and_scaled():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 8))
    a, b = landscape._layout(v), landscape._layout(v)
    assert np.allclose(a, b)
    assert np.allclose(a.min(axis=0), 0) and np.allclose(a.max(axis=0), 1)


def test_map_endpoints(index, monkeypatch):
    monkeypatch.setattr(search_module, "get_index", lambda: index)
    client = TestClient(api.app)
    m = client.get("/map").json()
    assert [p["doc_number"] for p in m["points"]] == ["A", "B"]
    assert m["points"][0]["subclass"] == "B60C" and m["points"][0]["group"] == "B60C 11"
    assert m["classifications"]
    r = client.post("/map/matches", json={"query": "spoke"}).json()
    assert r["total_matches"] == 3 and r["matches"][0]["doc_number"] in {"B", "C"}
    assert r["matches"][0]["score"] is not None and r["matches"][0]["best_claim"] == 1
    kw = client.post("/map/matches", json={"keywords": "sensor"}).json()
    assert kw == {"total_matches": 1, "matches": [
        {"doc_number": "A", "score": None, "matching_claims": 1, "best_claim": 2}]}
    assert client.post("/map/matches", json={}).status_code == 400
