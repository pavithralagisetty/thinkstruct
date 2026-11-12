import numpy as np
import pytest

from conftest import WordEmbedder
from patent_search import adapter as A
from patent_search import config


def test_identity_changes_nothing(index):
    q = WordEmbedder().embed_query("spoke tire")
    assert np.allclose(A.QueryAdapter.identity(5).apply(q), q)
    plain = index.with_store(index.vectors, WordEmbedder()).search("tire sensor", top_n=5)
    adapted = index.with_store(index.vectors, A.AdaptedQuery(WordEmbedder(), A.QueryAdapter.identity(5)))
    assert [(h.doc_number, h.claim_number, h.score) for h in adapted.search("tire sensor", top_n=5).hits] == \
        [(h.doc_number, h.claim_number, h.score) for h in plain.hits]


def test_adapter_output_is_normalized_and_saved(tmp_path):
    rng = np.random.default_rng(0)
    a = A.QueryAdapter(rng.normal(size=(4, 4)), rng.normal(size=4), {"model_name": "m"})
    out = a.apply(rng.normal(size=(3, 4)))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    a.save(tmp_path / "a.npz")
    b = A.QueryAdapter.load(tmp_path / "a.npz")
    assert np.allclose(b.weight, a.weight) and b.meta == {"model_name": "m"}
    with pytest.raises(FileNotFoundError, match="python -m patent_search adapter"):
        A.QueryAdapter.load(tmp_path / "missing.npz")


def test_check_adapter_refuses_another_model():
    a = A.QueryAdapter.identity(2)
    a.meta = {"model_name": "m", "model_revision": "r1"}
    A.check_adapter(a, {"model_name": "m", "model_revision": "r1"})
    with pytest.raises(ValueError, match="Retrain"):
        A.check_adapter(a, {"model_name": "m", "model_revision": "r2"})


def _toy(seed=0, patents=40, per=3, dim=8):
    """Claims near their patent's direction; abstracts rotated away, so the identity is not best."""
    rng = np.random.default_rng(seed)
    centers = A._normalize(rng.normal(size=(patents, dim)))
    claims = A._normalize(np.repeat(centers, per, axis=0) + 0.3 * rng.normal(size=(patents * per, dim)))
    rot, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    abstracts = A._normalize(0.6 * centers + 0.4 * centers @ rot.T)
    weeks = [f"2024-01-{1 + i // 4:02d}" for i in range(patents)]
    return A.Data([f"P{i}" for i in range(patents)], weeks, abstracts.astype(np.float32),
                  claims.astype(np.float32), np.repeat(np.arange(patents), per))


def test_training_starts_at_identity_and_improves():
    data = _toy()
    train_rows, val_rows, test_rows = A.split_weeks(data.weeks)
    assert not set(train_rows) & set(test_rows) and max(train_rows) < min(test_rows)
    adapter, history = A.train(data, train_rows, val_rows, epochs=200, lr=1e-2, penalty=0.0)
    assert history[0]["epoch"] == 0 and history[0]["loss"] is None      # epoch 0 is the identity
    before = A.retrieval_metrics(data.abstracts[test_rows], test_rows, data.claims, data.claim_patent)
    after = A.retrieval_metrics(adapter.apply(data.abstracts[test_rows]), test_rows, data.claims, data.claim_patent)
    assert after["gap"] > before["gap"]
    assert after["mrr@10"] >= before["mrr@10"]


def test_metrics_on_a_known_case():
    claims = np.eye(3, dtype=np.float32)
    owner = np.array([0, 0, 1])
    m = A.retrieval_metrics(np.array([[1, 0.1, 0]], dtype=np.float32), np.array([0]), claims, owner, k=2)
    assert m["hit@1"] == 1.0 and m["mrr@2"] == 1.0 and m["ndcg@2"] == pytest.approx(1.0)


def test_search_uses_adapter_when_configured(index, monkeypatch, tmp_path):
    import patent_search.embedder as embedder_module
    from patent_search.search import IndexNotReady
    flip = A.QueryAdapter(-np.eye(5, dtype=np.float32), np.zeros(5, dtype=np.float32),
                          {"model_name": index.meta["model_name"], "model_revision": index.meta["model_revision"]})
    flip.save(tmp_path / "flip.npz")
    monkeypatch.setattr(embedder_module, "Embedder", WordEmbedder)
    monkeypatch.setattr(config, "QUERY_ADAPTER", str(tmp_path / "flip.npz"))
    fresh = index.with_store(index.vectors, None)
    q = WordEmbedder().embed_query("spoke")
    assert np.allclose(fresh.embedder.embed_query("spoke"), -q)
    monkeypatch.setattr(config, "QUERY_ADAPTER", str(tmp_path / "missing.npz"))
    with pytest.raises(IndexNotReady, match="No query adapter"):
        index.with_store(index.vectors, None).embedder


def test_summaries_must_cover_every_patent(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"A": "A tire.", "B": "  "}')
    assert A.load_summaries(path, ["A"]) == ["A tire."]
    with pytest.raises(ValueError, match="1 patents have no summary"):
        A.load_summaries(path, ["A", "B"])
    with pytest.raises(FileNotFoundError):
        A.load_summaries(tmp_path / "missing.json", ["A"])



def test_unknown_source_is_refused(index):
    with pytest.raises(ValueError, match="source must be one of"):
        A.run(index.index_dir, WordEmbedder(), source="title")
    assert config.ADAPTER_SOURCES["summary"]["out"] != config.ADAPTER_SOURCES["abstract"]["out"]
    assert config.ADAPTER_SOURCES["summary"]["cache"] != config.ADAPTER_SOURCES["abstract"]["cache"]
