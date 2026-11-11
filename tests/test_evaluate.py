import json

import numpy as np
import pytest

from conftest import WordEmbedder
from patent_search import config, evaluate
from patent_search.evaluate import BM25Query, BM25Store

RANKED = ["A:1", "A:2", "B:1", "B:2", "C:1", "C:2", "D:1", "D:2", "E:1", "E:2"]


# ---------- metrics on hand-made inputs ----------

def test_metrics_graded():
    grades = {"A:2": 1, "C:1": 2, "Z:9": 2}          # Z:9 is relevant but never retrieved
    m = evaluate.question_metrics(RANKED, grades)
    assert m["hit@5"] == 1.0
    assert m["p@5"] == pytest.approx(2 / 5)
    assert m["recall@10"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx(1 / 2)
    assert m["first_full_rank"] == 5
    dcg = (2 ** 1 - 1) / np.log2(3) + (2 ** 2 - 1) / np.log2(6)
    idcg = (2 ** 2 - 1) / np.log2(2) + (2 ** 2 - 1) / np.log2(3) + (2 ** 1 - 1) / np.log2(4)
    assert m["ndcg@10"] == pytest.approx(dcg / idcg)


def test_metrics_perfect_and_empty():
    perfect = evaluate.question_metrics(["A:1", "A:2"], {"A:1": 2, "A:2": 1})
    assert perfect["ndcg@10"] == pytest.approx(1.0) and perfect["mrr"] == 1.0 and perfect["recall@10"] == 1.0
    nothing = evaluate.question_metrics(RANKED, {})
    assert nothing == {"hit@5": 0.0, "p@5": 0.0, "recall@10": None, "mrr": 0.0,
                       "ndcg@10": None, "first_full_rank": None}


def test_metrics_only_count_the_cutoff():
    grades = {"E:2": 2}                                # rank 10
    assert evaluate.hit_at_k(RANKED, grades, 5) == 0.0
    assert evaluate.recall_at_k(RANKED, grades, 10) == 1.0
    assert evaluate.recall_at_k(RANKED, grades, 5) == 0.0
    assert evaluate.first_full_rank(RANKED, {"E:2": 1}) is None   # partly relevant is not "fully"


# ---------- BM25 ----------

def test_bm25_ranks_word_matches():
    store = BM25Store(["a spoke made of carbon fiber", "a tire tread", "spoke spoke spoke wheel"],
                      ["spoke patent", "tire patent"])
    claim_sim, patent_sim = store.similarities(BM25Query().embed_query("Carbon spoke"),
                                               np.array([0, 1, 2]), np.array([0, 1, -1]))
    assert claim_sim[0] == pytest.approx(1.0)        # scaled so the best claim is 1
    assert claim_sim[1] == 0.0                       # no shared word
    assert 0 < claim_sim[2] < 1                      # "spoke" only, rarer word "carbon" missing
    assert patent_sim[0] == 1.0 and patent_sim[1] == 0.0 and patent_sim[2] == 0.0   # -1: no vector


def test_bm25_no_match_and_tokenizer():
    store = BM25Store(["a tire"], ["x"])
    assert store.similarities(["zzz"], np.array([0]), np.array([0]))[0].tolist() == [0.0]
    assert evaluate.tokenize("Fiber-reinforced, (1) TIRE") == ["fiber", "reinforced", "1", "tire"]


def test_bm25_goes_through_production_search(index):
    claims, patents = evaluate.load_documents(index.index_dir)
    searcher = index.with_store(BM25Store(claims, patents), BM25Query())
    r = searcher.search("sensor", top_n=3)
    assert (r.hits[0].doc_number, r.hits[0].claim_number) == ("A", 2)
    # keyword filter still applies before ranking
    assert searcher.search("sensor", keywords="spoke").total_matches == 3


# ---------- end to end on a tiny index ----------

MODELS = {
    "nomic": {"model_name": "fake", "revision": "r1", "reuse_index": True},
    "other": {"model_name": "fake-2", "revision": "r2", "query_prefix": "q: ", "document_prefix": "d: "},
}
QUESTIONS = [{"id": "q01", "text": "spoke"}, {"id": "q02", "text": "tire sensor"}]


def factory(calls):
    def make(spec, device):
        calls.append((spec["model_name"], device))
        return WordEmbedder()
    return make


def test_run_end_to_end(index, tmp_path):
    calls = []
    report = evaluate.run(questions=QUESTIONS, index=index, models=MODELS, device="cpu",
                          cache_dir=tmp_path / "cache", top_n=3, embedder_factory=factory(calls))
    assert list(report["results"]) == ["bm25", "nomic", "other"]
    for name in report["results"]:
        assert list(report["results"][name]) == ["q01", "q02"]
        assert all(len(hits) == 3 for hits in report["results"][name].values())
    # nomic reuses the built index, so it equals the production search exactly
    prod = index.search("spoke", top_n=3)
    assert [h["claim_id"] for h in report["results"]["nomic"]["q01"]] == \
        [f"{h.doc_number}:{h.claim_number}" for h in prod.hits]
    assert report["candidates"]["nomic"]["vectors"] == "data/index"
    assert report["candidates"]["other"]["vectors"] == "embedded"
    assert calls == [("fake", "cpu"), ("fake-2", "cpu")]
    path = evaluate.save_report(report, tmp_path / "results")
    assert json.loads(path.read_text())["settings"]["w_claim"] == config.W_CLAIM


def test_cache_reused_and_invalidated(index, tmp_path):
    kwargs = dict(questions=QUESTIONS, index=index, device="cpu", cache_dir=tmp_path / "cache",
                  top_n=3, embedder_factory=factory([]))
    first = evaluate.run(["other"], models=MODELS, **kwargs)
    second = evaluate.run(["other"], models=MODELS, **kwargs)
    assert second["candidates"]["other"]["vectors"] == "cache"
    assert second["results"] == first["results"]
    changed = {"other": {**MODELS["other"], "revision": "r3"}}
    assert evaluate.run(["other"], models=changed, **kwargs)["candidates"]["other"]["vectors"] == "embedded"


def test_unknown_candidate(index):
    with pytest.raises(ValueError, match="Unknown candidate"):
        evaluate.run(["nope"], questions=QUESTIONS, index=index, models=MODELS)


def test_questions_file():
    questions = evaluate.load_questions()
    assert [q["id"] for q in questions] == [f"q{i:02d}" for i in range(1, 11)]
    assert all(set(q) == {"id", "text"} and q["text"] for q in questions)


def test_make_embedder_uses_per_model_batch_size_and_length(monkeypatch):
    import patent_search.embedder as embedder_module
    from patent_search import config
    from patent_search.evaluate import make_embedder
    seen = []
    monkeypatch.setattr(embedder_module, "Embedder", lambda **kw: seen.append(kw) or kw)
    make_embedder({"model_name": "m", "revision": "r", "batch_size": 4, "max_seq_length": 8192}, "cpu")
    make_embedder({"model_name": "m", "revision": "r"}, "cpu")
    assert (seen[0]["batch_size"], seen[0]["max_seq_length"]) == (4, 8192)
    assert (seen[1]["batch_size"], seen[1]["max_seq_length"]) == (config.EVAL_BATCH_SIZE, None)


def test_qwen_entry():
    from patent_search import config
    spec = config.EVAL_MODELS["qwen3-0.6b"]
    assert spec["model_name"] == "Qwen/Qwen3-Embedding-0.6B" and len(spec["revision"]) == 40
    assert spec["document_prefix"] == "" and spec["batch_size"] == 4 and spec["max_seq_length"] == 8192
    assert spec["query_prefix"] == ("Instruct: Given a patent search question, retrieve patent claims "
                                    "that answer it\nQuery:")


def test_embedder_reports_pooling():
    from patent_search.embedder import Embedder

    class Pooling:                                   # named like the sentence-transformers module
        def __init__(self, mode):
            self.pooling_mode = mode

    e = Embedder.__new__(Embedder)
    e.model = [object(), Pooling("lasttoken"), object()]
    assert e.pooling == "lasttoken"
    e.model = [Pooling(("mean", "max"))]
    assert e.pooling == "mean+max"
    e.model = [object()]
    assert e.pooling == "none"
