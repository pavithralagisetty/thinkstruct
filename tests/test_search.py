import json

import pytest

from patent_search import config
from patent_search.search import Index, IndexNotReady

def test_needs_query_or_keyword(index):
    with pytest.raises(ValueError):
        index.search()
    with pytest.raises(ValueError):
        index.search("  ", keywords="*")
    with pytest.raises(ValueError):
        index.search("tire", top_n=0)


def test_keywords_only_sorted_without_score(index):
    r = index.search(keywords="tire")
    assert r.total_matches == 4
    assert [(h.doc_number, h.claim_number) for h in r.hits] == [("A", 1), ("A", 2), ("A", 3), ("B", 2)]
    assert all(h.score is None for h in r.hits)
    assert index.search(keywords="tire", top_n=2).total_matches == 4


def test_query_ranks_and_breaks_ties(index):
    r = index.search("spoke")
    assert r.total_matches == 6
    # B claim 1 matches the query and so does B's patent vector
    assert (r.hits[0].doc_number, r.hits[0].claim_number) == ("B", 1)
    scores = [h.score for h in r.hits]
    assert scores == sorted(scores, reverse=True)
    assert all(s == round(s, 6) for s in scores)
    # equal scores come out in doc_number, claim_number order
    for a, b in zip(r.hits, r.hits[1:]):
        if a.score == b.score:
            assert (a.doc_number, a.claim_number) < (b.doc_number, b.claim_number)


def test_patent_without_vector_scores_zero(index):
    # C has no title or abstract, so its score is only 0.8 * claim similarity
    hit = next(h for h in index.search("spoke", top_n=10).hits if h.doc_number == "C")
    assert hit.score == pytest.approx(config.W_CLAIM * 1.0, abs=1e-6)


def test_query_with_keywords(index):
    r = index.search("sensor", keywords="tire")
    assert r.total_matches == 4
    assert (r.hits[0].doc_number, r.hits[0].claim_number) == ("A", 2)


def test_same_input_same_output(index):
    assert index.search("tire spoke", top_n=6) == index.search("tire spoke", top_n=6)


def test_model_mismatch_message(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"model_name": "other", "model_revision": "x"}))
    with pytest.raises(IndexNotReady, match="python -m patent_search build"):
        Index(tmp_path)


def test_missing_index_message(tmp_path):
    with pytest.raises(IndexNotReady, match="No index found"):
        Index(tmp_path)
