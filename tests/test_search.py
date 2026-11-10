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


# ---------- filters ----------

from patent_search.search import make_filters  # noqa: E402


def ids(response):
    return [(h.doc_number, h.claim_number) for h in response.hits]


@pytest.mark.parametrize("value, end, expected", [
    ("2024", False, "2024-01-01"), ("2024", True, "2024-12-31"),
    ("2024-02", False, "2024-02-01"), ("2024-02", True, "2024-02-29"),
    ("2024-6", True, "2024-06-30"), ("2024-06-15", True, "2024-06-15"), (2025, False, "2025-01-01"),
])
def test_dates_cover_the_whole_year_or_month(value, end, expected):
    f = make_filters(date_to=value) if end else make_filters(date_from=value)
    assert (f.date_to if end else f.date_from) == expected


@pytest.mark.parametrize("kwargs, message", [
    ({"date_from": "2024/06"}, "date like 2024, 2024-06 or 2024-06-15"),
    ({"date_from": "June 2024"}, "date like"),
    ({"date_to": "2024-13"}, "not a real date"),
    ({"date_to": "2024-02-30"}, "not a real date"),
    ({"date_from": "2025", "date_to": "2024"}, "is after date_to"),
    ({"classification": "B60C 2"}, "no main group 2"),
    ({"classification": "spoke"}, "not a CPC code"),
    ({"classification": 5}, "code or a list of codes"),
    ({"independent_only": "yes"}, "true or false"),
])
def test_bad_filters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_filters(**kwargs)


def test_filter_alone_is_a_search(index):
    r = index.search(classification="B60C")
    assert r.total_matches == 3 and ids(r) == [("A", 1), ("A", 2), ("A", 3)]
    assert all(h.score is None and h.cpc_code == "B60C 11/12" for h in r.hits)


def test_classification_uses_whole_parts_and_any(index):
    assert index.search(classification="B60B 1").total_matches == 2      # B is B60B 1/00
    assert index.search(classification="B60C 1").total_matches == 0      # A is 11/12, not group 1
    assert index.search(classification="B60C 11/12").total_matches == 3
    assert index.search(classification="B60C 11, B60B 1").total_matches == 5
    assert index.search(classification=["B60C 11", "B60B"]).total_matches == 5


def test_dates_are_inclusive_and_skip_unknown_dates(index):
    assert index.search(date_from="2024-06-15", date_to="2024-06-15").total_matches == 3
    assert index.search(date_from="2024-06-16", date_to="2024").total_matches == 0
    # C has no publication date, so any date filter leaves it out
    assert {h.doc_number for h in index.search(date_from="2000", top_n=10).hits} == {"A", "B"}


def test_independent_only_leaves_out_unknown_type(index):
    r = index.search(independent_only=True)
    assert ids(r) == [("A", 1), ("B", 1), ("C", 1)]        # A:3 has a missing opening: unknown type


def test_filters_and_keywords_combine(index):
    r = index.search("spoke", keywords="spoke", classification="B60B", independent_only=True)
    assert ids(r) == [("B", 1)]


# ---------- group by patent ----------

def test_group_by_patent_with_query(index):
    claims = index.search("tire", top_n=10)
    r = index.search("tire", top_n=10, group_by="patent")
    assert r.group_by == "patent" and r.total_matches == 3
    # patents in the order of their best claim, with that claim's score
    first_seen = list(dict.fromkeys(h.doc_number for h in claims.hits))
    assert [h.doc_number for h in r.hits] == first_seen
    a = r.hits[0]
    assert a.doc_number == "A" and a.title == "TIRE" and a.abstract == "A tire."
    assert a.score == claims.hits[0].score and a.matching_claims == 3
    assert [c.claim_number for c in a.claims] == [h.claim_number for h in claims.hits if h.doc_number == "A"][:3]
    assert a.cpc_code == "B60C 11/12" and a.pub_date == "2024-06-15"


def test_group_by_patent_caps_claims_and_counts_patents(index, monkeypatch):
    from patent_search import config as cfg
    monkeypatch.setattr(cfg, "CLAIMS_PER_PATENT", 2)
    r = index.search(keywords="tire", group_by="patent", top_n=1)
    assert r.total_matches == 2                             # A and B have claims with "tire"
    [a] = r.hits
    assert a.doc_number == "A" and a.score is None and a.matching_claims == 3
    assert [c.claim_number for c in a.claims] == [1, 2]     # keyword only: claim number order


def test_group_by_must_be_known(index):
    with pytest.raises(ValueError, match="group_by must be one of"):
        index.search("tire", group_by="patents")


def test_claim_response_says_group_by(index):
    assert index.search("tire").to_dict()["group_by"] == "claim"


def test_highlights_in_claim_and_patent_hits(index):
    r = index.search("tire", keywords="tire*")
    for h in r.hits:
        assert h.highlights and all(h.text[a:b].lower().startswith("tire") for a, b in h.highlights)
    p = index.search("tire", keywords="sensor", group_by="patent")
    [claim] = p.hits[0].claims
    assert [claim.text[a:b] for a, b in claim.highlights] == ["sensor"]
    assert index.search("tire").hits[0].highlights == []
    assert index.search("tire", group_by="patent").hits[0].claims[0].highlights == []
