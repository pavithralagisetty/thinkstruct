import pytest

from patent_search import train_summary as T


def claim(doc, n, text=None, indep=True):
    return T.Claim(f"{doc}:{n}", doc, n, text or f"search_document: {doc} claim {n}", indep, 0)


def test_group_keys_use_the_first_code():
    assert T.group_keys("B60B100FI") == ("B60B 1", "B60B")
    assert T.group_keys("B60C1112FI, B60B100FI") == ("B60C 11", "B60C")
    assert T.group_keys(None) == (None, None)


def test_pick_claims_keeps_the_main_claim_and_is_repeatable():
    claims = [claim("A", 1, indep=None), claim("A", 2, indep=True)] + [claim("A", n, indep=False) for n in range(3, 10)]
    picked = T.pick_claims(claims, 5, seed=1)
    assert picked[0].claim_number == 2 and len(picked) == 5
    assert [c.claim_id for c in picked] == [c.claim_id for c in T.pick_claims(claims, 5, seed=1)]


def setup(scores):
    """A in group B60B 1; B also B60B 1; C in B60B 5 (same subclass); D in B60C 11."""
    claims = {d: [claim(d, 1)] for d in "ABCD"}
    info = {"A": {"keys": ("B60B 1", "B60B")}, "B": {"keys": ("B60B 1", "B60B")},
            "C": {"keys": ("B60B 5", "B60B")}, "D": {"keys": ("B60C 11", "B60C")}}
    rows = [T.Row("A", "A:1", "w", "search_query: a", claims["A"][0].text)]
    return rows, claims, info, (lambda doc, c: scores[c.doc_number])


def test_same_group_first_then_fallbacks_with_the_score_check():
    rows, claims, info, score = setup({"A": 1.0, "B": 0.5, "C": 0.5, "D": 0.5})
    assert T.add_negatives(rows, list("ABCD"), claims, info, score)["level_used"]["main_group"] == 1
    assert rows[0].negative_claim_id == "B:1"

    rows, claims, info, score = setup({"A": 1.0, "B": 0.95, "C": 0.5, "D": 0.5})    # B fails the 90% check
    counts = T.add_negatives(rows, list("ABCD"), claims, info, score)
    assert rows[0].negative_claim_id == "C:1" and counts["level_used"]["subclass"] == 1
    assert counts["candidates_skipped_score_check"] == 1

    rows, claims, info, score = setup({"A": 1.0, "B": 0.9, "C": 0.95, "D": 0.5})    # exactly 90% also fails
    assert T.add_negatives(rows, list("ABCD"), claims, info, score)["level_used"]["any"] == 1
    assert rows[0].negative_claim_id == "D:1"


def test_identical_text_is_skipped_and_no_candidate_is_an_error():
    rows, claims, info, score = setup({"A": 1.0, "B": 0.5, "C": 0.5, "D": 0.5})
    claims["B"] = [claim("B", 1, text=rows[0].positive)]
    counts = T.add_negatives(rows, list("ABCD"), claims, info, score)
    assert counts["candidates_skipped_identical_text"] == 1 and rows[0].negative_claim_id == "C:1"
    rows, claims, info, score = setup({"A": 1.0, "B": 0.99, "C": 0.99, "D": 0.99})
    with pytest.raises(ValueError, match="no wrong answer"):
        T.add_negatives(rows, list("ABCD"), claims, info, score)
