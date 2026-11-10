import logging

from patent_search.claims import parse_claims, split_number


def nums(parsed):
    return [c.claim_number for c in parsed.claims]


def test_split_number():
    assert split_number("17. The tire") == (17, 17, "The tire")
    assert split_number("6 - 9. (canceled)") == (6, 9, "(canceled)")
    assert split_number("1.- 15. (canceled)") == (1, 15, "(canceled)")
    assert split_number("1 - 15 : (canceled)") == (1, 15, "(canceled)")
    assert split_number("1 - 10 (canceled)") == (1, 10, "(canceled)")
    assert split_number("18 : The molded body") == (18, 18, "The molded body")
    assert split_number("2 ) The spring seat") == (2, 2, "The spring seat")
    assert split_number("0.5 phr of an accelerator") is None
    assert split_number("1:2 mixture of") is None
    assert split_number("a base; a column") is None


def test_cancelled_ranges_use_up_numbers():
    p = parse_claims("X", ["1. A tire.", "2 - 4. (canceled)", "5. (canceled)", "6. The tire of claim 1."])
    assert nums(p) == [1, 6]
    assert p.cancelled_numbers == {2, 3, 4, 5}


def test_position_recovery_and_flags():
    p = parse_claims("X", ["a base; a column", "2. The bassinet of claim 1, wherein", "a hub"])
    first, second, third = p.claims
    assert (first.claim_number, first.opening_missing, first.is_independent, first.parent_number) == (1, True, None, None)
    assert (second.parent_number, second.root_number, second.is_independent) == (1, 1, False)
    assert (third.claim_number, third.opening_missing) == (3, True)


def test_backward_numbering_for_gap(caplog):
    with caplog.at_level(logging.WARNING):
        p = parse_claims("X", ["1. A.", "8. The A of claim 1.", "a gap claim", "11. The A of claim 10."])
    assert nums(p) == [1, 8, 10, 11]
    assert caplog.records


def test_duplicate_number_gets_next_free(caplog):
    p = parse_claims("X", ["1. A.", "2. B.", "2. C."])
    assert nums(p) == [1, 2, 3]
    assert p.claims[2].text == "C."


def test_root_follows_parents_and_stops_at_cancelled():
    p = parse_claims("X", ["1. A.", "2. (canceled)", "3. The A of claim 2.", "4. The A of claim 3.",
                           "5. The A of claims 1 to 3."])
    by = {c.claim_number: c for c in p.claims}
    assert by[4].root_number == 3 and by[3].root_number == 3
    assert by[5].parent_number == 1 and by[5].root_number == 1


def test_loop_guard(caplog):
    p = parse_claims("X", ["1. The A of claim 2.", "2. The A of claim 1."])
    assert {c.root_number for c in p.claims} <= {1, 2}
