import logging
import sqlite3

import pytest

from patent_search import cpc
from patent_search.cpc import Cpc, matches, parse_query, split_code


@pytest.mark.parametrize("code, expected", [
    ("B60C1112FI", Cpc("B60C", 11, "12")),       # 1/112 is not a reading: 11 is the longest real group
    ("B60B1110FI", Cpc("B60B", 11, "10")),       # Emergency wheels
    ("B60D1167FI", Cpc("B60D", 1, "167")),       # B60D has no group 11
    ("B60C15024FI", Cpc("B60C", 15, "024")),
    ("B60B3300FI", Cpc("B60B", 33, "00")),
    ("B60B100FI", Cpc("B60B", 1, "00")),
    ("b60c 11 12", Cpc("B60C", 11, "12")),
    ("B60C", Cpc("B60C")),
])
def test_split_code(code, expected):
    assert split_code(code) == expected


def test_split_code_empty_and_bad():
    assert split_code(None) is None and split_code("  ") is None
    assert split_code("not a code") is None


def test_unknown_subclass(caplog):
    assert split_code("A01B100FI") == Cpc("A01B", 1, "00")      # only one reading
    with caplog.at_level(logging.WARNING):
        assert split_code("A01B1234FI") == Cpc("A01B")           # 1/234 or 12/34
    assert "can be read 2 ways" in caplog.text


def test_known_subclass_without_fitting_group(caplog):
    with caplog.at_level(logging.WARNING):
        assert split_code("B60D4400FI") == Cpc("B60D")
    assert "no main group" in caplog.text


@pytest.mark.parametrize("text, expected", [
    ("B60C", Cpc("B60C")),
    ("b60c 1", Cpc("B60C", 1)),
    ("B60C 11/12", Cpc("B60C", 11, "12")),
    ("B60C11/12", Cpc("B60C", 11, "12")),
    ("B60C1112", Cpc("B60C", 11, "12")),
    ("B60C1112FI", Cpc("B60C", 11, "12")),
    ("B60C 11/00", Cpc("B60C", 11)),             # /00 = the whole main group
    ("B60C1100", Cpc("B60C", 11)),
])
def test_parse_query(text, expected):
    assert parse_query(text) == expected


@pytest.mark.parametrize("text, message", [
    ("", "Empty"),
    ("B60C 12", "no main group 12"),
    ("B60C 2200/00", "no main group 2200"),      # indexing codes are left out
    ("B60C 11/1", "at least 2 digits"),
    ("spoke", "not a CPC code"),
])
def test_parse_query_errors(text, message):
    with pytest.raises(ValueError, match=message):
        parse_query(text)


def test_matches_whole_parts():
    codes = [split_code(c) for c in ["B60C100FI", "B60C1112FI", "B60C15024FI", "B60C1900FI", "B60B100FI"]]
    def hits(q):
        return [str(c) for c in codes if matches(parse_query(q), c)]
    assert hits("B60C 1") == ["B60C 1/00"]                       # not 11, 15 or 19
    assert hits("B60C") == ["B60C 1/00", "B60C 11/12", "B60C 15/024", "B60C 19/00"]
    assert hits("B60C 15/02") == ["B60C 15/024"]                 # covers subgroups under it
    assert hits("B60C 15/024") == ["B60C 15/024"]
    assert hits("B60C 11/12") == ["B60C 11/12"]
    assert hits("B60C 11/10") == []
    assert not matches(parse_query("B60C 1"), Cpc("B60C"))       # unsplit code has no group
    assert not matches(parse_query("B60C"), None)


def test_titles():
    assert cpc.SUBCLASS_TITLES["B60B"].startswith("VEHICLE WHEELS")
    assert cpc.group_title(Cpc("B60C", 11, "12")) == "Tyre tread bands; Tread patterns; Anti-skid inserts"
    assert all(g in cpc.MAIN_GROUPS[s] for s, g in cpc.GROUP_TITLES)
    assert all(g < 2000 for groups in cpc.MAIN_GROUPS.values() for g in groups)


def test_write_cpc_columns_updates_an_old_db(tmp_path):
    db = tmp_path / "search.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE patents (doc_number TEXT PRIMARY KEY, classification TEXT)")
    con.executemany("INSERT INTO patents VALUES (?, ?)", [("1", "B60C1112FI"), ("2", ""), ("3", "B60D1167FI")])
    con.commit()
    con.close()
    counts = cpc.write_cpc_columns(db)
    assert counts == {"patents": 3, "distinct_codes": 2, "split": 2, "subclass_only": 0, "unreadable": 0}
    con = sqlite3.connect(db)
    assert con.execute("SELECT doc_number, cpc_subclass, cpc_group, cpc_subgroup, cpc_code FROM patents "
                       "ORDER BY doc_number").fetchall() == [
        ("1", "B60C", 11, "12", "B60C 11/12"), ("2", None, None, None, None), ("3", "B60D", 1, "167", "B60D 1/167")]


def test_build_fills_cpc_columns(index):
    con = sqlite3.connect(index.index_dir / "search.db")
    cols = [r[1] for r in con.execute("PRAGMA table_info(patents)")]
    assert cols[-4:] == ["cpc_subclass", "cpc_group", "cpc_subgroup", "cpc_code"]


def test_classification_tree_order_and_counts():
    codes = [split_code(c) for c in ["B60C1510FI", "B60C1502FI", "B60C15024FI", "B60C1500FI", "B60C100FI", None]]
    [b60c] = cpc.classification_tree(codes)
    assert b60c["count"] == 5
    g15 = b60c["groups"][1]
    assert [s["code"] for s in g15["subgroups"]] == ["B60C 15/02", "B60C 15/024", "B60C 15/10"]   # scheme order
    assert [s["count"] for s in g15["subgroups"]] == [2, 1, 1]                                    # 15/02 covers 15/024
    assert g15["count"] == 4
