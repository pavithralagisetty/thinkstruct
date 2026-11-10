import json

import pytest

from patent_search import load


@pytest.mark.parametrize("value", [None, "", "   \n", [], [""], ["", "  "], [None]])
def test_empty_values(value):
    assert load.is_empty(value)


@pytest.mark.parametrize("value", ["x", ["", "x"], 0])
def test_not_empty(value):
    assert not load.is_empty(value)


def test_clean_text_spacing():
    assert load.clean_text("The spoke of claim 1 , wherein") == "The spoke of claim 1, wherein"
    assert load.clean_text("1 . A spoke star .") == "1. A spoke star."
    assert load.clean_text("from 0 . 75 to 0 . 90") == "from 0.75 to 0.90"
    assert load.clean_text("a  b\n c") == "a b c"


def test_remove_reference_numerals():
    f = load.remove_reference_numerals
    assert f("a hub ( 1 ) and a rim ( 2a ), wherein") == "a hub and a rim, wherein"
    assert f("elements ( 56 A, 56 B; 57 A, 57 B) form") == "elements form"
    assert f("tread ( 1, 1 a ).") == "tread."
    # real text in brackets stays
    assert f("formula (1) (2018) step (i) ( X )") == "formula (1) (2018) step (i) ( X )"


def test_parse_filename():
    assert load.parse_filename("US20240051333A1-20240215.XML") == ("A1", "2024-02-15")
    assert load.parse_filename("garbage") == (None, None)


def _write(tmp_path, name, records):
    (tmp_path / name).write_text(json.dumps(records))


def test_missing_fields_and_description(tmp_path):
    _write(tmp_path, "patents_1.json", [
        {"doc_number": "1", "filename": "US1A1-20240101.XML", "claims": ["1 . A thing ."],
         "detailed_description": ["", "Para one", " ", "Para two"]},
        {"doc_number": "2", "title": None, "abstract": "  ", "claims": None,
         "detailed_description": [""]},
        {"title": "no number"},
    ])
    a, b = load.load_patents(tmp_path)
    assert a.description == "Para one\n\nPara two" and a.has_description
    assert a.claims == ["1. A thing."]
    assert (b.title, b.abstract, b.claims, b.description, b.has_description) == ("", "", [], "", False)
    assert b.kind_code is None and b.pub_date is None


def test_duplicate_keeps_later_publication(tmp_path):
    _write(tmp_path, "patents_1.json", [{"doc_number": "7", "title": "new",
                                        "filename": "US7A9-20240301.XML"}])
    _write(tmp_path, "patents_2.json", [{"doc_number": "7", "title": "old",
                                        "filename": "US7A1-20240101.XML"}])
    [p] = load.load_patents(tmp_path)
    assert (p.title, p.kind_code, p.pub_date) == ("new", "A9", "2024-03-01")
