import json

import pytest

from patent_search import config
from patent_search.training import add_negatives, build_splits, load_rows, localize_remote_code, split_by_week


def patent(doc, cls="B60B100FI", abstract="An abstract.", claims=None, title="T", date="20240101"):
    return {"doc_number": doc, "title": title, "abstract": abstract, "classification": cls,
            "filename": f"US{doc}A1-{date}.XML", "claims": claims if claims is not None else [f"1. Claim of {doc} ( 1 )."]}


@pytest.fixture
def data_dir(tmp_path):
    weeks = {  # 5 weekly files: 4 train, 1 test
        "240101": [patent("A1"), patent("A2", cls="B60C1112FI"), patent("NOABS", abstract="")],
        "240108": [patent("B1"), patent("NOCLS", cls=""), patent("CANCEL", claims=["1. (canceled)"])],
        "240115": [patent("C1", cls="B60C500FI"), patent("DUP", abstract="old", date="20240115")],
        "240122": [patent("D1", title="")],
        "240129": [patent("E1"), patent("E2"), patent("E3", cls="B60D100FI"),
                   patent("DUP", abstract="newer", date="20240129")],
    }
    for d, records in weeks.items():
        (tmp_path / f"patents_ipa{d}.json").write_text(json.dumps(records))
    return tmp_path


def test_rows_skip_missing_fields_and_dedupe(data_dir):
    rows, counts = load_rows(data_dir)
    assert counts["skipped"] == {"abstract": 1, "claims": 1, "classification": 1}
    assert counts["duplicates_removed"] == 1 and counts["unique_patents"] == 12 and counts["usable"] == 9
    dup = next(r for r in rows if r.doc_number == "DUP")
    assert dup.query == config.QUERY_PREFIX + "newer" and dup.week == "2024-01-29"   # later publication kept
    a1 = next(r for r in rows if r.doc_number == "A1")
    assert a1.positive == config.DOCUMENT_PREFIX + "Claim of A1."     # first claim, reference numeral removed
    assert a1.prefix == "B60B"


def test_title_queries_also_skip_missing_titles(data_dir):
    rows, counts = load_rows(data_dir, "title")
    assert counts["skipped"]["title"] == 1 and "D1" not in {r.doc_number for r in rows}
    assert all(r.query == config.QUERY_PREFIX + "T" for r in rows)
    with pytest.raises(ValueError):
        load_rows(data_dir, "claims")


def test_split_by_week_oldest_train_newest_test(data_dir):
    train, test, counts = build_splits(data_dir, "abstract")
    assert {r.week for r in test} == {"2024-01-29"} and max(r.week for r in train) < "2024-01-29"
    assert not {r.doc_number for r in train} & {r.doc_number for r in test}
    assert counts["split"]["train_weeks"] == 4 and counts["split"]["test_weeks"] == 1


def test_negatives_same_split_same_prefix_then_fallback(data_dir):
    train, test, _ = build_splits(data_dir, "abstract")
    for split in (train, test):
        docs = {r.doc_number: r for r in split}
        for r in split:
            assert r.negative_doc in docs and r.negative_doc != r.doc_number     # same split, other patent
            assert r.negative == docs[r.negative_doc].positive != r.positive
            assert r.negative_same_prefix == (docs[r.negative_doc].prefix == r.prefix)
    e3 = next(r for r in test if r.doc_number == "E3")                           # only B60D patent in test
    assert not e3.negative_same_prefix
    assert all(r.negative_same_prefix for r in test if r.prefix == "B60B")


def test_negatives_are_repeatable(data_dir):
    a, _, _ = build_splits(data_dir, "abstract")
    b, _, _ = build_splits(data_dir, "abstract")
    assert [r.negative_doc for r in a] == [r.negative_doc for r in b]


def test_split_needs_two_parts(data_dir):
    rows, _ = load_rows(data_dir)
    with pytest.raises(ValueError):
        split_by_week([r for r in rows if r.week == "2024-01-29"])
    single = [r for r in rows if r.doc_number in ("A1",)]
    with pytest.raises(ValueError):
        add_negatives(single)


def test_localize_remote_code_points_auto_map_at_local_files(tmp_path):
    (tmp_path / "configuration_hf_nomic_bert.py").write_text("# code")
    (tmp_path / "modeling_hf_nomic_bert.py").write_text("# code")
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {
        "AutoConfig": "nomic-ai/nomic-bert-2048--configuration_hf_nomic_bert.NomicBertConfig",
        "AutoModel": "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel"}}))
    assert localize_remote_code(tmp_path) == []                 # files already there: nothing downloaded
    assert json.loads((tmp_path / "config.json").read_text())["auto_map"] == {
        "AutoConfig": "configuration_hf_nomic_bert.NomicBertConfig",
        "AutoModel": "modeling_hf_nomic_bert.NomicBertModel"}
    assert localize_remote_code(tmp_path) == []                 # already local: unchanged

