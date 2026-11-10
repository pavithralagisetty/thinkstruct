import json

import pytest

from patent_search import grading

QUESTIONS = [{"id": "q01", "text": "spoke"}, {"id": "q02", "text": "tire sensor"}]
REPORT = {
    "questions": QUESTIONS,
    "results": {
        "bm25": {"q01": [{"claim_id": "B:1", "score": 1.0}, {"claim_id": "B:2", "score": 0.9}],
                 "q02": [{"claim_id": "A:2", "score": 1.0}]},
        "nomic": {"q01": [{"claim_id": "B:2", "score": 0.8}, {"claim_id": "C:1", "score": 0.7}],
                  "q02": [{"claim_id": "A:2", "score": 0.9}, {"claim_id": "A:1", "score": 0.8}]},
    },
}


@pytest.fixture
def setup(index, tmp_path):
    report_path = tmp_path / "run_1.json"
    report_path.write_text(json.dumps(REPORT))
    return report_path, index.index_dir, tmp_path / "judgments"


def write_incoming(judgments, qid, grades):
    folder = judgments / "incoming"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{qid}.json").write_text(json.dumps({"question_id": qid, "grades": [
        {"claim_id": c, "grade": g, "reason": "r"} for c, g in grades.items()]}))


def test_pool_is_union_of_every_top_list():
    assert grading.build_pool(REPORT) == {"q01": ["B:1", "B:2", "C:1"], "q02": ["A:1", "A:2"]}


def test_packet_order_is_fixed_and_random():
    ids = [f"X:{i}" for i in range(20)]
    assert grading.packet_order("q01", ids) == grading.packet_order("q01", list(reversed(ids)))
    assert grading.packet_order("q01", ids) != sorted(ids)


def test_packets_are_blind(setup):
    report_path, index_dir, judgments = setup
    grading.prepare(report_path, index_dir, judgments)
    text = (judgments / "packets" / "q01.json").read_text()
    for word in ("bm25", "nomic", "score", "rank"):
        assert word not in text.lower()
    packet = json.loads(text)
    by_id = {c["claim_id"]: c for c in packet["claims"]}
    assert set(by_id) == {"B:1", "B:2", "C:1"}
    assert set(by_id["B:2"]) == {"claim_id", "patent_title", "main_claim", "claim_text", "opening_missing"}
    assert by_id["B:1"]["main_claim"] is None                 # independent claim: no main claim
    assert by_id["B:2"]["main_claim"] == "A spoke."           # dependent: its root claim's text
    assert by_id["B:2"]["patent_title"] == "SPOKE"


def test_existing_grades_are_reused(setup):
    report_path, index_dir, judgments = setup
    grading.prepare(report_path, index_dir, judgments)
    write_incoming(judgments, "q01", {"B:1": 2, "B:2": 1, "C:1": 0})
    grading.merge_incoming(judgments)
    summary = grading.prepare(report_path, index_dir, judgments)["questions"]
    assert summary["q01"] == {"pool": 3, "already_graded": 3, "to_grade": 0}
    assert not (judgments / "packets" / "q01.json").exists()   # nothing left to grade
    # a new candidate adds one claim: only that claim goes into the packet
    bigger = {**REPORT, "results": {**REPORT["results"], "new": {"q01": [{"claim_id": "A:1", "score": 1}], "q02": []}}}
    report_path.write_text(json.dumps(bigger))
    summary = grading.prepare(report_path, index_dir, judgments)["questions"]
    assert summary["q01"]["to_grade"] == 1
    assert [c["claim_id"] for c in json.loads((judgments / "packets" / "q01.json").read_text())["claims"]] == ["A:1"]


@pytest.mark.parametrize("grades, message", [
    ({"B:1": 2, "B:2": 1}, "not graded"),
    ({"B:1": 2, "B:2": 1, "C:1": 3}, "expected 0, 1 or 2"),
    ({"B:1": 2, "B:2": 1, "C:1": 0, "Z:9": 0}, "not in the packet"),
])
def test_bad_incoming_grades_are_rejected(setup, grades, message):
    report_path, index_dir, judgments = setup
    grading.prepare(report_path, index_dir, judgments)
    write_incoming(judgments, "q01", grades)
    with pytest.raises(ValueError, match=message):
        grading.merge_incoming(judgments)


def test_duplicate_grade_is_rejected(setup):
    report_path, index_dir, judgments = setup
    grading.prepare(report_path, index_dir, judgments)
    (judgments / "incoming").mkdir(parents=True)
    (judgments / "incoming" / "q01.json").write_text(json.dumps({"question_id": "q01", "grades": [
        {"claim_id": c, "grade": 0, "reason": "r"} for c in ["B:1", "B:1", "B:2", "C:1"]]}))
    with pytest.raises(ValueError, match="graded twice"):
        grading.merge_incoming(judgments)


def test_score_counts_ungraded_as_zero(setup, tmp_path):
    report_path, index_dir, judgments = setup
    grading.prepare(report_path, index_dir, judgments)
    write_incoming(judgments, "q01", {"B:1": 2, "B:2": 1, "C:1": 0})   # q02 never graded
    result, out = grading.run_score(report_path, judgments, tmp_path / "results")
    assert result["ungraded"] == {"q02": ["A:1", "A:2"]}
    assert result["per_question"]["bm25"]["q01"]["first_full_rank"] == 1
    assert result["per_question"]["nomic"]["q01"]["first_full_rank"] is None
    assert result["per_question"]["nomic"]["q02"]["mrr"] == 0.0
    assert result["summary"]["bm25"]["full@10"] == 1
    pooled = json.loads((judgments / "pooled_grades.json").read_text())
    assert pooled["grades"]["q01"]["B:1"] == {"grade": 2, "reason": "r"}
    assert json.loads(out.read_text())["summary"] == result["summary"]
