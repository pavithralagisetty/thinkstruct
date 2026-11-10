"""Pooled blind grading: the only source of relevance labels for the model comparison.

1. prepare: pool = union of every candidate's top claims per question (latest eval report).
   Write a blind packet per question with only the pooled claims that have no grade yet.
2. Judges (blind subagents) read a packet and write data/eval_judgments/incoming/<qid>.json.
3. score: merge incoming grades into grades/<qid>.json, check every pooled claim has exactly
   one grade, save pooled_grades.json and score every candidate with the evaluate metrics.
"""

import hashlib
import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from patent_search import config, evaluate

SCALE = {
    2: "The claim itself states what the question asks.",
    1: "Right topic and a closely related detail, but the claim does not state what is asked.",
    0: "Not relevant.",
}
INSTRUCTIONS = (
    "Grade each claim for the question. Judge the claim's own text (claim_text). "
    "main_claim, when present, is only context for a dependent claim; what it says does not "
    "count for this claim. When opening_missing is true the start of the claim was lost in the "
    "source data; grade the text that is there. Grades: "
    + " ".join(f"{g} = {d}" for g, d in SCALE.items())
    + " Give one short reason per claim."
)
MAX_REASON = 300


def _dir(judgments_dir) -> Path:
    return Path(judgments_dir or config.EVAL_JUDGMENTS_DIR)


def latest_report(results_dir: Path | None = None) -> Path:
    reports = sorted(Path(results_dir or config.EVAL_RESULTS_DIR).glob("run_*.json"))
    if not reports:
        raise FileNotFoundError("No eval report found. Run: python -m patent_search eval")
    return reports[-1]


def build_pool(report: dict) -> dict[str, list[str]]:
    """{question id: sorted claim ids returned by any candidate}."""
    pool = {}
    for q in report["questions"]:
        ids = {h["claim_id"] for results in report["results"].values() for h in results.get(q["id"], [])}
        pool[q["id"]] = sorted(ids)
    return pool


def packet_order(qid: str, claim_ids: list[str]) -> list[str]:
    """A fixed random order, the same every time for the same question and claims."""
    seed = int(hashlib.sha256(qid.encode()).hexdigest()[:16], 16)
    ids = sorted(claim_ids)
    random.Random(seed).shuffle(ids)
    return ids


def claim_details(index_dir: Path, claim_ids: list[str]) -> dict[str, dict]:
    """Title, main claim (dependent claims only), claim text and opening_missing."""
    con = sqlite3.connect(Path(index_dir) / "search.db")
    con.row_factory = sqlite3.Row
    try:
        details = {}
        for claim_id in claim_ids:
            c = con.execute("SELECT c.*, p.title FROM claims c JOIN patents p USING (doc_number) "
                            "WHERE claim_id = ?", (claim_id,)).fetchone()
            if c is None:
                raise KeyError(f"{claim_id} is not in the index; rebuild or rerun eval")
            main = None
            if c["parent_number"] is not None and c["root_number"] != c["claim_number"]:
                root = con.execute("SELECT text FROM claims WHERE doc_number = ? AND claim_number = ?",
                                   (c["doc_number"], c["root_number"])).fetchone()
                main = root["text"] if root else None
            details[claim_id] = {
                "claim_id": claim_id,
                "patent_title": c["title"],
                "main_claim": main,
                "claim_text": c["text"],
                "opening_missing": bool(c["opening_missing"]),
            }
        return details
    finally:
        con.close()


def load_grades(qid: str, judgments_dir=None) -> dict[str, dict]:
    path = _dir(judgments_dir) / "grades" / f"{qid}.json"
    if not path.exists():
        return {}
    return {g["claim_id"]: {"grade": g["grade"], "reason": g["reason"]}
            for g in json.loads(path.read_text())["grades"]}


def _write_grades(qid: str, question: str, grades: dict[str, dict], judgments_dir=None):
    folder = _dir(judgments_dir) / "grades"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"claim_id": c, **grades[c]} for c in sorted(grades)]
    (folder / f"{qid}.json").write_text(json.dumps(
        {"question_id": qid, "question": question, "grades": rows}, indent=2, ensure_ascii=False))


def prepare(report_path: Path | None = None, index_dir: Path | None = None, judgments_dir=None) -> dict:
    """Write a packet per question for the pooled claims that have no grade yet."""
    report_path = Path(report_path or latest_report())
    report = json.loads(report_path.read_text())
    pool = build_pool(report)
    packets = _dir(judgments_dir) / "packets"
    packets.mkdir(parents=True, exist_ok=True)
    summary = {}
    for q in report["questions"]:
        qid = q["id"]
        graded = load_grades(qid, judgments_dir)
        missing = [c for c in pool[qid] if c not in graded]
        path = packets / f"{qid}.json"
        if missing:
            details = claim_details(index_dir or config.INDEX_DIR, missing)
            path.write_text(json.dumps({
                "question_id": qid,
                "question": q["text"],
                "instructions": INSTRUCTIONS,
                "claims": [details[c] for c in packet_order(qid, missing)],
            }, indent=2, ensure_ascii=False))
        else:
            path.unlink(missing_ok=True)
        summary[qid] = {"pool": len(pool[qid]), "already_graded": len(pool[qid]) - len(missing),
                        "to_grade": len(missing)}
    return {"report": report_path.name, "questions": summary}


def validate_incoming(packet: dict, incoming: dict) -> dict[str, dict]:
    """Judged grades for exactly the packet's claims, each once, grade 0/1/2 with a reason."""
    expected = [c["claim_id"] for c in packet["claims"]]
    if incoming.get("question_id") != packet["question_id"]:
        raise ValueError(f"{packet['question_id']}: incoming grades are for {incoming.get('question_id')!r}")
    seen: dict[str, dict] = {}
    for row in incoming.get("grades", []):
        cid, grade, reason = row.get("claim_id"), row.get("grade"), str(row.get("reason") or "").strip()
        if cid in seen:
            raise ValueError(f"{packet['question_id']}: {cid} graded twice")
        if cid not in expected:
            raise ValueError(f"{packet['question_id']}: {cid} is not in the packet")
        if grade not in SCALE or isinstance(grade, bool):
            raise ValueError(f"{packet['question_id']}: {cid} has grade {grade!r}, expected 0, 1 or 2")
        if not reason:
            raise ValueError(f"{packet['question_id']}: {cid} has no reason")
        seen[cid] = {"grade": grade, "reason": reason[:MAX_REASON]}
    missing = [c for c in expected if c not in seen]
    if missing:
        raise ValueError(f"{packet['question_id']}: {len(missing)} claim(s) not graded: {', '.join(missing[:5])}")
    return seen


def merge_incoming(judgments_dir=None) -> dict[str, int]:
    """Move validated incoming grades into grades/<qid>.json. Existing grades are never changed."""
    root = _dir(judgments_dir)
    merged = {}
    for path in sorted((root / "incoming").glob("*.json")):
        incoming = json.loads(path.read_text())
        qid = incoming.get("question_id") or path.stem
        packet_path = root / "packets" / f"{qid}.json"
        if not packet_path.exists():
            raise ValueError(f"{path.name}: no packet for {qid}; run grade prepare first")
        packet = json.loads(packet_path.read_text())
        new = validate_incoming(packet, incoming)
        grades = load_grades(qid, judgments_dir)
        clash = set(new) & set(grades)
        if clash:
            raise ValueError(f"{qid}: already graded: {', '.join(sorted(clash)[:5])}")
        grades.update(new)
        _write_grades(qid, packet["question"], grades, judgments_dir)
        path.unlink()
        packet_path.unlink()
        merged[qid] = len(new)
    return merged


def pooled_grades(report: dict, judgments_dir=None) -> tuple[dict, dict[str, list[str]]]:
    """{qid: {claim_id: {grade, reason}}} for the pool, and the pooled claims still ungraded."""
    pool = build_pool(report)
    grades, ungraded = {}, {}
    for qid, ids in pool.items():
        graded = load_grades(qid, judgments_dir)
        grades[qid] = {c: graded[c] for c in ids if c in graded}
        if missing := [c for c in ids if c not in graded]:
            ungraded[qid] = missing
    return grades, ungraded


def score(report: dict, grades: dict) -> dict:
    """Metrics per candidate per question, and means. Claims never graded count as 0."""
    per_question, summary = {}, {}
    for name, results in report["results"].items():
        per_question[name] = {}
        for q in report["questions"]:
            ranked = [h["claim_id"] for h in results[q["id"]]]
            g = {c: v["grade"] for c, v in grades.get(q["id"], {}).items()}
            per_question[name][q["id"]] = evaluate.question_metrics(ranked, g)
        rows = per_question[name].values()

        def mean(key):
            values = [r[key] for r in rows if r[key] is not None]
            return round(sum(values) / len(values), 4) if values else None
        summary[name] = {
            "hit@5": mean("hit@5"), "p@5": mean("p@5"), "recall@10": mean("recall@10"),
            "mrr": mean("mrr"), "ndcg@10": mean("ndcg@10"),
            "full@10": sum(r["first_full_rank"] is not None for r in rows),
        }
    return {"summary": summary, "per_question": per_question}


def run_score(report_path: Path | None = None, judgments_dir=None, results_dir=None) -> tuple[dict, Path]:
    report_path = Path(report_path or latest_report(results_dir))
    report = json.loads(report_path.read_text())
    merged = merge_incoming(judgments_dir)
    grades, ungraded = pooled_grades(report, judgments_dir)
    root = _dir(judgments_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pooled_grades.json").write_text(json.dumps(
        {"report": report_path.name, "scale": SCALE, "grades": grades}, indent=2, ensure_ascii=False))
    result = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": report_path.name,
        "merged_now": merged,
        "pool": {qid: len(ids) for qid, ids in build_pool(report).items()},
        "ungraded": ungraded,
        "grade_counts": {qid: {str(k): sum(v["grade"] == k for v in g.values()) for k in SCALE}
                         for qid, g in grades.items()},
        **score(report, grades),
    }
    out_dir = Path(results_dir or config.EVAL_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"scores_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result, out
