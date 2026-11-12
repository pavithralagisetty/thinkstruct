"""Part 3, second fine tune: summaries as questions, every kind of claim as answers.

* Question: the patent's 1 to 2 sentence summary (config.ADAPTER_SUMMARIES_PATH, one per
  patent), with config.QUERY_PREFIX.
* Right answer: one of the patent's claims, as the index embeds it: config.DOCUMENT_PREFIX plus
  the stored embed_text (Patent / Main claim / Claim lines). Dependent claims are included; each
  patent gives its main claim (lowest numbered independent claim, else its lowest numbered claim)
  plus up to PART3S_CLAIMS_PER_PATENT - 1 others picked with a fixed seed.
* Wrong answer: a claim of another patent in the same split, tried in a seeded random order:
  first the same CPC main group (subclass plus main group, B60B 1), then the same 4 character
  subclass, then any patent. Each patent has exactly one classification code in this data; were
  there several, the first listed would be used. A candidate is skipped when its text equals the
  right answer's, or when base nomic scores the question against it at or above
  PART3S_FALSE_NEGATIVE_RATIO of the score against the right answer (likely also a right answer).
  A level whose candidates are all skipped falls back to the next. The counts are reported.
* Split, training settings and batching (no two rows with the same text in a batch, so a
  patent's rows never meet in one batch) are those of the first fine tune (training.py).

    python -m patent_search.train_summary [--dry_run] [--max_train N] [--batch_size 4]
"""

import argparse
import json
import logging
import random
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from patent_search import config, cpc
from patent_search.training import week_of

log = logging.getLogger(__name__)

LEVELS = ("main_group", "subclass", "any")


@dataclass
class Row:
    doc_number: str
    claim_id: str
    week: str
    query: str                  # with config.QUERY_PREFIX
    positive: str               # with config.DOCUMENT_PREFIX
    negative: str = ""
    negative_claim_id: str = ""
    negative_level: str = ""


@dataclass
class Claim:
    claim_id: str
    doc_number: str
    claim_number: int
    text: str                   # embed_text with config.DOCUMENT_PREFIX
    is_independent: bool | None
    vec_row: int


def group_keys(classification: str | None) -> tuple[str | None, str | None]:
    """(main group key, subclass key) of a patent's first classification code."""
    first = (classification or "").replace(";", ",").split(",")[0].strip()
    code = cpc.split_code(first) if first else None
    if code is None:
        return None, (first[:4] or None)
    return (f"{code.subclass} {code.group}" if code.group is not None else None), code.subclass


def pick_claims(claims: list[Claim], n: int, seed: int) -> list[Claim]:
    """The main claim plus up to n - 1 others, the same every run."""
    ordered = sorted(claims, key=lambda c: c.claim_number)
    main = next((c for c in ordered if c.is_independent), ordered[0])
    rest = [c for c in ordered if c is not main]
    random.Random(f"{seed}:{main.doc_number}").shuffle(rest)
    return [main] + sorted(rest[: n - 1], key=lambda c: c.claim_number)


def load(index_dir: Path, summaries_path: Path):
    """Patents with a summary and claims: their claims, summary, week and group keys."""
    from patent_search.adapter import load_summaries
    with sqlite3.connect(Path(index_dir) / "search.db") as db:
        patents = db.execute("SELECT doc_number, classification, source_file FROM patents ORDER BY doc_number").fetchall()
        rows = db.execute("SELECT claim_id, doc_number, claim_number, embed_text, is_independent, vec_row "
                          "FROM claims ORDER BY doc_number, claim_number").fetchall()
    claims: dict[str, list[Claim]] = {}
    for cid, doc, num, text, indep, vec in rows:
        claims.setdefault(doc, []).append(Claim(cid, doc, num, config.DOCUMENT_PREFIX + text,
                                                None if indep is None else bool(indep), vec))
    docs = [p[0] for p in patents if p[0] in claims]
    summaries = dict(zip(docs, load_summaries(summaries_path, docs)))
    info = {p[0]: {"week": week_of(Path(p[2])), "keys": group_keys(p[1])} for p in patents if p[0] in claims}
    return docs, claims, summaries, info


def summary_vectors(docs: list[str], summaries: dict[str, str]) -> np.ndarray:
    """Base nomic query vectors of the summaries, from the adapter's cache when it is current."""
    from patent_search.adapter import load_data

    class LazyEmbedder:
        def embed_query(self, text):
            if not hasattr(self, "_e"):
                from patent_search.embedder import Embedder
                self._e = Embedder()
            return self._e.embed_query(text)

    data = load_data(config.INDEX_DIR, LazyEmbedder(), config.ADAPTER_SOURCES["summary"]["cache"],
                     config.ADAPTER_SUMMARIES_PATH)
    row_of = {d: i for i, d in enumerate(data.docs)}
    return data.abstracts[[row_of[d] for d in docs]]


def add_negatives(rows: list[Row], split_docs: list[str], claims: dict[str, list[Claim]], info: dict,
                  score, ratio: float = config.PART3S_FALSE_NEGATIVE_RATIO, seed: int = config.PART3_SEED) -> dict:
    """Fill in each row's wrong answer. score(doc, claim) is base nomic's question-claim cosine."""
    pools: dict[str, dict[str, list[Claim]]] = {"main_group": {}, "subclass": {}}
    for doc in split_docs:
        main_group, subclass = info[doc]["keys"]
        for level, key in (("main_group", main_group), ("subclass", subclass)):
            if key:
                pools[level].setdefault(key, []).extend(claims[doc])
    everything = [c for doc in split_docs for c in claims[doc]]
    by_id = {c.claim_id: c for c in everything}
    used, skipped_identical, skipped_score, rows_with_score_skip = Counter(), 0, 0, 0
    for r in rows:
        main_group, subclass = info[r.doc_number]["keys"]
        right = score(r.doc_number, by_id[r.claim_id])
        limit = ratio * right
        candidates = {"main_group": pools["main_group"].get(main_group, []) if main_group else [],
                      "subclass": pools["subclass"].get(subclass, []) if subclass else [],
                      "any": everything}
        chosen, had_score_skip, tried = None, False, set()
        for level in LEVELS:
            pool = [c for c in candidates[level] if c.doc_number != r.doc_number]
            random.Random(f"{seed}:{r.claim_id}:{level}").shuffle(pool)
            for c in pool:
                if c.claim_id in tried:     # already skipped at a narrower level: not counted twice
                    continue
                tried.add(c.claim_id)
                if c.text == r.positive:
                    skipped_identical += 1
                    continue
                if score(r.doc_number, c) >= limit:
                    skipped_score += 1
                    had_score_skip = True
                    continue
                chosen = (level, c)
                break
            if chosen:
                break
        if chosen is None:
            raise ValueError(f"{r.claim_id}: no wrong answer passes the checks")
        level, c = chosen
        r.negative, r.negative_claim_id, r.negative_level = c.text, c.claim_id, level
        used[level] += 1
        rows_with_score_skip += had_score_skip
    return {"rows": len(rows), "level_used": {lv: used[lv] for lv in LEVELS},
            "candidates_skipped_identical_text": skipped_identical,
            "candidates_skipped_score_check": skipped_score, "rows_with_a_score_skip": rows_with_score_skip}


def build(index_dir: Path = config.INDEX_DIR, summaries_path: Path = config.ADAPTER_SUMMARIES_PATH,
          per_patent: int = config.PART3S_CLAIMS_PER_PATENT, seed: int = config.PART3_SEED):
    docs, claims, summaries, info = load(index_dir, summaries_path)
    weeks = sorted({info[d]["week"] for d in docs})
    n_train = round(len(weeks) * config.PART3_TRAIN_FRACTION)
    train_weeks = set(weeks[:n_train])
    split = {"train": [d for d in docs if info[d]["week"] in train_weeks],
             "test": [d for d in docs if info[d]["week"] not in train_weeks]}
    assert not set(split["train"]) & set(split["test"])

    q = summary_vectors(docs, summaries)
    claim_vectors = np.load(Path(index_dir) / "claim_vectors.npy", mmap_mode="r")
    q_row = {d: i for i, d in enumerate(docs)}
    cache: dict[tuple[str, int], float] = {}

    def score(doc: str, c: Claim) -> float:
        k = (doc, c.vec_row)
        if k not in cache:
            cache[k] = float(np.dot(q[q_row[doc]], claim_vectors[c.vec_row]))
        return cache[k]

    words = np.array([len(summaries[d].split()) for d in docs])
    counts = {"summaries": {"per": "patent", "count": len(docs), "median_words": float(np.median(words)),
                            "p95_words": float(np.percentile(words, 95))},
              "claims_available": sum(len(claims[d]) for d in docs), "claims_per_patent_cap": per_patent,
              "classification_rule": "each patent's first listed code (every patent here has exactly one)",
              "weeks": {"train": [min(train_weeks), max(train_weeks)],
                        "test": [weeks[n_train], weeks[-1]]}}
    out = {}
    for name, split_docs in split.items():
        rows = [Row(d, c.claim_id, info[d]["week"], config.QUERY_PREFIX + summaries[d], c.text)
                for d in split_docs for c in pick_claims(claims[d], per_patent, seed)]
        counts[name] = {"patents": len(split_docs),
                        "main_claims": sum(1 for d in split_docs for c in pick_claims(claims[d], per_patent, seed)[:1]),
                        **add_negatives(rows, split_docs, claims, info, score, seed=seed)}
        out[name] = rows
    out["test_claims"] = [c for d in split["test"] for c in claims[d]]
    return out, counts, summaries


# ---- evaluation and training (torch is imported only from here on) ----

def evaluate(model, data: dict, summaries: dict[str, str], batch_size: int = config.PART3_EVAL_BATCH_SIZE) -> dict:
    """Each test summary ranks every claim of every test patent; its own patent's claims are right."""
    test_docs = sorted({r.doc_number for r in data["test"]})
    corpus = data["test_claims"]

    def encode(texts):
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
    q = encode([config.QUERY_PREFIX + summaries[d] for d in test_docs])
    c = encode([x.text for x in corpus])
    owner = np.array([x.doc_number for x in corpus])
    sims = q @ c.T
    hit1, hit10, rr, ndcg = [], [], [], []
    for i, d in enumerate(test_docs):
        order = np.argsort(-sims[i], kind="stable")[:10]
        rel = owner[order] == d
        n_rel = int((owner == d).sum())
        hit1.append(float(rel[0]))
        hit10.append(float(rel.any()))
        rr.append(1.0 / (int(np.argmax(rel)) + 1) if rel.any() else 0.0)
        dcg = float(np.sum(rel / np.log2(np.arange(2, 12))))
        ideal = float(np.sum(1 / np.log2(np.arange(2, 2 + min(n_rel, 10)))))
        ndcg.append(dcg / ideal)
    tq = encode([r.query for r in data["test"]])
    tp = encode([r.positive for r in data["test"]])
    tn = encode([r.negative for r in data["test"]])
    sp, sn = np.sum(tq * tp, axis=1), np.sum(tq * tn, axis=1)
    return {"hit@1": float(np.mean(hit1)), "hit@10": float(np.mean(hit10)), "mrr@10": float(np.mean(rr)),
            "ndcg@10": float(np.mean(ndcg)), "triplet_accuracy": float(np.mean(sp > sn)),
            "mean_sim_correct": float(np.mean(sp)), "mean_sim_wrong": float(np.mean(sn))}


METRICS = [("hit@1", "Own claim ranked 1st"), ("hit@10", "Own claim in top 10"), ("mrr@10", "MRR@10"),
           ("ndcg@10", "nDCG@10"), ("triplet_accuracy", "Triplet accuracy"),
           ("mean_sim_correct", "Mean similarity, right claim"), ("mean_sim_wrong", "Mean similarity, wrong claim")]


def markdown(run: dict) -> str:
    c, t = run["data"], run["training"]
    lines = ["# Part 3 fine tune on summaries", "",
             f"Questions: {c['summaries']['count']} summaries, one per {c['summaries']['per']} "
             f"(median {c['summaries']['median_words']:.0f} words, 95th percentile {c['summaries']['p95_words']:.0f}). "
             f"Train {c['train']['patents']} patents, {c['train']['rows']} rows (weeks {c['weeks']['train'][0]} to "
             f"{c['weeks']['train'][1]}); test {c['test']['patents']} patents, {c['test']['rows']} rows "
             f"({c['weeks']['test'][0]} to {c['weeks']['test'][1]}). Up to {c['claims_per_patent_cap']} claims per patent, "
             f"main claim first. Training: {t['steps']} steps, batch {run['settings']['batch_size']}, "
             f"{t['seconds']} s on {t['device']}.", "",
             "Wrong answers (" + c["classification_rule"] + "):", ""]
    for name in ("train", "test"):
        s = c[name]
        lines.append(f"* {name}: main group {s['level_used']['main_group']}, subclass {s['level_used']['subclass']}, "
                     f"any patent {s['level_used']['any']}; candidates skipped: {s['candidates_skipped_score_check']} by "
                     f"the score check ({s['rows_with_a_score_skip']} rows had one), "
                     f"{s['candidates_skipped_identical_text']} with identical text")
    lines += ["", "Test: each test summary ranks every claim of the test patents; its own patent's claims are "
              "right. Both models read 512 tokens here.", "",
              "| Metric | Base | Fine tuned | Change |", "|---|---|---|---|"]
    for key, label in METRICS:
        b, f = run["base"][key], run["finetuned"][key]
        lines.append(f"| {label} | {b:.4f} | {f:.4f} | {f - b:+.4f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    from patent_search import training
    parser = argparse.ArgumentParser(description="Part 3: fine tune nomic with summaries as questions")
    parser.add_argument("--dry_run", action="store_true", help="build the rows and print the counts only")
    parser.add_argument("--batch_size", type=int, default=config.PART3_BATCH_SIZE)
    parser.add_argument("--max_train", type=int, default=None, help="use only the first N training rows (quick test)")
    parser.add_argument("--output_dir", type=Path, default=config.PART3S_MODEL_OUT)
    parser.add_argument("--results_dir", type=Path, default=config.PART3_RESULTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("sentence_transformers", "transformers", "httpx", "datasets", "patent_search.adapter"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data, counts, summaries = build()
    print(json.dumps(counts, indent=2))
    rows_path = Path(config.ADAPTER_SUMMARIES_PATH).with_name("train_rows.jsonl")
    with rows_path.open("w") as f:
        for name in ("train", "test"):
            for r in data[name]:
                f.write(json.dumps({"split": name, **asdict(r)}) + "\n")
    print(f"rows written to {rows_path}")
    if args.dry_run:
        return
    train_rows = data["train"][: args.max_train] if args.max_train else data["train"]

    training.require_mps()
    print("scoring the base model ...")
    base = evaluate(training.load_model(config.NOMIC_MODEL_NAME), data, summaries)
    training.free_mps_memory()
    print(f"fine tuning on {len(train_rows)} rows ...")
    trained = training.train(train_rows, args.output_dir, args.batch_size)
    print(f"trained {trained['steps']} steps in {trained['seconds']} s; saved to {args.output_dir}")
    training.free_mps_memory()
    print("scoring the fine tuned model ...")
    finetuned = evaluate(training.load_model(str(args.output_dir)), data, summaries)
    run = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "settings": {"base_model": config.NOMIC_MODEL_NAME, "base_revision": config.NOMIC_MODEL_REVISION,
                        "output_dir": str(args.output_dir), "max_seq_length": config.PART3_MAX_SEQ_LENGTH,
                        "batch_size": args.batch_size, "epochs": config.PART3_EPOCHS,
                        "learning_rate": config.PART3_LEARNING_RATE, "warmup_ratio": config.PART3_WARMUP_RATIO,
                        "seed": config.PART3_SEED, "max_train": args.max_train,
                        "false_negative_ratio": config.PART3S_FALSE_NEGATIVE_RATIO},
           "data": counts, "training": trained, "base": base, "finetuned": finetuned,
           "change": {k: finetuned[k] - base[k] for k in base}}
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "part3_summary_results.json").write_text(json.dumps(run, indent=2) + "\n")
    (args.results_dir / "part3_summary_results.md").write_text(markdown(run))
    print()
    print(markdown(run))


if __name__ == "__main__":
    main()
