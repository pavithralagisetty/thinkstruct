"""Part 3: build training and test data from the patent files, score the base nomic model,
fine tune it, score it again on the same test set, and report before vs after.

Each patent gives one row: query (its abstract, or title with --query_field title), correct
answer (its first claim) and wrong answer (the first claim of another patent in the same split
with the same 4 character classification prefix, else any other patent in that split). The
oldest 80% of weekly files train, the newest 20% test. Prefixes are written into the strings
themselves, so training, evaluation and search all see identical text.

Run: python part3_train_eval.py [--query_field abstract|title] [--batch_size 16] [--max_train N]
"""

import argparse
import json
import logging
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from patent_search import config
from patent_search.claims import parse_claims
from patent_search.load import parse_record, remove_reference_numerals

log = logging.getLogger(__name__)

WEEK_RE = re.compile(r"ipa(\d{6})")


@dataclass
class Row:
    doc_number: str
    week: str                   # YYYY-MM-DD from the file name
    prefix: str                 # first PART3_CLASS_PREFIX_LEN characters of the classification
    query: str                  # with config.QUERY_PREFIX
    positive: str               # first claim, with config.DOCUMENT_PREFIX
    negative: str = ""
    negative_doc: str = ""
    negative_same_prefix: bool = False


def week_of(path: Path) -> str:
    match = WEEK_RE.search(path.name)
    if not match:
        raise ValueError(f"{path.name}: no ipaYYMMDD week in the file name")
    d = match.group(1)
    return f"20{d[:2]}-{d[2:4]}-{d[4:]}"


def load_rows(data_dir: Path, query_field: str = "abstract") -> tuple[list[Row], dict]:
    """One row per usable patent (no negative yet), plus counts of what was skipped and why."""
    if query_field not in config.PART3_QUERY_FIELDS:
        raise ValueError(f"query_field must be one of {config.PART3_QUERY_FIELDS}")
    files = sorted(Path(data_dir).glob(config.RAW_GLOB), key=week_of)
    if not files:
        raise FileNotFoundError(f"No {config.RAW_GLOB} files in {data_dir}")

    # Dedupe first, with the same rule as the search index: the later publication wins.
    kept, records, duplicates, no_number = {}, 0, 0, 0
    for path in files:
        for raw in json.loads(path.read_text(encoding="utf-8")):
            records += 1
            patent = parse_record(raw, path.name) if isinstance(raw, dict) else None
            if patent is None:
                no_number += 1
                continue
            old = kept.get(patent.doc_number)
            if old is not None:
                duplicates += 1
                if (old[0].pub_date or "") > (patent.pub_date or ""):
                    continue
            kept[patent.doc_number] = (patent, week_of(path))

    skipped, opening_missing = Counter(), 0
    rows = []
    for patent, week in kept.values():
        first = parse_claims(patent.doc_number, patent.claims).claims if patent.claims else []
        missing = [name for name, empty in (("abstract", not patent.abstract), ("claims", not first),
                                            ("classification", not patent.classification),
                                            ("title", query_field == "title" and not patent.title)) if empty]
        if missing:
            skipped[missing[0]] += 1          # counted once, under the first missing field
            continue
        opening_missing += first[0].opening_missing
        query = patent.abstract if query_field == "abstract" else patent.title
        rows.append(Row(
            doc_number=patent.doc_number, week=week,
            prefix=patent.classification[: config.PART3_CLASS_PREFIX_LEN],
            query=config.QUERY_PREFIX + query,
            positive=config.DOCUMENT_PREFIX + remove_reference_numerals(first[0].text),
        ))
    rows.sort(key=lambda r: (r.week, r.doc_number))
    counts = {"files": len(files), "records": records, "without_doc_number": no_number,
              "duplicates_removed": duplicates, "unique_patents": len(kept),
              "skipped": dict(sorted(skipped.items())), "usable": len(rows),
              "first_claim_opening_missing": opening_missing}
    for reason, n in sorted(skipped.items()):
        log.info("skipped %d patents: no %s", n, reason)
    return rows, counts


def split_by_week(rows: list[Row], train_fraction: float = config.PART3_TRAIN_FRACTION):
    """Oldest weeks train, newest weeks test. A patent has one week, so no patent is in both."""
    weeks = sorted({r.week for r in rows})
    n_train = round(len(weeks) * train_fraction)
    if not 0 < n_train < len(weeks):
        raise ValueError(f"{len(weeks)} weeks cannot be split {train_fraction:.0%} / rest")
    train_weeks = set(weeks[:n_train])
    train = [r for r in rows if r.week in train_weeks]
    test = [r for r in rows if r.week not in train_weeks]
    return train, test, weeks[:n_train], weeks[n_train:]


def add_negatives(rows: list[Row], seed: int = config.PART3_SEED) -> None:
    """Wrong answer from the same split: a random other patent with the same classification
    prefix, else any other patent. A candidate with the identical claim text is never used."""
    rng = random.Random(seed)
    by_prefix: dict[str, list[Row]] = {}
    for r in rows:
        by_prefix.setdefault(r.prefix, []).append(r)
    for r in rows:
        same = [c for c in by_prefix[r.prefix] if c.doc_number != r.doc_number and c.positive != r.positive]
        pool = same or [c for c in rows if c.doc_number != r.doc_number and c.positive != r.positive]
        if not pool:
            raise ValueError(f"{r.doc_number}: no other patent to use as a wrong answer")
        pick = rng.choice(pool)
        r.negative, r.negative_doc, r.negative_same_prefix = pick.positive, pick.doc_number, bool(same)


def build_splits(data_dir: Path, query_field: str, seed: int = config.PART3_SEED):
    rows, counts = load_rows(data_dir, query_field)
    train, test, train_weeks, test_weeks = split_by_week(rows)
    add_negatives(train, seed)
    add_negatives(test, seed)
    assert not {r.doc_number for r in train} & {r.doc_number for r in test}
    counts["split"] = {
        "train_weeks": len(train_weeks), "test_weeks": len(test_weeks),
        "train_week_range": [train_weeks[0], train_weeks[-1]],
        "test_week_range": [test_weeks[0], test_weeks[-1]],
        "train_patents": len(train), "test_patents": len(test),
        "train_negatives_same_prefix": sum(r.negative_same_prefix for r in train),
        "test_negatives_same_prefix": sum(r.negative_same_prefix for r in test),
    }
    return train, test, counts


# ---- model, evaluation and training (torch is imported only from here on) ----

def require_mps():
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("Part 3 trains on Apple silicon: torch.backends.mps.is_available() is False. "
                         "Run it on an Apple silicon Mac with a torch build that has MPS.")


def free_mps_memory() -> None:
    """Return memory held by models that are no longer used; MPS keeps it cached otherwise,
    and training nomic next to a freed copy runs out of memory on a 16 GB Mac."""
    import gc
    import torch
    gc.collect()
    torch.mps.empty_cache()


def load_model(path_or_name: str, device: str = "mps"):
    from sentence_transformers import SentenceTransformer
    if path_or_name == config.NOMIC_MODEL_NAME:
        pinned = {"code_revision": config.NOMIC_CODE_REVISION}
        model = SentenceTransformer(path_or_name, revision=config.NOMIC_MODEL_REVISION, trust_remote_code=True,
                                    model_kwargs=pinned, config_kwargs=pinned, device=device)
    else:
        model = SentenceTransformer(path_or_name, trust_remote_code=True, device=device)
    model.max_seq_length = config.PART3_MAX_SEQ_LENGTH
    return model


def evaluate(model, test: list[Row], batch_size: int = config.PART3_EVAL_BATCH_SIZE) -> dict:
    import numpy as np
    from sentence_transformers import util
    from sentence_transformers.evaluation import InformationRetrievalEvaluator, TripletEvaluator

    # Each test query searches the first claims of all test patents; only its own is relevant.
    ir = InformationRetrievalEvaluator(
        queries={r.doc_number: r.query for r in test}, corpus={r.doc_number: r.positive for r in test},
        relevant_docs={r.doc_number: {r.doc_number} for r in test},
        accuracy_at_k=[1, 10], precision_recall_at_k=[1, 10], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
        score_functions={"cosine": util.cos_sim}, batch_size=batch_size, name="ir", write_csv=False)
    ir_scores = ir(model)
    triplet = TripletEvaluator([r.query for r in test], [r.positive for r in test], [r.negative for r in test],
                               similarity_fn_names=["cosine"], batch_size=batch_size, name="triplet",
                               write_csv=False)
    triplet_scores = triplet(model)

    def encode(texts):
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
    q, p, n = encode([r.query for r in test]), encode([r.positive for r in test]), encode([r.negative for r in test])
    return {
        "recall@1": ir_scores["ir_cosine_recall@1"],
        "recall@10": ir_scores["ir_cosine_recall@10"],
        "mrr@10": ir_scores["ir_cosine_mrr@10"],
        "ndcg@10": ir_scores["ir_cosine_ndcg@10"],
        "triplet_accuracy": triplet_scores["triplet_cosine_accuracy"],
        "mean_sim_correct": float(np.mean(np.sum(q * p, axis=1))),
        "mean_sim_wrong": float(np.mean(np.sum(q * n, axis=1))),
    }


def localize_remote_code(model_dir: Path) -> list[str]:
    """Make a saved nomic folder load without the Hub: the model's code files sit in the folder
    and auto_map in config.json points at them. Returns the files copied (empty if none needed)."""
    config_path = model_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    auto_map = cfg.get("auto_map", {})
    remote = {k: v for k, v in auto_map.items() if "--" in v}
    if not remote:
        return []
    from huggingface_hub import hf_hub_download
    copied = []
    for key, ref in remote.items():
        repo, target = ref.split("--", 1)
        module = target.rsplit(".", 1)[0] + ".py"
        if not (model_dir / module).exists():
            src = hf_hub_download(repo, module, revision=config.NOMIC_CODE_REVISION)
            shutil.copy(src, model_dir / module)
            copied.append(module)
        auto_map[key] = target
    cfg["auto_map"] = auto_map
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return copied


def train(train_rows: list[Row], out_dir: Path, batch_size: int) -> dict:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    if Path(out_dir).resolve() == Path(config.NOMIC_MODEL_NAME).resolve():
        raise ValueError("refusing to write over the base model")
    model = load_model(config.NOMIC_MODEL_NAME)
    dataset = Dataset.from_dict({"anchor": [r.query for r in train_rows],
                                 "positive": [r.positive for r in train_rows],
                                 "negative": [r.negative for r in train_rows]})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(config.PART3_CHECKPOINT_DIR), num_train_epochs=config.PART3_EPOCHS,
        per_device_train_batch_size=batch_size, learning_rate=config.PART3_LEARNING_RATE,
        warmup_ratio=config.PART3_WARMUP_RATIO, seed=config.PART3_SEED, data_seed=config.PART3_SEED,
        fp16=False, bf16=False, batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", logging_steps=1, report_to="none", disable_tqdm=True)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset,
                                         loss=MultipleNegativesRankingLoss(model))
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model.save(str(out_dir))
    copied = localize_remote_code(out_dir)
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    return {"seconds": round(seconds, 1), "steps": trainer.state.global_step, "device": str(model.device),
            "losses": [round(x, 4) for x in losses], "code_files_copied": copied}


# ---- reporting ----

METRICS = [("recall@1", "Recall@1"), ("recall@10", "Recall@10"), ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10"),
           ("triplet_accuracy", "Triplet accuracy"), ("mean_sim_correct", "Mean similarity, correct pairs"),
           ("mean_sim_wrong", "Mean similarity, wrong pairs")]


def markdown(results: dict) -> str:
    lines = ["# Part 3 results", ""]
    for field, run in results["runs"].items():
        d, s, t = run["data"], run["data"]["split"], run["training"]
        lines += [f"## Query field: {field}", "",
                  f"Train {s['train_patents']} patents ({s['train_weeks']} weeks, {s['train_week_range'][0]} to "
                  f"{s['train_week_range'][1]}), test {s['test_patents']} patents ({s['test_weeks']} weeks, "
                  f"{s['test_week_range'][0]} to {s['test_week_range'][1]}). Skipped: {d['skipped'] or 'none'}; "
                  f"duplicates removed: {d['duplicates_removed']}. Training: {t['steps']} steps, batch "
                  f"{run['settings']['batch_size']}, {t['seconds']} s on {t['device']}.", "",
                  "| Metric | Base | Fine tuned | Change |", "|---|---|---|---|"]
        for key, label in METRICS:
            b, f = run["base"][key], run["finetuned"][key]
            lines.append(f"| {label} | {b:.4f} | {f:.4f} | {f - b:+.4f} |")
        if run.get("note"):
            lines += ["", f"Note: {run['note']}"]
        lines.append("")
    return "\n".join(lines)


def save_results(run: dict, results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "part3_results.json"
    results = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    results["runs"][run["settings"]["query_field"]] = run
    path.write_text(json.dumps(results, indent=2) + "\n")
    md = results_dir / "part3_results.md"
    md.write_text(markdown(results))
    return path, md


def main(argv=None):
    parser = argparse.ArgumentParser(description="Part 3: fine tune nomic and compare with the base model")
    parser.add_argument("--data_dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--query_field", choices=config.PART3_QUERY_FIELDS, default="abstract")
    parser.add_argument("--batch_size", type=int, default=config.PART3_BATCH_SIZE)
    parser.add_argument("--max_train", type=int, default=None, help="use only the first N training rows (quick test)")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help=f"where to save the fine tuned model (default {config.PART3_MODEL_OUT}, "
                             "with -title for --query_field title)")
    parser.add_argument("--results_dir", type=Path, default=config.PART3_RESULTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("sentence_transformers", "transformers", "httpx", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out_dir = args.output_dir or (config.PART3_MODEL_OUT if args.query_field == "abstract"
                                  else config.PART3_MODEL_OUT.with_name(config.PART3_MODEL_OUT.name + "-title"))

    require_mps()
    train_rows, test_rows, counts = build_splits(args.data_dir, args.query_field)
    s = counts["split"]
    print(f"{counts['records']} records in {counts['files']} weekly files, {counts['duplicates_removed']} duplicates "
          f"removed, skipped {counts['skipped'] or 'none'}, {counts['usable']} usable patents")
    print(f"train: {s['train_patents']} patents, {s['train_weeks']} weeks {s['train_week_range'][0]}..{s['train_week_range'][1]}"
          f" ({s['train_negatives_same_prefix']} wrong answers share the prefix)")
    print(f"test:  {s['test_patents']} patents, {s['test_weeks']} weeks {s['test_week_range'][0]}..{s['test_week_range'][1]}"
          f" ({s['test_negatives_same_prefix']} wrong answers share the prefix)")
    if args.max_train:
        train_rows = train_rows[: args.max_train]
        print(f"quick test: training on the first {len(train_rows)} rows")

    print("scoring the base model ...")
    base = evaluate(load_model(config.NOMIC_MODEL_NAME), test_rows)
    note = None
    if base["recall@10"] >= config.PART3_RECALL_CEILING:
        note = (f"base Recall@10 is {base['recall@10']:.3f} (at or above {config.PART3_RECALL_CEILING}), "
                "so the abstract task is close to solved; rerun with --query_field title")
        print("note:", note)
    print("fine tuning ...")
    free_mps_memory()
    training = train(train_rows, out_dir, args.batch_size)
    print(f"trained {training['steps']} steps in {training['seconds']} s; saved to {out_dir}")
    free_mps_memory()
    print("scoring the fine tuned model ...")
    finetuned = evaluate(load_model(str(out_dir)), test_rows)

    run = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "settings": {"query_field": args.query_field, "base_model": config.NOMIC_MODEL_NAME,
                        "base_revision": config.NOMIC_MODEL_REVISION, "output_dir": str(out_dir),
                        "max_seq_length": config.PART3_MAX_SEQ_LENGTH, "batch_size": args.batch_size,
                        "epochs": config.PART3_EPOCHS, "learning_rate": config.PART3_LEARNING_RATE,
                        "warmup_ratio": config.PART3_WARMUP_RATIO, "seed": config.PART3_SEED,
                        "max_train": args.max_train, "query_prefix": config.QUERY_PREFIX,
                        "document_prefix": config.DOCUMENT_PREFIX},
           "data": counts, "training": training, "base": base, "finetuned": finetuned,
           "change": {k: finetuned[k] - base[k] for k in base}, "note": note}
    path, md = save_results(run, args.results_dir)
    print()
    print(markdown({"runs": {args.query_field: run}}))
    print(f"saved {path} and {md}")
