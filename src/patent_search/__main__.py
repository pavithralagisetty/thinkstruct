"""Command line: python -m patent_search build | search | serve | eval | grade"""

import argparse
import json
import logging
import sys
import textwrap

from patent_search import config


def cmd_build(args):
    from patent_search.build_index import build

    if args.cpc_only:
        from patent_search.cpc import write_cpc_columns

        if not config.DB_PATH.exists():
            sys.exit(f"Error: no index at {config.DB_PATH}. Run: python -m patent_search build")
        c = write_cpc_columns(config.DB_PATH)
        print(f"{c['distinct_codes']} distinct classification codes: {c['split']} split, "
              f"{c['subclass_only']} subclass only, {c['unreadable']} unreadable ({c['patents']} patents)")
        return
    counts = build().counts
    print(f"{counts['patents']} patents")
    print(f"{counts['claims']} claims indexed")
    print(f"{counts['patents_without_description']} patents without description")
    print(f"{counts['claims_opening_missing']} claims with missing openings")
    print(f"{counts['cancelled_claim_numbers']} cancelled claim numbers")


def claim_label(is_independent):
    return {True: "main claim", False: "dependent claim"}.get(is_independent, "")


def cmd_search(args):
    from patent_search.keywords import parse_keywords
    from patent_search.search import IndexNotReady, search

    keywords = [k for option in args.keyword for k in parse_keywords(option)]
    title_abstract = [k for option in args.title_abstract for k in parse_keywords(option)]
    try:
        response = search(args.query, keywords or None, args.top, args.classification or None,
                          args.date_from, args.date_to, args.main_only, args.by, title_abstract or None)
    except (ValueError, IndexNotReady) as e:
        sys.exit(f"Error: {e}")

    if args.json:
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
        return
    shown = len(response.hits)
    unit = "patents" if response.group_by == "patent" else "claims"
    more = f", showing top {shown}" if response.total_matches > shown else ""
    print(f"{response.total_matches} matching {unit}{more}\n")
    for rank, h in enumerate(response.hits, 1):
        score = f"{h.score:.2f}  " if h.score is not None else ""
        if response.group_by == "patent":
            print(f"{rank}. {score}{h.doc_number}  {h.title}")
            print(f"   {h.cpc_code or h.classification} · published {h.pub_date} · "
                  f"{h.matching_claims} matching claim{'s' if h.matching_claims != 1 else ''}")
            for c in h.claims:
                label = claim_label(c.is_independent)
                cscore = f" {c.score:.2f}" if c.score is not None else ""
                print(f"   - claim {c.claim_number}{cscore}{'  [' + label + ']' if label else ''}")
                print(textwrap.indent(textwrap.fill(textwrap.shorten(c.text, 300), 96), "     "))
            print()
            continue
        label = claim_label(h.is_independent)
        print(f"{rank}. {score}{h.doc_number} claim {h.claim_number}"
              f"{'  [' + label + ']' if label else ''}  {h.title}")
        print(f"   {h.classification} · published {h.pub_date}")
        print(textwrap.indent(textwrap.fill(h.text, 100), "   "))
        if h.opening_missing:
            print("   (The start of this claim is missing in the source data.)")
        print()


def cmd_serve(args):
    import uvicorn

    uvicorn.run("patent_search.api:app", host=args.host, port=args.port)


def cmd_eval(args):
    from patent_search import evaluate
    from patent_search.search import Index, IndexNotReady

    try:
        index = Index()
        report = evaluate.run(candidates=args.models, device=args.device, top_n=args.top, index=index)
    except (ValueError, IndexNotReady) as e:
        sys.exit(f"Error: {e}")
    path = evaluate.save_report(report)

    for q in report["questions"]:
        print(f"{q['id']}  {q['text']}")
        for name, results in report["results"].items():
            for rank, hit in enumerate(results[q["id"]][:3], 1):
                title = index.patents[hit["claim_id"].split(":")[0]]["title"]
                print(f"  {name if rank == 1 else '':13} {rank}. {hit['claim_id']:15} {hit['score']:.3f}  {title[:60]}")
        print()
    for name, info in report["candidates"].items():
        print(f"{name:13} {info['seconds']:7.1f}s  {info.get('vectors', '')}")
    print(f"\nReport: {path}")


def cmd_grade(args):
    from patent_search import grading

    try:
        if args.action == "prepare":
            out = grading.prepare(args.report)
            print(f"Pool from {out['report']}")
            for qid, n in out["questions"].items():
                print(f"  {qid}: {n['pool']:3} pooled, {n['already_graded']:3} already graded, {n['to_grade']:3} to grade")
            return
        result, path = grading.run_score(args.report)
    except (ValueError, FileNotFoundError, KeyError) as e:
        sys.exit(f"Error: {e}")

    if result["merged_now"]:
        print("Merged grades: " + ", ".join(f"{q} {n}" for q, n in result["merged_now"].items()))
    if result["ungraded"]:
        n = sum(len(v) for v in result["ungraded"].values())
        print(f"WARNING: {n} pooled claims have no grade and count as 0: "
              + ", ".join(f"{q} ({len(v)})" for q, v in result["ungraded"].items()))
    print(f"\n{'model':13} {'Hit@5':>6} {'P@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG@10':>8} {'Full@10':>8}")
    for name, m in result["summary"].items():
        fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
        print(f"{name:13} {fmt(m['hit@5']):>6} {fmt(m['p@5']):>6} {fmt(m['recall@10']):>6} "
              f"{fmt(m['mrr']):>6} {fmt(m['ndcg@10']):>8} {str(m['full@10']) + '/' + str(len(result['pool'])):>8}")
    names = list(result["per_question"])
    print("\nRank of the first fully relevant claim (- = none in the top 10)")
    print(f"{'':5}" + "".join(f"{n:>13}" for n in names))
    for qid in result["pool"]:
        ranks = [result["per_question"][n][qid]["first_full_rank"] for n in names]
        print(f"{qid:5}" + "".join(f"{(r if r is not None else '-'):>13}" for r in ranks))
    print(f"\nScores: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m patent_search")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="read the raw dir, clean, embed, write data/index")
    b.add_argument("--cpc-only", action="store_true",
                   help="only (re)write the cpc_* columns of an existing search.db; no re-embedding")
    b.set_defaults(func=cmd_build)
    s = sub.add_parser("search", help="search claims by question and/or exact keywords")
    s.add_argument("query", nargs="?", help="plain English question")
    s.add_argument("-k", "--keyword", action="append", default=[],
                   help='exact keyword; repeatable; "double quotes" keep a phrase together')
    s.add_argument("-t", "--title-abstract", action="append", default=[], metavar="KEYWORD",
                   help="exact keyword that must be in the patent's title or abstract; repeatable")
    s.add_argument("-n", "--top", type=int, default=config.TOP_N_DEFAULT, help="number of results")
    s.add_argument("--json", action="store_true", help="print the response as JSON")
    s.add_argument("--by", choices=["claim", "patent"], default="claim",
                   help="list claims (default) or group them by patent")
    s.add_argument("-c", "--classification", action="append", default=[], metavar="CODE",
                   help='CPC filter, e.g. "B60C 11/12"; repeatable, commas allowed; any code matches')
    s.add_argument("--from", dest="date_from", metavar="DATE", help="published on or after (2024, 2024-06, 2024-06-15)")
    s.add_argument("--to", dest="date_to", metavar="DATE", help="published on or before (inclusive)")
    s.add_argument("--main-only", action="store_true",
                   help="only independent (main) claims; claims with a missing opening are left out")
    s.set_defaults(func=cmd_search)
    v = sub.add_parser("serve", help="run the web page and API")
    v.add_argument("--host", default=config.HOST)
    v.add_argument("--port", type=int, default=config.PORT)
    v.set_defaults(func=cmd_serve)
    e = sub.add_parser("eval", help="compare models: top claims per question per model")
    e.add_argument("--models", nargs="+", metavar="NAME",
                   help=f"candidates to run (default: bm25 {' '.join(config.EVAL_MODELS)})")
    e.add_argument("--device", help="torch device for the models (default: mps when available, else cpu)")
    e.add_argument("-n", "--top", type=int, default=config.EVAL_TOP_N, help="claims saved per question")
    e.set_defaults(func=cmd_eval)
    g = sub.add_parser("grade", help="pooled blind grading of the eval results")
    g.add_argument("action", choices=["prepare", "score"],
                   help="prepare: write blind packets; score: merge grades and score every model")
    g.add_argument("--report", help="eval report to use (default: latest data/eval_results/run_*.json)")
    g.set_defaults(func=cmd_grade)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
