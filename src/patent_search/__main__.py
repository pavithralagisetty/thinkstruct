"""Command line: python -m patent_search build | search | serve | eval"""

import argparse
import json
import logging
import sys
import textwrap

from patent_search import config


def cmd_build(args):
    from patent_search.build_index import build

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
    try:
        response = search(args.query, keywords or None, args.top)
    except (ValueError, IndexNotReady) as e:
        sys.exit(f"Error: {e}")

    if args.json:
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
        return
    shown = len(response.hits)
    more = f", showing top {shown}" if response.total_matches > shown else ""
    print(f"{response.total_matches} matching claims{more}\n")
    for rank, h in enumerate(response.hits, 1):
        score = f"{h.score:.2f}  " if h.score is not None else ""
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m patent_search")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="read the raw dir, clean, embed, write data/index").set_defaults(func=cmd_build)
    s = sub.add_parser("search", help="search claims by question and/or exact keywords")
    s.add_argument("query", nargs="?", help="plain English question")
    s.add_argument("-k", "--keyword", action="append", default=[],
                   help='exact keyword; repeatable; "double quotes" keep a phrase together')
    s.add_argument("-n", "--top", type=int, default=config.TOP_N_DEFAULT, help="number of results")
    s.add_argument("--json", action="store_true", help="print the response as JSON")
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
