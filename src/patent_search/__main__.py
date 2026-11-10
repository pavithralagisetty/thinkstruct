"""Command line: python -m patent_search build"""

import argparse
import logging


def cmd_build(args):
    from patent_search.build_index import build

    counts = build().counts
    print(f"{counts['patents']} patents")
    print(f"{counts['claims']} claims indexed")
    print(f"{counts['patents_without_description']} patents without description")
    print(f"{counts['claims_opening_missing']} claims with missing openings")
    print(f"{counts['cancelled_claim_numbers']} cancelled claim numbers")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m patent_search")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="read the raw dir, clean, embed, write data/index").set_defaults(func=cmd_build)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
