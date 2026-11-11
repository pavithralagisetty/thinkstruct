"""Time searches with and without the hybrid filters.

    python -m patent_search.benchmark [REPEATS]

Each case is searched for every evaluation question, REPEATS times (default 20) after one
warm-up, through the production Index.search. Embedding the question costs the same whatever
the filters are, so it is timed on its own and the searches reuse a vector embedded up front.
Each filtered case is also timed without a question ("filters only": the exact step alone).
"""

import platform
import statistics
import sys
import time

from patent_search.embedder import Embedder
from patent_search.evaluate import load_questions
from patent_search.search import Index

CASES = [
    ("no filters (semantic only)", {}),
    ("no filters, grouped by patent", {"group_by": "patent"}),
    ("classification B60B", {"classification": "B60B"}),
    ("classification B60C 11", {"classification": "B60C 11"}),
    ("claim keyword tire*", {"keywords": "tire*"}),
    ("claim keyword sipe", {"keywords": "sipe"}),
    ("title or abstract keyword wheel*", {"title_abstract_keywords": "wheel*"}),
    ("published 2024-06 to 2024-12", {"date_from": "2024-06", "date_to": "2024-12"}),
    ("main claims only", {"independent_only": True}),
    ("all: B60B + spoke* + wheel* + 2024", {"classification": "B60B", "keywords": "spoke*",
                                            "title_abstract_keywords": "wheel*", "date_from": "2024", "date_to": "2024"}),
]


class CachedEmbedder:
    """Query vectors embedded once up front, so search timings leave out the model."""

    def __init__(self, embedder, questions):
        self.vectors = {q.strip(): embedder.embed_query(q.strip()) for q in questions}

    def embed_query(self, text):
        return self.vectors[text]


def timed(fn, repeats):
    fn()                                            # warm-up, not counted
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return times


def stats(ms):
    ms = sorted(ms)
    return statistics.median(ms), ms[round(0.95 * (len(ms) - 1))]


def main(repeats=20):
    start = time.perf_counter()
    index = Index()
    load_s = time.perf_counter() - start
    start = time.perf_counter()
    embedder = Embedder()
    model_s = time.perf_counter() - start
    questions = [q["text"] for q in load_questions()]
    print(f"{len(index.claims)} claims, {len(questions)} questions x {repeats} repeats per case "
          f"({platform.platform()}, {platform.machine()})")
    print(f"Startup: index {load_s:.2f} s, model {model_s:.2f} s (once per process)\n")

    embed = [t for q in questions for t in timed(lambda: embedder.embed_query(q.strip()), repeats)]
    fast = index.with_store(index.vectors, CachedEmbedder(embedder, questions))

    print(f"{'case':38} {'claims scored':>13} {'median ms':>10} {'p95 ms':>8} {'vs none':>8} {'filters only ms':>16}")
    base = None
    for name, kwargs in CASES:
        median, p95 = stats([t for q in questions for t in timed(lambda: fast.search(q, **kwargs), repeats)])
        base = base or median
        filters = {k: v for k, v in kwargs.items() if k != "group_by"}
        scored = fast.search(questions[0], **filters).total_matches        # same for every question
        only = f"{stats(timed(lambda: fast.search(None, **filters), repeats * len(questions)))[0]:.2f}" if filters else "-"
        print(f"{name:38} {scored:>13} {median:>10.2f} {p95:>8.2f} {median / base:>7.2f}x {only:>16}")
    median, p95 = stats(embed)
    print(f"\nEmbedding one question with the model (same for every case): median {median:.1f} ms, p95 {p95:.1f} ms")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
