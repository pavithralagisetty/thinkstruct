"""Search over the built index: exact keyword and field filters, then semantic ranking."""

import calendar
import copy
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

from patent_search import config, cpc
from patent_search.keywords import compile_keywords, highlight_spans, matches_all

REBUILD = "Rebuild it with: python -m patent_search build"


class IndexNotReady(RuntimeError):
    """The index is missing or does not match the current config."""


GROUP_BY = ("claim", "patent")
_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")


@dataclass
class SearchHit:
    doc_number: str
    title: str
    claim_number: int
    text: str
    score: float | None
    is_independent: bool | None
    opening_missing: bool
    pub_date: str | None
    classification: str
    cpc_code: str | None
    source_file: str
    highlights: list[list[int]]     # [start, end) keyword matches in text


@dataclass
class PatentClaim:
    claim_number: int
    text: str
    score: float | None
    is_independent: bool | None
    opening_missing: bool
    highlights: list[list[int]]     # [start, end) keyword matches in text


@dataclass
class PatentHit:
    doc_number: str
    title: str
    abstract: str
    score: float | None             # the score of its best claim
    matching_claims: int            # its claims that passed step 1
    claims: list[PatentClaim]       # its best CLAIMS_PER_PATENT claims, in ranked order
    pub_date: str | None
    classification: str
    cpc_code: str | None
    source_file: str


@dataclass
class SearchResponse:
    total_matches: int              # claims, or patents when group_by is "patent"
    hits: list
    group_by: str = "claim"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Filters:
    """Field filters applied in step 1 together with the keywords (AND)."""
    classification: tuple[cpc.Cpc, ...] = ()        # any of these
    date_from: str | None = None                    # inclusive, YYYY-MM-DD
    date_to: str | None = None                      # inclusive, YYYY-MM-DD
    independent_only: bool = False
    title_abstract: tuple[re.Pattern, ...] = ()      # every keyword in the title or the abstract

    @property
    def active(self) -> bool:
        return bool(self.classification or self.date_from or self.date_to or self.independent_only
                    or self.title_abstract)

    @property
    def by_patent(self) -> bool:
        """True when a filter decides per patent (all of them except independent_only)."""
        return bool(self.classification or self.date_from or self.date_to or self.title_abstract)


def _date_bound(value, name: str, end: bool) -> str | None:
    """'2024', '2024-06' or '2024-06-15'. A partial date covers the whole year or month."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    match = _DATE_RE.match(text)
    if not match:
        raise ValueError(f"{name} must be a date like 2024, 2024-06 or 2024-06-15, not {text!r}.")
    year, month, day = int(match.group(1)), match.group(2), match.group(3)
    try:
        if month is None:
            d = date(year, 12, 31) if end else date(year, 1, 1)
        elif day is None:
            m = int(month)
            if not 1 <= m <= 12:
                raise ValueError
            d = date(year, m, calendar.monthrange(year, m)[1] if end else 1)
        else:
            d = date(year, int(month), int(day))
    except ValueError:
        raise ValueError(f"{name} {text!r} is not a real date.") from None
    return d.isoformat()


def make_filters(classification: str | list[str] | None = None, date_from=None, date_to=None,
                 independent_only: bool = False,
                 title_abstract_keywords: str | list[str] | None = None) -> Filters:
    """Build and check the filters. Several classification codes (a list, or one string with
    commas) mean any of them. Title or abstract keywords follow the claim keyword rules; each
    one must be in the title or in the abstract. Raises ValueError for bad input."""
    if isinstance(classification, str):
        classification = [classification]
    if classification is not None and not isinstance(classification, (list, tuple)):
        raise ValueError("classification must be a code or a list of codes.")
    codes = []
    for item in classification or []:
        if not isinstance(item, str):
            raise ValueError("classification must be a code or a list of codes.")
        for part in item.split(","):
            if part.strip():
                codes.append(cpc.parse_query(part))
    start = _date_bound(date_from, "date_from", end=False)
    stop = _date_bound(date_to, "date_to", end=True)
    if start and stop and start > stop:
        raise ValueError(f"date_from ({start}) is after date_to ({stop}).")
    if not isinstance(independent_only, bool):
        raise ValueError("independent_only must be true or false.")
    if title_abstract_keywords is not None and not (
            isinstance(title_abstract_keywords, str)
            or (isinstance(title_abstract_keywords, (list, tuple))
                and all(isinstance(k, str) for k in title_abstract_keywords))):
        raise ValueError("title_abstract_keywords must be a string or a list of strings.")
    title_abstract = tuple(compile_keywords(title_abstract_keywords))
    return Filters(tuple(dict.fromkeys(codes)), start, stop, independent_only, title_abstract)


class VectorStore:
    """The only code that touches vectors, so it can move to FAISS or Qdrant later."""

    def __init__(self, claim_vectors: np.ndarray, patent_vectors: np.ndarray):
        self.claim_vectors = claim_vectors
        self.patent_vectors = patent_vectors

    def similarities(self, query_vector: np.ndarray, claim_rows: np.ndarray,
                     patent_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cosine similarity of the query to the given claim and patent rows.
        A patent row of -1 means the patent has no vector and scores 0."""
        q = query_vector.astype(np.float32)
        claim_sim = self.claim_vectors[claim_rows] @ q
        has_vec = patent_rows >= 0
        patent_sim = np.zeros(len(patent_rows), dtype=np.float32)
        patent_sim[has_vec] = self.patent_vectors[patent_rows[has_vec]] @ q
        return claim_sim, patent_sim


class Index:
    """Everything needed at search time, loaded once into memory."""

    def __init__(self, index_dir: Path | None = None, embedder=None):
        index_dir = Path(index_dir or config.INDEX_DIR)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            raise IndexNotReady(f"No index found in {index_dir}. {REBUILD}")
        meta = json.loads(meta_path.read_text())
        wanted = (config.MODEL_NAME, config.MODEL_REVISION, config.MODEL_CODE_REVISION)
        built = (meta.get("model_name"), meta.get("model_revision"), meta.get("model_code_revision"))
        if embedder is None and built != wanted:
            raise IndexNotReady(
                f"The index was built with model {built[0]} (revision {built[1]}, code {built[2]}), "
                f"but config uses {wanted[0]} (revision {wanted[1]}, code {wanted[2]}). {REBUILD}")
        self.index_dir = index_dir
        self.meta = meta
        self._embedder = embedder

        con = sqlite3.connect(index_dir / "search.db")
        con.row_factory = sqlite3.Row
        try:
            self.patents = {r["doc_number"]: dict(r) for r in con.execute(
                "SELECT doc_number, title, abstract, classification, pub_date, source_file, vec_row FROM patents")}
            self.claims = [dict(r) for r in con.execute(
                "SELECT doc_number, claim_number, text, is_independent, opening_missing, vec_row "
                "FROM claims ORDER BY doc_number, claim_number")]
        finally:
            con.close()

        self.vectors = VectorStore(np.load(index_dir / "claim_vectors.npy"),
                                   np.load(index_dir / "patent_vectors.npy"))
        if len(self.vectors.claim_vectors) != len(self.claims):
            raise IndexNotReady(f"claim_vectors.npy does not match search.db. {REBUILD}")
        # Classification split once per patent (same rule as the stored cpc_* columns).
        for p in self.patents.values():
            p["cpc"] = cpc.split_code(p["classification"])
            p["cpc_code"] = str(p["cpc"]) if p["cpc"] else None
        self.claim_rows = np.array([c["vec_row"] for c in self.claims], dtype=np.int64)
        self.patent_rows = np.array(
            [v if (v := self.patents[c["doc_number"]]["vec_row"]) is not None else -1 for c in self.claims],
            dtype=np.int64)

    @property
    def embedder(self):
        if self._embedder is None:
            from patent_search.embedder import Embedder
            embedder = Embedder()
            if config.QUERY_ADAPTER:
                # Question -> model -> query adapter -> normalized; claim vectors stay as built.
                from patent_search.adapter import AdaptedQuery, QueryAdapter, check_adapter
                try:
                    adapter = QueryAdapter.load(config.QUERY_ADAPTER)
                    check_adapter(adapter, self.meta)
                except (FileNotFoundError, ValueError) as e:
                    raise IndexNotReady(str(e)) from e
                embedder = AdaptedQuery(embedder, adapter)
            self._embedder = embedder
        return self._embedder

    def with_store(self, vectors, embedder) -> "Index":
        """The same claims and patents searched with other vectors (model comparison).
        `vectors` needs a VectorStore-shaped similarities(); `embedder` an embed_query()."""
        other = copy.copy(self)
        other.vectors = vectors
        other._embedder = embedder
        return other

    def hit(self, i: int, score: float | None, patterns=()) -> SearchHit:
        c = self.claims[i]
        p = self.patents[c["doc_number"]]
        return SearchHit(
            doc_number=c["doc_number"], title=p["title"], claim_number=c["claim_number"],
            text=c["text"], score=score,
            is_independent=None if c["is_independent"] is None else bool(c["is_independent"]),
            opening_missing=bool(c["opening_missing"]), pub_date=p["pub_date"],
            classification=p["classification"], cpc_code=p["cpc_code"], source_file=p["source_file"],
            highlights=highlight_spans(c["text"], patterns),
        )

    def classification_tree(self) -> list[dict]:
        """Classification dropdown data, built once from the patents in the index."""
        if getattr(self, "_classification_tree", None) is None:
            self._classification_tree = cpc.classification_tree([p["cpc"] for p in self.patents.values()])
        return self._classification_tree

    def patent_hit(self, doc_number: str, ranked: list[tuple[float | None, int]], patterns=()) -> PatentHit:
        """ranked: this patent's matching claims as (score, claim index), best first."""
        p = self.patents[doc_number]
        claims = []
        for score, i in ranked[:config.CLAIMS_PER_PATENT]:
            c = self.claims[i]
            claims.append(PatentClaim(
                claim_number=c["claim_number"], text=c["text"], score=score,
                is_independent=None if c["is_independent"] is None else bool(c["is_independent"]),
                opening_missing=bool(c["opening_missing"]),
                highlights=highlight_spans(c["text"], patterns)))
        return PatentHit(
            doc_number=doc_number, title=p["title"], abstract=p["abstract"], score=ranked[0][0],
            matching_claims=len(ranked), claims=claims, pub_date=p["pub_date"],
            classification=p["classification"], cpc_code=p["cpc_code"], source_file=p["source_file"],
        )

    def _patent_passes(self, p: dict, filters: Filters) -> bool:
        if filters.classification and not any(cpc.matches(q, p["cpc"]) for q in filters.classification):
            return False
        if filters.date_from and not (p["pub_date"] and p["pub_date"] >= filters.date_from):
            return False
        if filters.date_to and not (p["pub_date"] and p["pub_date"] <= filters.date_to):
            return False
        # Each keyword in the title or the abstract, checked separately so a phrase cannot
        # run from the end of the title into the abstract.
        if not all(k.search(p["title"] or "") or k.search(p["abstract"] or "") for k in filters.title_abstract):
            return False
        return True

    def search(self, query: str | None = None, keywords: str | list[str] | None = None,
               top_n: int = config.TOP_N_DEFAULT, classification: str | list[str] | None = None,
               date_from=None, date_to=None, independent_only: bool = False,
               group_by: str = "claim",
               title_abstract_keywords: str | list[str] | None = None) -> SearchResponse:
        if group_by not in GROUP_BY:
            raise ValueError(f"group_by must be one of: {', '.join(GROUP_BY)}.")
        if not isinstance(top_n, int) or isinstance(top_n, bool) or not 1 <= top_n <= config.TOP_N_MAX:
            raise ValueError(f"top_n must be a whole number from 1 to {config.TOP_N_MAX}.")
        matched, ranked, patterns = self._rank(query, keywords, classification, date_from, date_to,
                                               independent_only, title_abstract_keywords)

        if group_by == "claim":
            return SearchResponse(len(matched), [self.hit(i, s, patterns) for s, i in ranked[:top_n]], "claim")

        groups = self._by_patent(ranked)
        hits = [self.patent_hit(d, g, patterns) for d, g in list(groups.items())[:top_n]]
        return SearchResponse(len(groups), hits, "patent")

    def patent_matches(self, query: str | None = None, keywords: str | list[str] | None = None,
                       classification: str | list[str] | None = None, date_from=None, date_to=None,
                       independent_only: bool = False,
                       title_abstract_keywords: str | list[str] | None = None) -> list[dict]:
        """Every patent with a claim that passes the search, ranked as group_by="patent" ranks them,
        not cut at TOP_N_MAX. Used by the map, which lights up matches among all patents."""
        _, ranked, _ = self._rank(query, keywords, classification, date_from, date_to,
                                  independent_only, title_abstract_keywords)
        return [{"doc_number": d, "score": g[0][0], "matching_claims": len(g),
                 "best_claim": self.claims[g[0][1]]["claim_number"]}
                for d, g in self._by_patent(ranked).items()]

    def _by_patent(self, ranked: list[tuple[float | None, int]]) -> dict[str, list[tuple[float | None, int]]]:
        """Group ranked claims by patent in ranked order: a patent ranks by its best claim."""
        groups: dict[str, list[tuple[float | None, int]]] = {}
        for s, i in ranked:
            groups.setdefault(self.claims[i]["doc_number"], []).append((s, i))
        return groups

    def _rank(self, query, keywords, classification, date_from, date_to, independent_only,
              title_abstract_keywords):
        """Steps 1 and 2: the claims that pass, and them as (score, claim index), best first."""
        query = query.strip() if query else None
        patterns = compile_keywords(keywords)
        filters = make_filters(classification, date_from, date_to, independent_only, title_abstract_keywords)
        if not query and not patterns and not filters.active:
            raise ValueError("Give a query, at least one keyword, or a filter.")

        # Step 1: exact keywords on the claim's own display text, AND the field filters.
        allowed = None
        if filters.by_patent:
            allowed = {d for d, p in self.patents.items() if self._patent_passes(p, filters)}
        matched = [
            i for i, c in enumerate(self.claims)
            if (allowed is None or c["doc_number"] in allowed)
            and (not filters.independent_only or c["is_independent"] == 1)
            and (not patterns or matches_all(c["text"], patterns))
        ]

        if not query:
            # Step 2b: no question. self.claims is already in doc_number, claim_number order.
            ranked = [(None, i) for i in matched]
        elif not matched:
            ranked = []
        else:
            # Step 2a: weighted claim and patent similarity.
            rows = np.array(matched, dtype=np.int64)
            q = self.embedder.embed_query(query)
            claim_sim, patent_sim = self.vectors.similarities(q, self.claim_rows[rows], self.patent_rows[rows])
            scores = config.W_CLAIM * claim_sim.astype(np.float64) + config.W_PATENT * patent_sim.astype(np.float64)
            ranked = [(round(float(s), 6), i) for s, i in zip(scores, matched)]
            # Ties: doc_number then claim_number, which is the order of i.
            ranked.sort(key=lambda t: (-t[0], t[1]))
        return matched, ranked, patterns


@lru_cache(maxsize=1)
def get_index() -> Index:
    return Index()


def search(query: str | None = None, keywords: str | list[str] | None = None,
           top_n: int = config.TOP_N_DEFAULT, classification: str | list[str] | None = None,
           date_from=None, date_to=None, independent_only: bool = False,
           group_by: str = "claim",
           title_abstract_keywords: str | list[str] | None = None) -> SearchResponse:
    return get_index().search(query, keywords, top_n, classification, date_from, date_to,
                              independent_only, group_by, title_abstract_keywords)
