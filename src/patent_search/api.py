"""HTTP API: POST /search, the patent map and the web page at GET /."""

import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from patent_search import config, search as search_module
from patent_search.search import IndexNotReady

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Patent Claim Search")
_index_lock = threading.Lock()


class SearchRequest(BaseModel):
    query: str | None = None
    keywords: str | list[str] | None = None
    top_n: int = config.TOP_N_DEFAULT
    classification: str | list[str] | None = None
    date_from: str | int | None = None
    date_to: str | int | None = None
    independent_only: bool = False
    group_by: str = "claim"
    title_abstract_keywords: str | list[str] | None = None


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


FIELD_RULES = {
    "query": "query must be a string",
    "keywords": "keywords must be a string or a list of strings",
    "top_n": "top_n must be a whole number",
    "classification": "classification must be a code or a list of codes",
    "date_from": "date_from must be a date like 2024, 2024-06 or 2024-06-15",
    "date_to": "date_to must be a date like 2024, 2024-06 or 2024-06-15",
    "independent_only": "independent_only must be true or false",
    "group_by": "group_by must be claim or patent",
    "title_abstract_keywords": "title_abstract_keywords must be a string or a list of strings",
}


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    messages = []
    for err in exc.errors():
        loc = [p for p in err.get("loc", []) if p != "body"]
        if err.get("type") == "json_invalid":
            message = "The request body is not valid JSON"
        elif loc and loc[0] in FIELD_RULES:
            message = FIELD_RULES[loc[0]]
        else:
            message = "The request body must be a JSON object (query, keywords, top_n and the filters)"
        if message not in messages:
            messages.append(message)
    return _error(400, ". ".join(messages) + ".")


@app.post("/search")
def post_search(body: SearchRequest):
    # Plain def: FastAPI runs it in a worker thread, so embedding does not block the server.
    try:
        with _index_lock:
            index = search_module.get_index()
        response = index.search(body.query, body.keywords, body.top_n, body.classification,
                                body.date_from, body.date_to, body.independent_only, body.group_by,
                                body.title_abstract_keywords)
    except ValueError as e:
        return _error(400, str(e))
    except IndexNotReady as e:
        return _error(503, str(e))
    return response.to_dict()


@app.get("/classifications")
def get_classifications():
    """Subclass -> main group -> subgroup, with patent counts, for the filter dropdowns."""
    try:
        with _index_lock:
            index = search_module.get_index()
    except IndexNotReady as e:
        return _error(503, str(e))
    return {"subclasses": index.classification_tree()}


class MapRequest(BaseModel):
    """The search fields without top_n and group_by: the map lights every matching patent."""
    query: str | None = None
    keywords: str | list[str] | None = None
    classification: str | list[str] | None = None
    date_from: str | int | None = None
    date_to: str | int | None = None
    independent_only: bool = False
    title_abstract_keywords: str | list[str] | None = None


@app.get("/map")
def get_map():
    """One point per patent (x, y in 0 to 1), with what the page needs to color and label it."""
    from patent_search.landscape import patent_map
    try:
        with _index_lock:
            index = search_module.get_index()
            if getattr(index, "_map_points", None) is None:
                xy = patent_map(index)
                points = []
                for doc, (x, y) in sorted(xy.items()):
                    p, c = index.patents[doc], index.patents[doc]["cpc"]
                    points.append({"doc_number": doc, "x": x, "y": y, "title": p["title"],
                                   "pub_date": p["pub_date"], "cpc_code": p["cpc_code"],
                                   "subclass": c.subclass if c else "",
                                   "group": f"{c.subclass} {c.group}" if c and c.group is not None else ""})
                index._map_points = points
    except IndexNotReady as e:
        return _error(503, str(e))
    return {"points": index._map_points, "classifications": index.classification_tree()}


@app.post("/map/matches")
def post_map_matches(body: MapRequest):
    """Every patent that passes the search (not only the top n), best first."""
    try:
        with _index_lock:
            index = search_module.get_index()
        matches = index.patent_matches(body.query, body.keywords, body.classification, body.date_from,
                                       body.date_to, body.independent_only, body.title_abstract_keywords)
    except ValueError as e:
        return _error(400, str(e))
    except IndexNotReady as e:
        return _error(503, str(e))
    return {"total_matches": len(matches), "matches": matches}


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")
