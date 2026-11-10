"""HTTP API: POST /search and the web page at GET /."""

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
                                body.date_from, body.date_to, body.independent_only, body.group_by)
    except ValueError as e:
        return _error(400, str(e))
    except IndexNotReady as e:
        return _error(503, str(e))
    return response.to_dict()


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")
