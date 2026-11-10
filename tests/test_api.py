import pytest
from fastapi.testclient import TestClient

from patent_search import api, search as search_module
from patent_search.search import IndexNotReady


@pytest.fixture
def client(index, monkeypatch):
    monkeypatch.setattr(search_module, "get_index", lambda: index)
    return TestClient(api.app)


def test_search_matches_python_function(client, index):
    for body, args in [
        ({"query": "spoke"}, ("spoke", None, 10)),
        ({"keywords": "tire", "top_n": 2}, (None, "tire", 2)),
        ({"query": "sensor", "keywords": ["tire"]}, ("sensor", ["tire"], 10)),
    ]:
        r = client.post("/search", json=body)
        assert r.status_code == 200
        assert r.json() == index.search(*args).to_dict()


def test_keyword_string_uses_the_same_parser(client):
    a = client.post("/search", json={"keywords": 'tire "a sensor"'}).json()
    b = client.post("/search", json={"keywords": ["tire", "a sensor"]}).json()
    assert a == b and a["total_matches"] == 1


@pytest.mark.parametrize("body, text", [
    ({}, "query or at least one keyword"),
    ({"query": "  ", "keywords": ""}, "query or at least one keyword"),
    ({"query": "tire", "top_n": 0}, "top_n"),
    ({"query": "tire", "top_n": 1000}, "top_n"),
    ({"query": "tire", "top_n": "ten"}, "top_n must be a whole number"),
    ([1, 2], "JSON object"),
    ({"query": 5}, "query"),
    ({"keywords": {"a": 1}}, "keywords must be a string or a list of strings"),
])
def test_bad_input_is_400(client, body, text):
    r = client.post("/search", json=body)
    assert r.status_code == 400
    assert text in r.json()["error"]


def test_body_that_is_not_json_is_400(client):
    r = client.post("/search", content="not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"] == "The request body is not valid JSON."


def test_index_not_ready_is_503(monkeypatch):
    def missing():
        raise IndexNotReady("No index found. Rebuild it with: python -m patent_search build")
    monkeypatch.setattr(search_module, "get_index", missing)
    r = TestClient(api.app).post("/search", json={"query": "tire"})
    assert r.status_code == 503 and "patent_search build" in r.json()["error"]


def test_page(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    page = r.text
    for text in ['placeholder=\'sipe "tread block" spoke*\'', "Exact keyword filters", "Type a question or a keyword",
                 "Searching...", "No claims matched.", "The start of this claim is missing in the source data.",
                 "Show full claim", "prefers-color-scheme", 'fetch("/search"']:
        assert text in page, text


def test_page_loads_nothing_from_the_internet(client):
    import re
    page = client.get("/").text
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page)
    assert "@import" not in page and "url(http" not in page
