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
    ({}, "at least one keyword, or a filter"),
    ({"query": "  ", "keywords": ""}, "at least one keyword, or a filter"),
    ({"classification": "B60C 2"}, "no main group 2"),
    ({"classification": ["B60C 11", 5]}, "classification must be a code or a list of codes"),
    ({"date_from": "2024/06"}, "date like 2024, 2024-06 or 2024-06-15"),
    ({"date_to": "2024-02-30"}, "not a real date"),
    ({"date_from": "2025", "date_to": "2024"}, "is after date_to"),
    ({"independent_only": "maybe"}, "independent_only must be true or false"),
    ({"query": "tire", "group_by": "patents"}, "group_by must be one of"),
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
    for text in ['class="tagbox"', 'Press Enter after each keyword', "Exact filters", "Type a question or a keyword",
                 'data-group="claim"', 'data-group="patent"', 'fetch("/classifications")', 'data-group="map"',
                 "Published from", "Published to", "No patents matched.", "Best matching claims", "Start missing", " match`",
                 "Searching...", "No claims matched.", "The start of this claim is missing in the source data.",
                 "Show full claim", "prefers-color-scheme", 'fetch("/search"']:
        assert text in page, text


def test_page_loads_nothing_from_the_internet(client):
    import re
    page = client.get("/").text
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page)
    assert "@import" not in page and "url(http" not in page


def test_filters_and_grouping_through_the_api(client, index):
    body = {"classification": "B60C", "date_from": 2024, "independent_only": False, "group_by": "patent"}
    r = client.post("/search", json=body)
    assert r.status_code == 200
    assert r.json() == index.search(None, None, 10, "B60C", 2024, None, False, "patent").to_dict()
    assert r.json()["group_by"] == "patent"


def test_classifications(client):
    r = client.get("/classifications")
    assert r.status_code == 200
    tree = r.json()["subclasses"]
    assert [n["code"] for n in tree] == ["B60B", "B60C"]          # C has no classification
    b60c = tree[1]
    assert b60c["count"] == 1 and b60c["title"].startswith("VEHICLE TYRES")
    [group] = b60c["groups"]
    assert group == {"code": "B60C 11", "count": 1, "title": "Tyre tread bands; Tread patterns; Anti-skid inserts",
                     "subgroups": [{"code": "B60C 11/12", "count": 1}]}
    assert tree[0]["groups"][0]["subgroups"] == []                # B60B 1/00 is the group itself


def test_classification_counts_use_the_filter_matching(client, index):
    for node in client.get("/classifications").json()["subclasses"]:
        for code in [node["code"]] + [g["code"] for g in node["groups"]]:
            assert index.search(classification=code, group_by="patent").total_matches == \
                next(n for n in [node] + node["groups"] if n["code"] == code)["count"]



def test_title_abstract_keywords(client, index):
    r = client.post("/search", json={"title_abstract_keywords": "tire", "group_by": "patent"})
    assert r.status_code == 200
    assert r.json() == index.search(None, None, 10, None, None, None, False, "patent", "tire").to_dict()
    r = client.post("/search", json={"title_abstract_keywords": 5})
    assert r.status_code == 400 and "title_abstract_keywords must be" in r.json()["error"]
