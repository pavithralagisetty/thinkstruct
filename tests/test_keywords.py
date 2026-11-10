import pytest

from patent_search.keywords import compile_keywords, matches_all, normalize_keywords, parse_keywords


def match(keywords, text):
    return matches_all(text, compile_keywords(keywords))


def test_parse_keywords():
    assert parse_keywords('sipe "tread block" spoke*') == ["sipe", "tread block", "spoke*"]
    assert parse_keywords('  "tread   block"  ') == ["tread block"]
    assert parse_keywords('a "unclosed phrase') == ["a", "unclosed phrase"]
    assert parse_keywords('"" ') == []
    assert parse_keywords(None) == parse_keywords("") == []


def test_normalize_keywords():
    assert normalize_keywords('sipe "tread block"') == ["sipe", "tread block"]
    assert normalize_keywords(["tread block", '"sipe"', " ", None]) == ["tread block", "sipe"]
    assert normalize_keywords(None) == []


@pytest.mark.parametrize("keywords, text, expected", [
    ("spoke*", "a plurality of spokes", True),
    ("spoke*", "spoked wheel", True),
    ("spoke*", "a spoke", True),
    ("spoke*", "a wheelspoke", False),
    ("spoke", "spokes", False),                         # whole word, no stemming
    ("SIPE", "a Sipe in the block", True),              # case insensitive
    ('"fiber reinforced"', "a fiber-reinforced resin", True),
    ("fiber-reinforced", "a fiber reinforced resin", True),
    ('"fiber reinforced"', "fiberreinforced", False),
    ("fibre", "carbon fiber", False),                   # no spelling variants
    ('"tread block"', "a tread block", True),
    ('"tread block"', "a block of the tread", False),   # phrase, not two words
    ("tread block", "a block of the tread", True),      # unquoted: two keywords
    ("tread sipe", "a tread", False),                   # AND
    ("3.5", "about 3.5 mm", True),
    ("3.5", "about 305 mm", False),                     # regex characters are literal
])
def test_matching(keywords, text, expected):
    assert match(keywords, text) is expected


def test_wildcard_only_is_ignored():
    assert compile_keywords("*") == []


def spans(keywords, text):
    from patent_search.keywords import highlight_spans
    return [text[a:b] for a, b in highlight_spans(text, compile_keywords(keywords))]


def test_highlight_spans():
    from patent_search.keywords import highlight_spans
    assert spans("spoke*", "a spoke and two spokes, not wheelspoke") == ["spoke", "spokes"]
    assert spans('"fiber reinforced"', "a fiber-reinforced or fiber reinforced rim") == ["fiber-reinforced", "fiber reinforced"]
    assert spans("Sipe", "SIPE and sipe") == ["SIPE", "sipe"]
    # overlapping and touching matches merge, output is sorted
    assert highlight_spans("tread block", compile_keywords('block "tread block" tread')) == [[0, 11]]
    assert highlight_spans("b a", compile_keywords("a b")) == [[0, 1], [2, 3]]
    assert highlight_spans("anything", []) == []
