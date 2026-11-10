import json
from pathlib import Path

import pytest

from patent_search.request_words import strip_request_words

QUESTIONS = [q["text"] for q in json.loads((Path(__file__).parent / "eval" / "questions.json").read_text())]
EXAMPLES = ["wheel spoke made of carbon fiber", "bassinet arm that rotates the infant tub",
            "setting up a tire pressure sensor from a phone", "robot that drives up walls using magnetic wheels",
            "tire for heavy trucks"]


@pytest.mark.parametrize("typed, searched", [
    ("claim which talks about spoke wheels", "spoke wheels"),
    ("find me patents about spoke wheels", "spoke wheels"),
    ("show claims that mention spoked wheels please", "spoked wheels"),
    ("Claims about spoke wheels?", "spoke wheels"),
    ("I'm looking for patents related to tire pressure sensors", "tire pressure sensors"),
    ("I’m looking for rims", "rims"),                               # curly apostrophe
    ("what are the claims about sipes", "sipes"),
    ("can you find claims with a magnetic wheel", "a magnetic wheel"),
    ("list all patents on bassinets", "bassinets"),
    ("which claims describe a hub motor", "a hub motor"),
    ("claims that relate to spoke wheels", "spoke wheels"),
    ("search for claims that are about carbon fiber spokes, thanks", "carbon fiber spokes"),
    ("find   spoke\twheels", "spoke wheels"),
])
def test_request_words_removed(typed, searched):
    assert strip_request_words(typed) == searched


@pytest.mark.parametrize("typed", [
    "return spring for a valve",        # "return" is a verb only before me, a determiner, noun or link
    "a wheel with spokes",              # a determiner or link alone is not request wording
    "the spoke of a wheel",
    "what is a sipe",                   # "what" only before a noun such as claims
    "wheel claims a priority",          # only the start and the end are touched
    "spoke wheels",
])
def test_plain_questions_unchanged(typed):
    assert strip_request_words(typed) == typed


@pytest.mark.parametrize("typed", ["claims", "please", "find me patents about"])
def test_nothing_left_keeps_the_question(typed):
    assert strip_request_words(typed) == typed


def test_eval_and_example_questions_unchanged():
    for q in QUESTIONS + EXAMPLES:
        assert strip_request_words(q) == q


WRAPPERS = ["find claims about {}", "claims which talk about {}", "show me patents related to {}",
            "I'm looking for claims that describe {}", "{} please", "can you find patents on {}?"]


def test_wrapped_eval_questions_become_the_plain_question():
    for q in QUESTIONS + EXAMPLES:
        for w in WRAPPERS:
            assert strip_request_words(w.format(q)) == q, w.format(q)
