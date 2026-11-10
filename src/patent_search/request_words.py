"""Remove request wording around a question before it is embedded.

"claims which talk about spoke wheels" and "find me patents about spoke wheels, please"
are searched as "spoke wheels". The word lists are in config. Only the start and the end
of the question are touched, never words in the middle, and if nothing would be left the
question is kept as typed. Request words are removed in this order at the start:

    polite*  [verb [me]]  [ask]  [determiner]  [noun]  [relative] [describing verb] [link]

A determiner, relative or link word is only removed after a request verb or noun, so a
plain question such as "a wheel with spokes" or "the backing film ..." is left alone.
"""

import re

from patent_search import config


def _words(words: list[str]) -> re.Pattern:
    """Any of the words or phrases at the start, as whole words, plus what separates them."""
    alternatives = sorted(words, key=len, reverse=True)
    body = "|".join(re.escape(w).replace(r"\ ", r"\s+").replace("'", "['’]") for w in alternatives)
    return re.compile(rf"(?:{body})(?:[\s,:;]+|$)", re.IGNORECASE)


_POLITE = _words(config.REQUEST_POLITE)
_VERB = _words(config.REQUEST_VERBS)
_VERB_ALONE = _words(config.REQUEST_VERBS_ALONE)
_ME = _words(config.REQUEST_ME)
_ASK = _words(config.REQUEST_ASK)
_DETERMINER = _words(config.REQUEST_DETERMINERS)
_NOUN = _words(config.REQUEST_NOUNS)
_RELATIVE = _words(config.REQUEST_RELATIVE)
_RELATIVE_VERB = _words(config.REQUEST_RELATIVE_VERBS)
_LINK = _words(config.REQUEST_LINKS)
_TRAILING = re.compile(
    r"(?:[\s,]+(?:" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in config.REQUEST_TRAILING)
    + r"))*[\s?!.]*$", re.IGNORECASE)


def _take(pattern: re.Pattern, text: str) -> str | None:
    """The text after the pattern at its start, or None when it does not start with it."""
    m = pattern.match(text)
    return text[m.end():] if m else None


def _skip(pattern: re.Pattern, text: str) -> str:
    """The text after the pattern at its start, or the text unchanged."""
    rest = _take(pattern, text)
    return text if rest is None else rest


def strip_request_words(question: str) -> str:
    """The question without request words at its start and end (see the module docstring)."""
    original = " ".join(question.split())
    text = _TRAILING.sub("", " " + original).strip()

    while (rest := _take(_POLITE, text)) is not None:
        text = rest

    asked = False                     # a request verb or noun was removed
    rest = _take(_VERB, text)
    if rest is not None:
        after_me = _take(_ME, rest)
        follows = after_me if after_me is not None else rest
        if (after_me is not None or _VERB_ALONE.match(text)
                or any(p.match(follows) for p in (_DETERMINER, _NOUN, _LINK))):
            text, asked = follows, True

    rest = _take(_ASK, text)
    if rest is not None:
        if _NOUN.match(_skip(_DETERMINER, rest)):
            text = rest

    rest = _take(_DETERMINER, text)
    if rest is not None and (asked or _NOUN.match(rest)):
        text = rest
    rest = _take(_NOUN, text)
    if rest is not None:
        text, asked = rest, True

    if asked:
        text = _skip(_RELATIVE, text)
        text = _skip(_RELATIVE_VERB, text)
        text = _skip(_LINK, text)

    text = text.strip(" ,:;")
    return text if text else original
