"""CPU kernel extracted from research.author_feedback_round3_delivery; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import re

from handnote.markdown import parse

from hwsol.jobs.layout import plain_emphasis


def f_words(source, title, records):
    """Bind complete original words to visible text, independent of f treatment."""
    spans = [title]
    for block in parse(plain_emphasis(source)):
        spans.extend((text for kind, text in block.segments if kind == "t"))
    rows = [r for r in records if r["context"] == "text"]
    if [ch for span in spans for ch in span if not ch.isspace()] != [r["semantic"] for r in rows]:
        raise ValueError("Full-word comparison text binding failed")
    result, offset = ([], 0)
    for span_index, span in enumerate(spans):
        mapping = {}
        for j, ch in enumerate(span):
            if not ch.isspace():
                mapping[j] = rows[offset]["item_index"]
                offset += 1
        for match in re.finditer("[A-Za-z]+", span):
            if "f" in match.group():
                result.append(
                    {
                        "word": match.group(),
                        "span": span_index,
                        "range": list(match.span()),
                        "items": [mapping[j] for j in range(*match.span())],
                    }
                )
    return result
