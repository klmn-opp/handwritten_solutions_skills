"""Syntax checks preserve the written meaning; they do not solve/correct mathematics."""

import re


def latex_structure(raw: str) -> dict:
    clean = re.sub(r"\\[{}]", "", raw)
    depth = 0
    balanced = True
    for char in clean:
        depth += (char == "{") - (char == "}")
        if depth < 0:
            balanced = False
    stack = []
    for action, name in re.findall(r"\\(begin|end)\{([^}]+)\}", raw):
        if action == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            balanced = False
    return {
        "raw_latex": raw,
        "normalized_expression": raw.strip(),
        "normalization": "outer-whitespace-only; no semantic rewrites",
        "balanced": balanced and depth == 0 and not stack,
        "structures": [
            name
            for name, token in {
                "fraction": "\\frac",
                "radical": "\\sqrt",
                "superscript": "^",
                "subscript": "_",
                "summation": "\\sum",
                "integral": "\\int",
                "matrix": "matrix",
                "alignment": "aligned",
            }.items()
            if token in raw
        ],
        "semantic_correctness": "not-assessed",
        "renderability": "preview-required",
    }
