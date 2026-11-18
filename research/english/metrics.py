"""Alignment of recognizer observations; these are not ground-truth generation errors."""

from collections import Counter


def align(expected: list, observed: list) -> list[dict]:
    """Deterministic Levenshtein alignment: match, substitution, deletion, insertion tie order."""
    n, m = len(expected), len(observed)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(
                costs[i - 1][j - 1] + (expected[i - 1] != observed[j - 1]),
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
            )
    result = []
    i, j = n, m
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (expected[i - 1] != observed[j - 1]):
            op = "equal" if expected[i - 1] == observed[j - 1] else "substitution"
            result.append(
                {
                    "op": op,
                    "expected": expected[i - 1],
                    "observed": observed[j - 1],
                    "expected_index": i - 1,
                    "observed_index": j - 1,
                }
            )
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            result.append(
                {
                    "op": "deletion",
                    "expected": expected[i - 1],
                    "observed": None,
                    "expected_index": i - 1,
                    "observed_index": j,
                }
            )
            i -= 1
        else:
            result.append(
                {
                    "op": "insertion",
                    "expected": None,
                    "observed": observed[j - 1],
                    "expected_index": i,
                    "observed_index": j - 1,
                }
            )
            j -= 1
    return list(reversed(result))


def score(expected: str, observed: str) -> dict:
    expected, observed = " ".join(expected.split()), " ".join(observed.split())
    if not expected:
        raise ValueError("Expected text must be nonempty")
    chars = align(list(expected), list(observed))
    words = align(expected.split(), observed.split())
    counts = Counter(a["op"] for a in chars)
    wcounts = Counter(a["op"] for a in words)
    repeated = [
        a
        for a in words
        if a["op"] == "insertion"
        and a["observed"]
        in {
            expected.split()[i]
            for i in (a["expected_index"] - 1, a["expected_index"])
            if 0 <= i < len(expected.split())
        }
    ]
    tail = 0
    for a in reversed(chars):
        if a["op"] != "deletion":
            break
        tail += 1
    return {
        "cer": sum(counts[k] for k in ("substitution", "deletion", "insertion")) / len(expected),
        "wer": sum(wcounts[k] for k in ("substitution", "deletion", "insertion"))
        / len(expected.split()),
        "exact_match": expected == observed,
        "expected_characters": len(expected),
        "expected_words": len(expected.split()),
        "character_errors": {k: counts[k] for k in ("substitution", "deletion", "insertion")},
        "word_errors": {k: wcounts[k] for k in ("substitution", "deletion", "insertion")},
        "repeated_inserted_words": len(repeated),
        "tail_deletions": tail,
        "word_alignment": words,
        "character_alignment": chars,
        "interpretation": "recognizer discrepancy; true error requires independent adjudication",
    }


def aggregate(rows: list[dict]) -> dict:
    """Micro error rates and exact match; missing observations stay outside this denominator."""
    chars = sum(r["expected_characters"] for r in rows)
    words = sum(r["expected_words"] for r in rows)
    return {
        "observed_count": len(rows),
        "micro_cer": sum(r["cer"] * r["expected_characters"] for r in rows) / chars
        if chars
        else None,
        "micro_wer": sum(r["wer"] * r["expected_words"] for r in rows) / words if words else None,
        "exact_match_rate": sum(r["exact_match"] for r in rows) / len(rows) if rows else None,
        "character_errors": {
            k: sum(r["character_errors"][k] for r in rows)
            for k in ("substitution", "deletion", "insertion")
        },
        "repeated_inserted_words": sum(r["repeated_inserted_words"] for r in rows),
        "tail_deletion_cases": sum(r["tail_deletions"] > 0 for r in rows),
        "human_verified_error_rate": None,
    }
