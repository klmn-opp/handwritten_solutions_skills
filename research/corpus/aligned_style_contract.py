"""Pure-Python admission rules; production does not import a neural dependency."""


def text_tokens(texts: list[str]) -> list[list[int]]:
    result = []
    for text in texts:
        if not text or len(text) > 32 or any(ord(c) > 127 for c in text):
            raise ValueError("Only complete nonempty ASCII words up to 32 characters admitted")
        result.append([ord(c) + 1 for c in text] + [0] * (32 - len(text)))
    return result


def training_indices(details: list[dict], rows: list[dict]) -> list[int]:
    if len(details) != len(rows) or {q["index"] for q in details} != set(range(len(rows))):
        raise ValueError("Registration must cover every source exactly once")
    for q in details:
        if q["split"] != rows[q["index"]]["dataset_split"] or q["text"] != rows[q["index"]]["text"]:
            raise ValueError("Registration identity differs from source")
    ids = [q["index"] for q in details if q["eligible"] and q["split"] == "train"]
    if len(ids) < 64 or len({rows[i]["text"] for i in ids}) < 20:
        raise ValueError("Predeclared real-registration coverage gate failed")
    return ids
