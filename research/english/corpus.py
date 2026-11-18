"""Fixed backend-independent English-line corpus; holdout cannot tune policies."""

import hashlib
import json
from pathlib import Path

CORPUS = Path(__file__).parent / "benchmarks/v1/lines.json"


def load(split="development", *, purpose="evaluation") -> tuple[list[dict], str]:
    digest = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    if digest != (CORPUS.parent / "SHA256SUMS").read_text().split()[0]:
        raise ValueError("Benchmark digest changed; create a new version")
    if purpose not in {"evaluation", "diagnostic"} or (
        split in {"all", "holdout"} and purpose != "evaluation"
    ):
        raise ValueError("Holdout is evaluation only; no training or tuning")
    if split not in {"all", "holdout", "development", "smoke"}:
        raise ValueError("Unknown split")
    data = json.loads(CORPUS.read_text())["cases"]
    if len({row["id"] for row in data}) != len(data) or len(data) < 100:
        raise ValueError("Case IDs must be unique; at least 100 cases required")
    return [
        r
        for r in data
        if split == "all" or r["split"] == split or (split == "smoke" and r["smoke"])
    ], digest
