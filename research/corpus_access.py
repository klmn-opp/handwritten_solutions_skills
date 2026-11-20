"""Read-only research catalogue access with metadata-first heldout isolation.

Labels are fetched only after forbidden sources and their ancestry-connected
source groups have been excluded. No corpus split or record is rewritten.
"""

import json
import re
import sqlite3
from pathlib import Path

from hwsol.profiles.io import sha256
from research.corpus_registry import SEALED

FORBIDDEN_NAME = re.compile(r"session[ _-]*c(?:[ _.-]|$)|heldout", re.I)
HELD_SPLITS = frozenset({"holdout", "heldout", "test", "sealed"})
SPLIT_PATHS = (
    "$.dataset_split",
    "$.split",
    "$.derivation.selection.split",
    "$.derivation.split",
)


def catalogue(path: Path) -> dict:
    """Resolve split/source/ancestry metadata before retrieving any raw label JSON."""
    path = Path(path)
    split_sql = ",".join(f"json_extract(data,'{field}')" for field in SPLIT_PATHS)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        docs = db.execute(
            "SELECT id,json_extract(data,'$.source_hash'),"
            "json_extract(data,'$.source_path'),json_extract(data,'$.original_name'),"
            f"{split_sql} FROM records WHERE kind='document'"
        ).fetchall()
        pages = db.execute(
            "SELECT id,json_extract(data,'$.source_hash'),"
            f"{split_sql} FROM records WHERE kind='page'"
        ).fetchall()
        samples = db.execute(
            "SELECT id,json_extract(data,'$.source_hash'),json_extract(data,'$.parents'),"
            f"{split_sql} FROM records WHERE kind='sample'"
        ).fetchall()
        named_sources = {
            digest
            for _, digest, name, original_name, *_ in docs
            if FORBIDDEN_NAME.search(name or "") or FORBIDDEN_NAME.search(original_name or "")
        }
        held_sources = (
            {digest for _, digest, _, _, *splits in docs if any(s in HELD_SPLITS for s in splits)}
            | {digest for _, digest, *splits in pages if any(s in HELD_SPLITS for s in splits)}
            | {digest for _, digest, _, *splits in samples if any(s in HELD_SPLITS for s in splits)}
        )
        blocked_sources = (set(SEALED) | named_sources | held_sources) - {None, ""}
        source_by_id = {row[0]: row[1] for row in [*docs, *pages, *samples]}
        parents = {}
        for sid, _, raw_parents, *_ in samples:
            value = json.loads(raw_parents or "[]")
            if not isinstance(value, list) or any(not isinstance(p, str) for p in value):
                raise ValueError("Invalid ancestry metadata; labels were not read")
            parents[sid] = set(value)
        blocked_ids = {sid for sid, digest in source_by_id.items() if not digest}
        # Whole-source and both ancestry directions are deliberate: a heldout
        # derivative must not expose its training-labeled original or siblings.
        while True:
            old_ids, old_sources = set(blocked_ids), set(blocked_sources)
            blocked_ids |= {
                sid for sid, digest in source_by_id.items() if digest in blocked_sources
            }
            for sid, ancestors in parents.items():
                if sid in blocked_ids or ancestors & blocked_ids:
                    blocked_ids.add(sid)
                    blocked_ids |= ancestors
            blocked_sources |= {source_by_id[sid] for sid in blocked_ids if source_by_id.get(sid)}
            if old_ids == blocked_ids and old_sources == blocked_sources:
                break
        allowed_hashes = [d[1] for d in docs if d[1] and d[1] not in blocked_sources]
        allowed_samples = [s[0] for s in samples if s[0] not in blocked_ids]

        def rows(kind: str, field: str, values: list) -> list[dict]:
            if not values:
                return []
            parameters = ",".join("?" for _ in values)
            return [
                json.loads(raw)
                for (raw,) in db.execute(
                    f"SELECT data FROM records WHERE kind=? AND {field} IN ({parameters})",
                    [kind, *values],
                )
            ]

        return {
            "path": str(path),
            "sha256": sha256(path),
            "documents": rows("document", "json_extract(data,'$.source_hash')", allowed_hashes),
            "pages": rows("page", "json_extract(data,'$.source_hash')", allowed_hashes),
            "samples": rows("sample", "id", allowed_samples),
            "recognitions": rows(
                "recognition", "json_extract(data,'$.sample_id')", allowed_samples
            ),
            "filter": {
                "excluded_source_hashes": sorted(blocked_sources),
                "nested_or_top_level_heldout_source_hashes": sorted(held_sources - {None, ""}),
                "name_blocked_source_hashes": sorted(named_sources - {None, ""}),
                "blocked_sample_count": sum(s[0] in blocked_ids for s in samples),
                "split_metadata_fields": list(SPLIT_PATHS),
                "method": (
                    "source SHA/name and top-level/nested split metadata; whole-source "
                    "plus ancestors/descendants/siblings excluded before label JSON retrieval"
                ),
            },
        }
