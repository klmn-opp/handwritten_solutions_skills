"""Private corpus: immutable assets, transactional records and append-only review history."""

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from hwsol.profiles.io import sha256

SCHEMA = "personal-homework-corpus/1"
DOMAINS = {"personal-real", "public", "base-font", "synthetic", "model-generated", "unknown"}
TYPES = {
    "page",
    "english-line",
    "english-word",
    "math-expression",
    "math-symbol",
    "glyph",
    "unknown",
}
ROLES = {"personal-handwriting", "printed", "other-writer", "mixed", "unknown"}
STATUSES = {"auto-candidate", "needs-review", "verified", "rejected"}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Conflict(ValueError):
    pass


class Corpus:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL,
                    action TEXT NOT NULL, entity TEXT NOT NULL, data TEXT NOT NULL);
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / "corpus.sqlite", timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _get(db, kind: str, key: str) -> dict:
        row = db.execute("SELECT data FROM records WHERE kind=? AND id=?", (kind, key)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    @staticmethod
    def _put(db, kind: str, key: str, data: dict) -> None:
        db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (kind, key, json.dumps(data)))

    @staticmethod
    def event(db, action: str, entity: str, data: dict) -> None:
        db.execute(
            "INSERT INTO events(time,action,entity,data) VALUES(?,?,?,?)",
            (now(), action, entity, json.dumps(data)),
        )

    def get(self, kind: str, key: str) -> dict:
        with self.db() as db:
            return self._get(db, kind, key)

    def rows(self, kind: str) -> list[dict]:
        with self.db() as db:
            return [
                json.loads(r[0])
                for r in db.execute("SELECT data FROM records WHERE kind=? ORDER BY id", (kind,))
            ]

    def history(self, entity: str) -> list[dict]:
        with self.db() as db:
            return [
                {"seq": r[0], "time": r[1], "action": r[2], "data": json.loads(r[3])}
                for r in db.execute(
                    "SELECT seq,time,action,data FROM events WHERE entity=? ORDER BY seq", (entity,)
                )
            ]

    def asset(self, content: bytes, suffix: str) -> str:
        if suffix not in {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".json"}:
            raise ValueError("Unsupported asset extension")
        sha = hashlib.sha256(content).hexdigest()
        rel = Path("assets") / sha[:2] / (sha + suffix)
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=dest.parent, prefix=".incoming-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o444)
            try:
                os.link(temporary, dest)
            except FileExistsError:
                if sha256(dest) != sha:
                    raise ValueError("Immutable asset corruption") from None
        finally:
            os.unlink(temporary)
        return rel.as_posix()

    def path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root / "assets") or not path.is_file():
            raise ValueError("Not a corpus asset")
        if sha256(path) != path.stem:
            raise ValueError("Asset digest mismatch")
        return path

    def propose(
        self,
        page_id: str,
        bbox: list,
        *,
        content_type="unknown",
        label="",
        label_source="segmentation",
        role="unknown",
        parents=None,
        notes="",
    ) -> dict:
        with self.db() as db:
            return self._propose(
                db,
                page_id,
                bbox,
                content_type=content_type,
                label=label,
                label_source=label_source,
                role=role,
                parents=parents,
                notes=notes,
            )

    def _propose(
        self,
        db,
        page_id,
        bbox,
        *,
        content_type="unknown",
        label="",
        label_source="segmentation",
        role="unknown",
        parents=None,
        notes="",
    ):
        page = self._get(db, "page", page_id)
        bbox = validate_bbox(bbox, page["width"], page["height"])
        if content_type not in TYPES or role not in ROLES:
            raise ValueError("Invalid sample classification")
        key = "s-" + digest([page_id, bbox, content_type, label_source, parents or []])[:24]
        try:
            return self._get(db, "sample", key)
        except KeyError:
            pass
        import io

        with Image.open(self.path(page["color_image"])) as image:
            out = io.BytesIO()
            image.crop(bbox).save(out, format="PNG")
        crop = self.asset(out.getvalue(), ".png")
        from research.corpus.quality import inspect_image

        sample = {
            "schema": SCHEMA,
            "sample_id": key,
            "document_id": page["document_id"],
            "page_id": page_id,
            "source_hash": page["source_hash"],
            "image_crop": crop,
            "crop_sha256": Path(crop).stem,
            "bounding_box": bbox,
            "content_type": content_type,
            "candidate_text": label,
            "verified_text": None,
            "verified_latex": None,
            "label_source": label_source,
            "recognizer_name": None,
            "recognizer_version": None,
            "confidence": None,
            "review_status": "auto-candidate",
            "writer_provenance": page["writer_provenance"],
            "region_role": role,
            "preprocessing_transforms": [{"operation": "crop", "box": bbox}],
            "dataset_split": None,
            "notes": notes,
            "parents": parents or [],
            "revision": 0,
            "created_at": now(),
            "privacy_checked": False,
            "review_attestation": None,
            "quality_status": "unassessed",
            "defects": [],
            "quality_triage": inspect_image(self.path(crop)),
        }
        self._put(db, "sample", key, sample)
        self.event(db, "propose", key, sample)
        return sample

    def add_recognition(self, sample_id: str, result: dict) -> dict:
        required = {"engine", "version", "text", "raw", "input_sha256"}
        if not required <= result.keys():
            raise ValueError("Incomplete recognition provenance")
        with self.db() as db:
            sample = self._get(db, "sample", sample_id)
            if result["input_sha256"] != sample["crop_sha256"]:
                raise ValueError("Recognition image does not match sample")
            key = "r-" + digest([sample_id, result])[:24]
            result = {
                **result,
                "id": key,
                "sample_id": sample_id,
                "created_at": now(),
                "is_ground_truth": False,
            }
            self._put(db, "recognition", key, result)
            self.event(db, "recognize", sample_id, result)
            # Recognition never edits an existing manual decision or supplies verified labels.
            return result

    def reshape(self, items: list[dict], boxes: list[list] | None, actor: str) -> list[dict]:
        """Merge or split atomically; new regions lose all old verification claims."""
        if not actor.strip() or not items or len({i["id"] for i in items}) != len(items):
            raise ValueError("Unique samples and reviewer required")
        with self.db() as db:
            old = [self._get(db, "sample", i["id"]) for i in items]
            if any(s["revision"] != i["revision"] for s, i in zip(old, items, strict=True)):
                raise Conflict("Region revisions changed")
            if len({s["page_id"] for s in old}) != 1 or any(
                s["review_status"] == "rejected" for s in old
            ):
                raise ValueError("Only active regions on the same page can be reshaped")
            bounds = [min(s["bounding_box"][j] for s in old) for j in (0, 1)] + [
                max(s["bounding_box"][j] for s in old) for j in (2, 3)
            ]
            if boxes is None:
                if len(old) < 2:
                    raise ValueError("Merge requires at least two regions")
                boxes = [bounds]
            elif len(old) != 1 or len(boxes) < 2 or len({tuple(b) for b in boxes}) != len(boxes):
                raise ValueError("Split requires one parent and at least two distinct boxes")
            for box in boxes:
                if len(box) != 4 or not (
                    bounds[0] <= box[0] < box[2] <= bounds[2]
                    and bounds[1] <= box[1] < box[3] <= bounds[3]
                ):
                    raise ValueError("Split boxes must lie inside the parent")
            parents = [s["sample_id"] for s in old]
            children = [
                self._propose(
                    db,
                    old[0]["page_id"],
                    box,
                    parents=parents,
                    label_source="manual-region-edit",
                    notes="Merged/split geometry requires new transcription review",
                )
                for box in boxes
            ]
            for sample in old:
                before = dict(sample)
                sample.update(
                    review_status="rejected",
                    verified_text=None,
                    verified_latex=None,
                    review_attestation=None,
                    revision=sample["revision"] + 1,
                )
                self._put(db, "sample", sample["sample_id"], sample)
                self.event(
                    db,
                    "reshape",
                    sample["sample_id"],
                    {
                        "actor": actor,
                        "before": before,
                        "after": sample,
                        "children": [c["sample_id"] for c in children],
                    },
                )
            return children

    def review(
        self,
        sample_id: str,
        revision: int,
        changes: dict,
        actor: str,
        attestation: dict | None = None,
    ) -> dict:
        allowed = {
            "candidate_text",
            "content_type",
            "region_role",
            "review_status",
            "notes",
            "privacy_checked",
            "quality_status",
            "defects",
        }
        if set(changes) - allowed or not actor.strip():
            raise ValueError("Invalid review fields or missing reviewer")
        with self.db() as db:
            old = self._get(db, "sample", sample_id)
            if old["revision"] != revision:
                raise Conflict("A newer revision exists; reload before saving")
            new = {**old, **changes, "revision": revision + 1}
            from research.corpus.quality import DEFECTS

            if (
                new.get("quality_status", "unassessed")
                not in {"unassessed", "clean", "readable-correction", "unusable"}
                or set(new.get("defects", [])) - DEFECTS
            ):
                raise ValueError("Invalid quality classification")
            if (
                new["review_status"] not in STATUSES
                or new["content_type"] not in TYPES
                or new["region_role"] not in ROLES
            ):
                raise ValueError("Invalid review classification")
            if not isinstance(new["candidate_text"], str) or len(new["candidate_text"]) > 20000:
                raise ValueError("Invalid transcription")
            new["verified_text"] = new["verified_latex"] = None
            new["review_attestation"] = None
            if new["review_status"] == "verified":
                checks = {"actual_transcription", "crop_complete", "writer_checked", "human_review"}
                if not attestation or not all(attestation.get(k) is True for k in checks):
                    raise ValueError(
                        "Verification requires explicit human transcription/crop/writer attestation"
                    )
                if (
                    not new["candidate_text"].strip()
                    or new["content_type"] == "unknown"
                    or new["region_role"] in {"unknown", "mixed"}
                ):
                    raise ValueError("Verify a nonempty transcription and classified region")
                if new["content_type"] in {"math-expression", "math-symbol"}:
                    from research.corpus.validation import latex_structure

                    new["latex_validation"] = latex_structure(new["candidate_text"])
                    if not new["latex_validation"]["balanced"]:
                        raise ValueError("Unbalanced LaTeX; keep it in review")
                    new["verified_latex"] = new["candidate_text"]
                else:
                    new["verified_text"] = new["candidate_text"]
                new["label_source"] = "human-review"
                new["review_attestation"] = {**attestation, "actor": actor, "time": now()}
            self._put(db, "sample", sample_id, new)
            self.event(db, "review", sample_id, {"actor": actor, "before": old, "after": new})
            return new


def validate_bbox(box: list, width: int, height: int) -> list[int]:
    if len(box) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box):
        raise ValueError("Four finite pixel coordinates required")
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Bounding box outside original page")
    return [x0, y0, x1, y1]
