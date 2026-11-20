"""Immutable, private federation of existing reviewed handwriting manifests.

This registers evidence and exposure. It does not approve labels, change source
splits, train a model, or advance the production pointer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run

SCHEMA = "federated-handwriting-registry/2"
KINDS = {"documents", "lines", "words", "glyphs", "math"}
SEALED = {"ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"}
REVIEW_FIELDS = (
    "review",
    "legacy_review",
    "review_status",
    "human_review",
    "label_source",
    "review_sha256",
    "raw_review_id",
    "character_positions_verified",
    "boundary_guard_passed",
)


def review_state(evidence: dict) -> str:
    """Expose source claims; never infer label approval from the intended text."""
    statuses = [evidence.get("review_status")]
    statuses.extend(
        evidence[k].get("status")
        for k in ("review", "legacy_review")
        if isinstance(evidence.get(k), dict)
    )
    statuses = [s for s in statuses if s]
    if any(s not in {"accept", "accepted", "agent-accepted", "human-accepted"} for s in statuses):
        return "held-or-unresolved"
    return "source-accepted" if statuses else "unknown"


def identity(prefix: str, value: object) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return prefix + ":" + hashlib.sha256(blob).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_hash(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("A complete lowercase SHA-256 is required")
    if value in SEALED:
        raise ValueError("Sealed source cannot enter a corpus registry")
    return value


def box(value: list) -> list[float]:
    if len(value) != 4 or not all(math.isfinite(float(x)) for x in value):
        raise ValueError("Finite original source rectangle required")
    x, y, right, bottom = map(float, value)
    if x < 0 or y < 0 or right <= x or bottom <= y:
        raise ValueError("Positive original source rectangle required")
    return [x, y, right, bottom]


def add_box(parent: list, child: list) -> list[float]:
    p, c = box(parent), box(child)
    if c[2] > p[2] - p[0] or c[3] > p[3] - p[1]:
        raise ValueError("Child rectangle exceeds original parent")
    return [p[0] + c[0], p[1] + c[1], p[0] + c[2], p[1] + c[3]]


class Registry:
    def __init__(self, parent: dict | None = None):
        self.data = (
            copy.deepcopy(parent)
            if parent is not None
            else {
                "schema": SCHEMA,
                **{
                    k: {}
                    for k in (
                        "documents",
                        "observations",
                        "views",
                        "memberships",
                        "manifests",
                        "assets",
                    )
                },
            }
        )
        if self.data.get("schema") != SCHEMA:
            raise ValueError("Unsupported parent registry schema")
        self.word_crops: dict[str, set[str]] = {}
        for view in self.data["views"].values():
            if view["kind"] == "original-word":
                self.word_crops.setdefault(view["asset_sha256"], set()).add(view["observation"])
        self.checked: dict[str, str] = {}

    def asset(self, path: str | Path, expected: str) -> str:
        expected = check_hash(expected)
        path = Path(path).expanduser().resolve()
        if re.search(r"session[ _-]*c(?:[ _.-]|$)", str(path), re.IGNORECASE):
            raise ValueError("Sealed Session C asset cannot be read")
        if not path.is_file():
            raise ValueError(f"Missing source asset: {path}")
        if self.checked.get(str(path)) != expected:
            if sha256(path) != expected:
                raise ValueError(f"Source asset hash mismatch: {path}")
            self.checked[str(path)] = expected
        prior = self.data["assets"].get(str(path))
        if prior is not None and prior != expected:
            raise ValueError(f"Historical asset changed: {path}")
        self.data["assets"][str(path)] = expected
        return str(path)

    def document(
        self, digest: str, *, path=None, group=None, session=None, legacy_group=None
    ) -> str:
        digest = check_hash(digest)
        key = "document:" + digest
        prior = self.data["documents"].get(key, {})
        group = group or prior.get("group_id", key)
        record = {"source_sha256": digest, "group_id": group}
        if session is not None:
            record["capture_session"] = session
        if legacy_group is not None:
            record["legacy_split_group"] = legacy_group
        prior = self.data["documents"].setdefault(key, record)
        for k, value in record.items():
            if k in prior and prior[k] != value:
                raise ValueError(f"Conflicting document group or session: {digest}")
            prior[k] = value
        if path is not None:
            actual = self.asset(path, digest)
            prior["original_assets"] = sorted(set(prior.get("original_assets", [])) | {actual})
        return key

    def observation(self, row: dict, kind: str, label: str, bounds: list, *, dpi=None) -> str:
        if not isinstance(label, str) or not label:
            raise ValueError("Existing nonempty transcription required")
        digest = row.get("source_hash", row.get("source_sha256"))
        doc = self.document(digest, group=row.get("group_id"))
        page = row.get("page_number", row.get("page"))
        if not isinstance(page, int) or page < 1:
            raise ValueError("Original page number required")
        dpi = dpi or row.get("dpi", row.get("render_dpi"))
        if not isinstance(dpi, (int, float)) or not math.isfinite(dpi) or dpi <= 0:
            raise ValueError("Positive original raster DPI required")
        # Coordinates are converted to physical raster units for cross-DPI identity.
        bounds = box(bounds)
        physical = [round(v / dpi, 10) for v in bounds]
        anchor = {"document": doc, "page": page, "box_inches": physical}
        key = identity("observation", anchor)
        record = {
            **anchor,
            "kind": kind,
            "label": label,
            "group_id": self.data["documents"][doc]["group_id"],
        }
        prior = self.data["observations"].get(key)
        if prior is not None and prior != record:
            raise ValueError(f"Conflicting original label or group: {key}")
        self.data["observations"][key] = record
        return key

    def view(
        self,
        observation: str,
        kind: str,
        asset: str,
        digest: str,
        geometry: dict,
        reference: dict,
        *,
        label: str | None = None,
    ) -> str:
        self.asset(asset, digest)
        record = {
            "observation": observation,
            "kind": kind,
            "asset_sha256": digest,
            "geometry": geometry,
            "label": label,
        }
        key = identity("view", record)
        stored = self.data["views"].setdefault(key, {**record, "source_records": []})
        if reference not in stored["source_records"]:
            stored["source_records"].append(reference)
        if kind == "original-word":
            self.word_crops.setdefault(digest, set()).add(observation)
        return key

    def membership(self, view: str, spec: dict, split: str | None, *, role="source-observation"):
        declared = spec.get("split")
        if declared is not None and split is not None and declared != split:
            raise ValueError("Explicit split conflicts with the historical source split")
        split = split or declared or "unspecified"
        if split in {"holdout", "test", "sealed"}:
            raise ValueError("Sealed/final evaluation cannot be admitted by this importer")
        dataset = spec.get("dataset_id", spec["path"])
        key = identity("membership", [dataset, view])
        record = {
            "dataset_id": dataset,
            "view": view,
            "split": split,
            "exposure": sorted(set(spec.get("exposure", ["unspecified"]))),
            "independent_holdout_claimed": False,
        }
        prior = self.data["memberships"].get(key)
        prior_core = {k: v for k, v in prior.items() if k != "source_roles"} if prior else None
        if prior_core is not None and prior_core != record:
            raise ValueError("Conflicting dataset split or exposure; keep historical membership")
        record["source_roles"] = sorted(
            set(prior.get("source_roles", []) if prior else []) | {role}
        )
        self.data["memberships"][key] = record

    def word(self, row: dict, spec: dict, ref: dict) -> str:
        dpi = row.get("dpi", row.get("render_dpi", spec.get("source_dpi")))
        obs = self.observation(
            row,
            "english-word",
            row["text"],
            add_box(row["parent_bbox"], row["box_in_source_crop"]),
            dpi=dpi,
        )
        source = self.asset(row["source_crop"], row["source_crop_sha256"])
        crop = self.asset(row["crop"], row["crop_sha256"])
        with Image.open(source) as image:
            expected = image.convert("RGB").crop(row["box_in_source_crop"])
        with Image.open(crop) as image:
            actual = image.convert("RGB")
        if actual.size != expected.size or actual.tobytes() != expected.tobytes():
            raise ValueError("Original word pixels do not match the declared parent rectangle")
        view = self.view(
            obs,
            "original-word",
            crop,
            row["crop_sha256"],
            {
                "parent_bbox": row["parent_bbox"],
                "box_in_source_crop": row["box_in_source_crop"],
                "parent_sha256": row["source_crop_sha256"],
                "dpi": dpi,
            },
            ref,
        )
        self.membership(view, spec, row.get("dataset_split", row.get("split")))
        if row.get("raw_training_image"):
            view = self.view(
                obs,
                "normalized-word",
                row["raw_training_image"],
                row["raw_training_image_sha256"],
                {
                    "transform": row.get("target_transform"),
                    "source_word_sha256": row["crop_sha256"],
                },
                {**ref, "asset_field": "raw_training_image"},
            )
            self.membership(
                view, spec, row.get("dataset_split", row.get("split")), role="training-target"
            )
        return obs

    def ingest(self, spec: dict, rows: list[dict], manifest_key: str):
        kind = spec["kind"]
        for index, row in enumerate(rows):
            ref = {
                "manifest": manifest_key,
                "row": index,
                "review_evidence": {k: row[k] for k in REVIEW_FIELDS if k in row},
            }
            if kind == "documents":
                doc = self.document(
                    row["source_hash"],
                    path=row.get("verified_original_path", row.get("source_path")),
                    group=row.get("group_id", row.get("split_group")),
                    session=row.get("capture_session"),
                    legacy_group=row.get("split_group"),
                )
                refs = self.data["documents"][doc].setdefault("source_records", [])
                if ref not in refs:
                    refs.append(ref)
            elif kind == "lines":
                obs = self.observation(
                    row, "english-line", row["verified_text"], row["parent_bbox"]
                )
                view = self.view(
                    obs,
                    "original-line",
                    row["source_crop"],
                    row["source_crop_sha256"],
                    {"box": row["parent_bbox"], "dpi": row["dpi"]},
                    ref,
                )
                self.membership(
                    view, spec, row.get("dataset_split", row.get("existing_dataset_split"))
                )
                for link in row.get("training_word_links", []):
                    child = {
                        "source_hash": row["source_hash"],
                        "page_number": row["page_number"],
                        "text": link["text"],
                        "crop": link["crop"],
                        "crop_sha256": link["sha256"],
                        "source_crop": row["source_crop"],
                        "source_crop_sha256": row["source_crop_sha256"],
                        "parent_bbox": row["parent_bbox"],
                        "box_in_source_crop": link["source_bbox"],
                        "dpi": row["dpi"],
                        "group_id": row.get("group_id"),
                    }
                    self.word(child, spec, {**ref, "word_position": link["position"]})
            elif kind == "words":
                self.word(row, spec, ref)
            elif kind == "glyphs":
                parents = self.word_crops.get(row["source_crop_sha256"], set())
                parents = {
                    p
                    for p in parents
                    if self.data["observations"][p]["document"] == "document:" + row["source_hash"]
                    and self.data["observations"][p]["page"] == row["page_number"]
                }
                if len(parents) != 1:
                    raise ValueError(
                        "Glyph requires one registered original parent word with coordinates"
                    )
                obs = next(iter(parents))
                parent = self.data["observations"][obs]
                if (
                    parent["label"] != row["word"]
                    or not isinstance(row["position"], int)
                    or not 0 <= row["position"] < len(row["word"])
                    or row["word"][row["position"]] != row["label"]
                ):
                    raise ValueError("Glyph label or position conflicts with original word")
                self.asset(row["source_crop"], row["source_crop_sha256"])
                geometry = {
                    "position": row["position"],
                    "source_word_sha256": row["source_crop_sha256"],
                    "source_normalized_bbox": row.get("raw_normalized_bbox"),
                    "transform": row.get("transform"),
                }
                if not any(
                    row.get(field) for field in ("training_image", "image", "before_density_image")
                ):
                    raise ValueError("A glyph derivative asset and hash are required")
                for field, hash_field in [
                    ("training_image", "training_image_sha256"),
                    ("image", "sha256"),
                    ("before_density_image", "before_density_sha256"),
                ]:
                    if row.get(field):
                        view = self.view(
                            obs,
                            "isolated-glyph",
                            row[field],
                            row[hash_field],
                            geometry,
                            {**ref, "asset_field": field},
                            label=row["label"],
                        )
                        is_target = field == "training_image" or (
                            field == "image" and not row.get("training_image")
                        )
                        self.membership(
                            view,
                            spec,
                            row.get("dataset_split", row.get("split")),
                            role="training-target" if is_target else "lineage-reference",
                        )
            elif kind == "math":
                self.document(
                    row["source_sha256"],
                    path=row["pdf"],
                    group=row.get("group_id"),
                    session=row.get("capture_session"),
                )
                obs = self.observation(
                    row, "math-symbol", row["character"], row["box"], dpi=row["render_dpi"]
                )
                base = Path(spec["path"]).parent
                parent = self.asset(base / row["parent_300dpi_crop"], row["parent_300dpi_sha256"])
                crop = self.asset(base / row["crop"], row["crop_sha256"])
                p, c = box(row["parent_box"]), box(row["box"])
                local = [c[0] - p[0], c[1] - p[1], c[2] - p[0], c[3] - p[1]]
                add_box(p, local)
                with Image.open(parent) as image:
                    expected = image.convert("RGB").crop(local)
                with Image.open(crop) as image:
                    actual = image.convert("RGB")
                if actual.size != expected.size or actual.tobytes() != expected.tobytes():
                    raise ValueError(
                        "Original math pixels do not match the declared parent rectangle"
                    )
                view = self.view(
                    obs,
                    "original-math-symbol",
                    crop,
                    row["crop_sha256"],
                    {
                        "source_box": row["box"],
                        "parent_box": row["parent_box"],
                        "parent_asset": parent,
                        "crop_to_pdf": row["crop_to_pdf"],
                        "raster_to_pdf": row["raster_to_pdf"],
                        "dpi": row["render_dpi"],
                    },
                    ref,
                    label=row["character"],
                )
                self.membership(view, spec, row.get("dataset_split", row.get("split")))

    def verify_assets(self):
        self.checked.clear()
        for path, digest in list(self.data["assets"].items()):
            self.asset(path, digest)
        for doc in self.data["documents"].values():
            if not doc.get("original_assets"):
                raise ValueError("Original document asset is missing; supply documents manifest")

    def quality_report(self) -> dict:
        """Keep historical memberships and expose group leakage without rewriting splits."""
        grouped = {}
        for membership in self.data["memberships"].values():
            view = self.data["views"][membership["view"]]
            observation = self.data["observations"][view["observation"]]
            doc = self.data["documents"][observation["document"]]
            # Check both explicit source grouping and shared capture-session identity.
            groups = {
                ("source-group", observation["group_id"]),
                ("source-document", observation["document"]),
            }
            if doc.get("capture_session"):
                groups.add(("capture-session", doc["capture_session"]))
            for kind, group in groups:
                key = identity("group-membership", [membership["dataset_id"], kind, group])
                item = grouped.setdefault(
                    key,
                    {
                        "dataset_id": membership["dataset_id"],
                        "group_kind": kind,
                        "group_id": group,
                        "splits": {},
                    },
                )
                item["splits"].setdefault(membership["split"], set()).add(view["observation"])
        groups = [
            {**v, "splits": {s: sorted(ids) for s, ids in sorted(v["splits"].items())}}
            for v in grouped.values()
        ]
        leaks = [group for group in groups if len(group["splits"]) > 1]
        reviews = {}
        for key, view in self.data["views"].items():
            states = [review_state(r.get("review_evidence", {})) for r in view["source_records"]]
            state = (
                "held-or-unresolved"
                if "held-or-unresolved" in states
                else "source-accepted"
                if "source-accepted" in states
                else "unknown"
            )
            reviews[key] = {
                "kind": view["kind"],
                "label_review_state": state,
                "training_ready": False,
                "reason": "Source review only; model admission needs separate approval.",
            }
        return {
            "dataset_group_splits": groups,
            "group_split_leakage": leaks,
            "group_split_gate": "no-go"
            if leaks
            else "no-detected-conflict-in-registered-memberships",
            "independent_holdout_claimed": False,
            "views": reviews,
            "math_label_review_counts": dict(
                Counter(
                    v["label_review_state"]
                    for v in reviews.values()
                    if v["kind"] == "original-math-symbol"
                )
            ),
            "training_admission": "no-go: registration alone never admits model training",
        }


def register(
    specs: list[dict],
    *,
    parent: Path | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
) -> Path:
    if not specs:
        raise ValueError("At least one explicit source manifest is required")
    normalized = []
    for spec in specs:
        if spec.get("kind") not in KINDS:
            raise ValueError("Unsupported source manifest kind")
        if "exposure" in spec and (
            not isinstance(spec["exposure"], list)
            or not all(isinstance(item, str) for item in spec["exposure"])
        ):
            raise ValueError("Exposure must be an explicit list of descriptions")
        normalized.append({**spec, "path": str(Path(spec["path"]).expanduser().resolve())})
    run = Run.create(
        "federated-corpus-registry",
        workspace=workspace,
        run_id=run_id,
        config={
            "purpose": "private immutable evidence registry; no training or PDF",
            "inputs": normalized,
            "parent": str(parent.resolve()) if parent else None,
            "new_neural_updates": 0,
            "new_real_observations": 0,
            "sealed_C_used": False,
            "label_approval_performed": False,
        },
    )
    (run.path / "logs/corpus_registry.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(run.path / "solutions/import-spec.json", {"manifests": normalized})
    try:
        baseline = None
        parent_digest = None
        if parent:
            verify_run(parent)
            parent_digest = sha256(parent / "run.json")
            baseline = read(parent / "reports/registry.json")
        registry = Registry(baseline)
        registry.verify_assets()
        initial = {k: len(v) for k, v in registry.data.items() if isinstance(v, dict)}
        ordered = {k: i for i, k in enumerate(["documents", "lines", "words", "math", "glyphs"])}
        for spec in sorted(normalized, key=lambda s: ordered[s["kind"]]):
            path = Path(spec["path"])
            payload = path.read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            if "sha256" in spec and actual != spec["sha256"]:
                raise ValueError("Input manifest hash mismatch")
            rows = json.loads(payload)
            if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
                raise ValueError("Source manifest must contain a list of original records")
            key = identity("manifest", [spec["kind"], actual])
            dest = run.path / f"solutions/manifests/{actual}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            registry.data["manifests"].setdefault(
                key,
                {
                    "kind": spec["kind"],
                    "sha256": actual,
                    "original_path": str(path),
                    "records": len(rows),
                },
            )
            registry.ingest(spec, rows, key)
        registry.verify_assets()
        # Keep all inherited raw-record snapshots locally addressable in the new version.
        if parent:
            for source in (parent / "solutions/manifests").glob("*.json"):
                dest = run.path / "solutions/manifests" / source.name
                if not dest.exists():
                    dest.write_bytes(source.read_bytes())
            if sha256(parent / "run.json") != parent_digest:
                raise ValueError("Parent registry changed during registration")
            verify_run(parent)
        atomic_json(run.path / "reports/registry.json", registry.data)
        quality = registry.quality_report()
        atomic_json(run.path / "reports/quality.json", quality)
        counts = {k: len(v) for k, v in registry.data.items() if isinstance(v, dict)}
        result = {
            "schema": SCHEMA,
            "counts": counts,
            "added": {k: v - initial[k] for k, v in counts.items()},
            "observations_by_kind": dict(
                Counter(v["kind"] for v in registry.data["observations"].values())
            ),
            "views_by_kind": dict(Counter(v["kind"] for v in registry.data["views"].values())),
            "unique_asset_hashes": len(set(registry.data["assets"].values())),
            "group_split_leakage_count": len(quality["group_split_leakage"]),
            "group_split_gate": quality["group_split_gate"],
            "math_label_review_counts": quality["math_label_review_counts"],
            "training_admission": quality["training_admission"],
            "declared_training_targets_by_kind": dict(
                Counter(
                    registry.data["views"][view]["kind"]
                    for view in {
                        m["view"]
                        for m in registry.data["memberships"].values()
                        if "training-target" in m.get("source_roles", [])
                    }
                )
            ),
            "parent": str(parent.resolve()) if parent else None,
            "parent_run_sha256": parent_digest,
            "all_original_assets_verified": True,
            "original_labels_and_groups_preserved": True,
            "new_real_observations": 0,
            "new_neural_updates": 0,
            "independent_holdout_claimed": False,
            "sealed_C_used": False,
        }
        run.finish("held", result=result)
        verify_run(run.path)
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=[], metavar="KIND=PATH")
    parser.add_argument(
        "--spec", type=Path, help="JSON object with explicit manifests list and exposure"
    )
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    specs = read(args.spec)["manifests"] if args.spec else []
    for value in args.manifest:
        kind, sep, path = value.partition("=")
        if not sep or not path:
            parser.error("--manifest requires KIND=PATH")
        specs.append({"kind": kind, "path": path})
    print(register(specs, parent=args.parent, workspace=args.workspace))


if __name__ == "__main__":
    main()
