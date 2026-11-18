"""Append-only sample rounds; original bytes and template/permission context survive builds."""

import json
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fontTools.ttLib import TTLibError
from pydantic import Field

from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.schema import Asset, FrozenModel, ProfileError, name
from hwsol.profiles.store import BUNDLED, ProfileStore


def now() -> str:
    return datetime.now(UTC).isoformat()


class Sample(FrozenModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^s-[a-f0-9]{32}$")
    profile: str
    kind: Literal["font", "capture", "scan", "archive"]
    created_at: str
    files: tuple[Asset, ...]
    originals: tuple[str, ...]
    template: str | None = None
    license_label: str = "LicenseRef-Unspecified-Local-Only"
    quality: float = Field(default=1, ge=0, le=1)
    weight: float = Field(default=1, ge=0, le=10)
    state: Literal["accepted", "quarantined", "archived"]
    reason: str = ""
    has_recorded_strokes: bool = False


def read_sample(folder: Path) -> Sample:
    sample = Sample.model_validate_json((folder / "sample.json").read_text("utf-8"))
    name(sample.profile)
    if sample.id != folder.name or sample.profile != folder.parent.name:
        raise ProfileError("Sample identity does not match its directory")
    for item in sample.files:
        path = inside(folder, item.path)
        if not path.is_file() or sha256(path) != item.sha256:
            raise ProfileError(f"Original sample integrity check failed: {sample.id}/{item.path}")
    for path in [*sample.originals, *([sample.template] if sample.template else [])]:
        if path not in {a.path for a in sample.files}:
            raise ProfileError("Sample references an unhashed asset")
    return sample


def import_sample(
    store: ProfileStore,
    profile: str,
    sources: list[Path],
    *,
    kind: str | None = None,
    template: Path | None = None,
    quality: float = 1,
    weight: float = 1,
    license_label: str = "LicenseRef-Unspecified-Local-Only",
    license_file: Path | None = None,
) -> Sample:
    if not sources or any(not p.is_file() for p in sources):
        raise ProfileError("Provide at least one existing sample file")
    if sum(p.stat().st_size for p in sources) > 128 * 1024**2:
        raise ProfileError("A sample round is limited to 128 MiB; split large collections")
    folder = store.profile_dir(profile, writable=True)
    if not (folder / "definition.json").exists():
        store.create(profile, f"{BUNDLED}@v1")
    suffix = sources[0].suffix.lower()
    kind = kind or (
        "font"
        if suffix in {".ttf", ".otf"}
        else "capture"
        if suffix == ".json" and template
        else "scan"
        if suffix in {".png", ".jpg", ".jpeg"} and template
        else "archive"
    )
    identifier = "s-" + uuid.uuid4().hex
    samples = inside(store.root, f"samples/{profile}")
    samples.mkdir(parents=True, exist_ok=True)
    with store.writer(profile), tempfile.TemporaryDirectory(prefix=".import-", dir=samples) as tmp:
        stage = Path(tmp)
        assets, originals = [], []
        for index, source in enumerate(sources):
            # Preserve order and suffix without trusting arbitrary uploaded filenames.
            dest = stage / f"original-{index:03d}{source.suffix.lower()}"
            shutil.copyfile(source, dest)
            originals.append(dest.name)
            assets.append(Asset(path=dest.name, sha256=sha256(dest), role="provenance"))
        for source, dest_name, role in (
            (template, "template.json", "provenance"),
            (license_file, "LICENSE.txt", "license"),
        ):
            if source is not None:
                shutil.copyfile(source, stage / dest_name)
                assets.append(Asset(path=dest_name, sha256=sha256(stage / dest_name), role=role))
        state = "archived" if kind == "archive" else "accepted"
        reason = (
            "Unstructured originals retained; label/segment before building"
            if kind == "archive"
            else ""
        )
        if quality < 0.5 or weight == 0:
            state, reason = "quarantined", "Quality below 0.5 or zero user weight"
        sample = Sample(
            id=identifier,
            profile=profile,
            kind=kind,
            created_at=now(),
            files=tuple(assets),
            originals=tuple(originals),
            template="template.json" if template else None,
            license_label=license_label,
            quality=quality,
            weight=weight,
            state=state,
            reason=reason,
            has_recorded_strokes=kind == "capture",
        )
        if state == "accepted":
            try:
                validate_source(sample, stage)
            except (ValueError, KeyError, TypeError, OSError, TTLibError) as exc:
                sample = sample.model_copy(
                    update={
                        "state": "quarantined",
                        "reason": str(exc),
                        "has_recorded_strokes": False,
                    }
                )
        atomic_json(stage / "sample.json", sample.model_dump(mode="json"))
        stage.rename(samples / identifier)
    return sample


def validate_source(sample: Sample, folder: Path) -> None:
    if sample.kind == "font":
        from hwsol.profiles.fonts import inspect_font

        if len(sample.originals) != 1:
            raise ProfileError("One font per import round")
        inspect_font(folder / sample.originals[0])
    elif sample.kind in {"capture", "scan"}:
        from handnote.symbols import Template

        if not sample.template:
            raise ProfileError("Structured capture/scan needs its original template JSON")
        spec = Template.from_dict(json.loads((folder / sample.template).read_text("utf-8")))
        if not spec.cells or [c.index for c in spec.cells] != list(range(len(spec.cells))):
            raise ProfileError("Template cell indices must be sequential")
        if any(len(c.char) != 1 for c in spec.cells) or not 1 <= spec.sets <= 8:
            raise ProfileError("Template requires single-character labels and 1..8 sets")
        if (
            len(spec.cells) > 8192
            or not 1 <= spec.pages <= 128
            or len(spec.page_size) != 2
            or any(not 1 <= size <= 10000 for size in spec.page_size)
            or not 1 <= spec.cell_px <= 2048
        ):
            raise ProfileError("Template dimensions or cell count exceed supported limits")
        if sample.kind == "capture":
            from handnote.capture import capture_set_crops

            if len(sample.originals) != 1:
                raise ProfileError("One capture JSON per round")
            payload = json.loads((folder / sample.originals[0]).read_text("utf-8"))
            capture_set_crops(payload.get("capture", payload), spec)
        else:
            from PIL import Image

            if len(sample.originals) > 8 * spec.pages:
                raise ProfileError("At most eight sets per scan round; import additional rounds")
            for path in sample.originals:
                with Image.open(folder / path) as image:
                    image.verify()


def list_samples(store: ProfileStore, profile: str) -> list[tuple[Path, Sample]]:
    folder = inside(store.root, f"samples/{name(profile)}")
    result = []
    if folder.exists():
        for path in folder.iterdir():
            if path.name.startswith("s-") and path.is_dir():
                result.append((path, read_sample(path)))
    return sorted(result, key=lambda pair: (pair[1].created_at, pair[1].id))
