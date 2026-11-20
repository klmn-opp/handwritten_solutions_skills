"""Local immutable snapshots, atomic activation, and safe runtime-only portability."""

import json
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from hwsol.config import data_dir, test_font
from hwsol.profiles.io import atomic_json, inside, lock, sha256
from hwsol.profiles.schema import ProfileError, ProfileVersion, name, reference
from hwsol.profiles.style import PageStyle

BUNDLED = "bundled-default"


@dataclass(frozen=True)
class LoadedProfile:
    manifest: ProfileVersion
    directory: Path
    manifest_sha256: str
    styles: tuple[PageStyle, ...] = ()

    @property
    def font(self) -> Path:
        return inside(self.directory, self.manifest.font)

    def provenance(self) -> dict:
        return {
            "reference": self.manifest.ref,
            "manifest_sha256": self.manifest_sha256,
            "source_kind": self.manifest.source_kind,
            "distribution": self.manifest.distribution,
            "font_sha256": next(
                a.sha256 for a in self.manifest.assets if a.path == self.manifest.font
            ),
        }


def load_snapshot(directory: Path) -> LoadedProfile:
    path = inside(directory, "profile.json")
    manifest = ProfileVersion.model_validate_json(path.read_text("utf-8"))
    styles = ()
    for asset in manifest.assets:
        target = inside(directory, asset.path)
        if not target.is_file() or sha256(target) != asset.sha256:
            raise ProfileError(f"Asset integrity check failed: {asset.path}")
        if asset.role == "calibration":
            styles = TypeAdapter(tuple[PageStyle, ...]).validate_json(target.read_bytes())
        if asset.role == "glyph-metrics":
            data = json.loads(target.read_bytes())
            if data.get("font_sha256") != next(
                a.sha256 for a in manifest.assets if a.path == manifest.font
            ):
                raise ProfileError("Glyph metrics do not match the Profile font")
    return LoadedProfile(manifest, directory, sha256(path), styles)


def copy_snapshot(loaded: LoadedProfile, destination: Path) -> None:
    """Expose a complete copy by rename; readers cannot see a half-written version."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ProfileError(f"Snapshot already exists: {destination}")
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=destination.parent) as tmp:
        stage = Path(tmp) / "snapshot"
        stage.mkdir()
        for path in ["profile.json", *[a.path for a in loaded.manifest.assets]]:
            target = inside(stage, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inside(loaded.directory, path), target)
        copied = load_snapshot(stage)
        if copied.manifest_sha256 != loaded.manifest_sha256:
            raise ProfileError("Source snapshot changed while copying")
        stage.rename(destination)


def ensure_base(loaded: LoadedProfile, folder: Path) -> None:
    """A crash between base copy and definition write can safely be retried."""
    target = folder / "base"
    if target.exists():
        if load_snapshot(target).manifest_sha256 != loaded.manifest_sha256:
            raise ProfileError("An unfinished Profile has a different base snapshot")
    else:
        copy_snapshot(loaded, target)
    atomic_json(
        folder / "definition.json",
        {
            "profile": folder.name,
            "base": loaded.manifest.ref,
            "base_sha256": loaded.manifest_sha256,
        },
    )


class ProfileStore:
    def __init__(self, root: Path | None = None):
        self.root = (root or data_dir()).expanduser().resolve()

    def profile_dir(self, profile: str, *, writable: bool = False) -> Path:
        name(profile)
        if profile == BUNDLED and writable:
            raise ProfileError("bundled-default is read-only; create a derived Profile")
        return inside(self.root, f"profiles/{profile}")

    def writer(self, profile: str):
        return lock(self.profile_dir(profile, writable=True) / ".writer.lock")

    def settings(self) -> dict:
        path = self.root / "settings.json"
        return json.loads(path.read_text("utf-8")) if path.exists() else {}

    def version_names(self, profile: str, area: str = "versions") -> list[str]:
        if area not in {"versions", "candidates"}:
            raise ProfileError("Unknown snapshot area")
        if profile == BUNDLED:
            return ["v1"] if area == "versions" else []
        folder = self.profile_dir(profile) / area
        found = []
        if folder.exists():
            for item in folder.iterdir():
                if item.is_dir() and (item / "profile.json").is_file():
                    reference(f"{profile}@{item.name}")
                    found.append(item.name)
        return sorted(found, key=lambda v: int(v[1:]))

    def resolve(self, spec: str | None = None, *, candidate: bool = False) -> LoadedProfile:
        settings = self.settings()
        spec = spec or settings.get("default_profile", f"{BUNDLED}@v1")
        if "@" not in spec:
            name(spec)
            versions = self.version_names(spec)
            chosen = settings.get("active_versions", {}).get(spec)
            if not chosen:
                if not versions:
                    raise ProfileError(f"No published versions for {spec}")
                chosen = versions[-1]
            spec = f"{spec}@{chosen}"
        reference(spec)
        profile, version = spec.split("@")
        if profile == BUNDLED:
            if version != "v1" or candidate:
                raise ProfileError("Only bundled-default@v1 is available")
            folder = test_font().parent
        else:
            folder = inside(
                self.profile_dir(profile), f"{'candidates' if candidate else 'versions'}/{version}"
            )
        if not (folder / "profile.json").is_file():
            raise ProfileError(
                f"{'Candidate' if candidate else 'Published Profile'} not found: {spec}"
            )
        loaded = load_snapshot(folder)
        if loaded.manifest.ref != spec:
            raise ProfileError(f"Snapshot identity mismatch: {spec}")
        return loaded

    def create(self, profile: str, base: str = f"{BUNDLED}@v1") -> dict:
        folder = self.profile_dir(profile, writable=True)
        loaded = self.resolve(base)
        with self.writer(profile):
            definition = folder / "definition.json"
            if definition.exists():
                raise ProfileError(f"Profile already exists: {profile}")
            result = {
                "profile": profile,
                "base": loaded.manifest.ref,
                "base_sha256": loaded.manifest_sha256,
            }
            # Copy the base snapshot: removing a source Profile cannot break future training.
            ensure_base(loaded, folder)
        return result

    def activate(self, spec: str) -> dict:
        loaded = self.resolve(spec)
        if loaded.manifest.ink_bundle:
            raise ProfileError("Experimental personal ink cannot become the default Profile")
        with lock(self.root / ".settings.lock"):
            settings = self.settings()
            settings["default_profile"] = loaded.manifest.ref
            settings.setdefault("active_versions", {})[loaded.manifest.profile] = (
                loaded.manifest.version
            )
            atomic_json(self.root / "settings.json", settings)
        return {"active": loaded.manifest.ref}

    def list(self) -> dict:
        names = {BUNDLED}
        folder = self.root / "profiles"
        if folder.exists():
            names.update(
                p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")
            )
        return {
            "default": self.settings().get("default_profile", f"{BUNDLED}@v1"),
            "profiles": [
                {
                    "name": n,
                    "read_only": n == BUNDLED,
                    "published": self.version_names(n),
                    "candidates": self.version_names(n, "candidates"),
                }
                for n in sorted(names)
            ],
        }

    def compare(self, older: str, newer: str) -> dict:
        a, b = self.resolve(older), self.resolve(newer)
        return compare(a, b)

    def export(self, spec: str, output: Path, *, candidate: bool = False) -> dict:
        loaded = self.resolve(spec, candidate=candidate)
        output = output.expanduser().absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects previous portable snapshots.
        try:
            with (
                output.open("xb") as stream,
                zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z,
            ):
                for path in ["profile.json", *[a.path for a in loaded.manifest.assets]]:
                    z.write(inside(loaded.directory, path), path)
        except FileExistsError as exc:
            raise ProfileError(f"Export already exists: {output}") from exc
        return {
            "reference": loaded.manifest.ref,
            "archive": str(output),
            "sha256": sha256(output),
            "contents": (
                "private candidate ink, original crops and provenance; not public distribution"
                if loaded.manifest.ink_bundle
                else "includes explicitly packaged pen traces"
                if any(a.role == "trajectory" for a in loaded.manifest.assets)
                else "runtime assets only; no raw samples"
            ),
        }

    def import_archive(self, source: Path) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".import-", dir=self.root) as tmp:
            stage = Path(tmp)
            with zipfile.ZipFile(source) as archive:
                entries = archive.infolist()
                if len(entries) > 256 or sum(i.file_size for i in entries) > 128 * 1024**2:
                    raise ProfileError("Runtime archive exceeds 256 files or 128 MiB")
                names = [i.filename for i in entries]
                if len({p.casefold() for p in names}) != len(names):
                    raise ProfileError("Duplicate archive entries")
                for item in entries:
                    path = inside(stage, item.filename)
                    if item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):
                        raise ProfileError("Archive must contain regular files only")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as src, path.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
            loaded = load_snapshot(stage)
            expected = {"profile.json", *[a.path for a in loaded.manifest.assets]}
            if set(names) != expected:
                raise ProfileError("Unexpected files in runtime archive")
            m = loaded.manifest
            folder = self.profile_dir(m.profile, writable=True)
            with self.writer(m.profile):
                is_candidate = bool(m.ink_bundle)
                dest = folder / ("candidates" if is_candidate else "versions") / m.version
                if (folder / "versions" / m.version).exists() or (
                    folder / "candidates" / m.version
                ).exists():
                    raise ProfileError(f"Version already exists: {m.ref}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Imported published assets are still subject to hash/format validation.
                from hwsol.profiles.fonts import inspect_font

                details = inspect_font(loaded.font)
                if tuple(details["coverage"]) != m.coverage or details["variants"] != m.variants:
                    raise ProfileError("Font coverage/variant metadata mismatch")
                if is_candidate:
                    from hwsol.personal_ink.bundle import load_recipe

                    load_recipe(loaded)
                if not (folder / "definition.json").exists():
                    ensure_base(loaded, folder)
                copy_snapshot(loaded, dest)
        return {"imported": m.ref, "activated": False, "candidate": is_candidate}


def compare(a: LoadedProfile, b: LoadedProfile) -> dict:
    old, new = set(a.manifest.coverage), set(b.manifest.coverage)
    return {
        "older": a.manifest.ref,
        "newer": b.manifest.ref,
        "added_codepoints": sorted(new - old),
        "removed_codepoints": sorted(old - new),
        "old_variants": a.manifest.variants,
        "new_variants": b.manifest.variants,
        "old_samples": len(a.manifest.samples),
        "new_samples": len(b.manifest.samples),
        "font_changed": sha256(a.font) != sha256(b.font),
        "old_metric_assets": [x.sha256 for x in a.manifest.assets if x.role == "glyph-metrics"],
        "new_metric_assets": [x.sha256 for x in b.manifest.assets if x.role == "glyph-metrics"],
    }
