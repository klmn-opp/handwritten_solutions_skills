"""Export reviewed CPU assets into a new, private data-only candidate Profile.

This development utility reads only the explicitly pinned prior runs. It does not
publish or activate a Profile, modify old receipts, or access the original corpus.
Runtime code lives in hwsol; no research module is copied into the portable data.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.schema import ProfileVersion
from hwsol.profiles.store import load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root

ROOT = Path(__file__).resolve().parents[1]
PARENTS = {
    "baseline": "20261120T115012607677Z-author-feedback-final-20-pages-50910a3a",
    "round2": "20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6",
    "round3": "20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e",
    "session_o": "20261120T135929419362Z-session-o-reviewed-glyph-bank-b803206a",
    "reviewed_delivery": "20261120T141332892747Z-session-o-reviewed-24-page-delivery-e4bc22e8",
}
PINNED_RECEIPTS = {
    "round2": "c6a2fbc94e384335271262828de7c7e1e1a329932527b42a2cf5b0e963e3323f",
    "round3": "9419cb9713df8d5118603bd166f5f6c4ec31e2148afcea5b2058a11e4789248f",
    "session_o": "5a58f990f4b062f36725015c25c7144fb93d2218113b5be31a1c243983e27fe6",
    "reviewed_delivery": "e0c65bd37fdc3c5af478a29364a7a669b2f9de97314909dea83d82a9a45c6075",
}
MAX_FILES = 256
MAX_BYTES = 128 * 1024**2


def read(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def literal_constant(module: str, name: str) -> Any:
    """Read a frozen data binding without importing research drivers or corpora."""
    source = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    tree = ast.parse(source.read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal source binding: {module}.{name}")


def bindings() -> dict:
    font_sha = literal_constant("research.math_latin_feedback", "FONT_SHA")
    if font_sha != literal_constant("research.numeral_pi_feedback", "FONT_SHA"):
        raise ValueError("Different Latin and digit calibration fonts")
    return {
        "font_sha256": font_sha,
        "digit_sample_ids": literal_constant("research.numeral_pi_feedback", "SAMPLES"),
        "horizontal_source_ids": list(
            literal_constant("research.structural_rule_feedback", "SOURCE_IDS")
        ),
        "horizontal_source_sha256": literal_constant(
            "research.structural_rule_feedback", "SOURCE_SHA"
        ),
        "single_bar_source_id": literal_constant("research.math_paired_bars", "SAMPLE"),
    }


class AssetCopy:
    """Copy exact source files and retain an explicit input-to-output audit map."""

    def __init__(self, destination: Path):
        self.destination = destination
        self.rows: list[dict] = []

    def file(
        self,
        source_root: Path,
        source_path: str,
        target_path: str,
        *,
        expected_sha256: str | None = None,
        purpose: str = "runtime",
    ) -> None:
        src = inside(source_root, source_path)
        dest = inside(self.destination, target_path)
        digest = sha256(src)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"Approved source file changed: {source_path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as incoming, dest.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha256(dest) != digest:
            raise ValueError(f"Copied source file changed: {target_path}")
        self.rows.append(
            {
                "source": str(src),
                "destination": target_path,
                "sha256": digest,
                "bytes": dest.stat().st_size,
                "purpose": purpose,
                "byte_exact": True,
            }
        )

    def profile(self, source: Path, target: str) -> dict:
        loaded = load_snapshot(source)
        self.file(source, "profile.json", f"{target}/profile.json", purpose="base-identity")
        for asset in loaded.manifest.assets:
            self.file(
                source,
                asset.path,
                f"{target}/{asset.path}",
                expected_sha256=asset.sha256,
            )
        clone = load_snapshot(self.destination / target)
        if clone.manifest_sha256 != loaded.manifest_sha256:
            raise ValueError("Embedded base Profile identity changed")
        return {
            "path": target,
            "manifest_sha256": loaded.manifest_sha256,
            "reference": loaded.manifest.ref,
            "font_sha256": sha256(loaded.font),
        }

    def scan_bank(self, source: Path, target: str) -> dict:
        rows = read(source / "sources.json")
        self.file(source, "sources.json", f"{target}/sources.json", purpose="source-metadata")
        accepted = [row for row in rows if row["review_status"] == "agent-accepted"]
        seen: set[str] = set()
        for row in accepted:
            if row.get("original_pixels_edited") or not row["clean_isolated_ink"]:
                raise ValueError(f"Unapproved edited/connected glyph: {row['id']}")
            if row["crop"] not in seen:
                self.file(
                    source,
                    row["crop"],
                    f"{target}/{row['crop']}",
                    expected_sha256=row["crop_sha256"],
                    purpose="original-rgb-crop",
                )
                seen.add(row["crop"])
        return {
            "path": target,
            "source_rows": len(rows),
            "accepted_rows": len(accepted),
            "copied_crops": len(seen),
            "manifest_sha256": sha256(source / "sources.json"),
            "held_row_metadata_preserved": True,
            "held_crops_not_runtime_assets": len(rows) - len(accepted),
        }

    def historical_bank(self, source: Path, target: str) -> dict:
        manifest = read(source / "manifest.json")
        self.file(source, "manifest.json", f"{target}/manifest.json", purpose="source-metadata")
        for row in manifest["sources"]:
            if row["review_status"] != "agent-accepted" or row["original_pixels_edited"]:
                raise ValueError("Historical selected-contour source not accepted")
            for path_key, sha_key in (
                ("crop", "crop_sha256"),
                ("contour_archive", "contour_archive_sha256"),
                ("reviewed_component_selection", "reviewed_component_selection_sha256"),
            ):
                self.file(
                    source,
                    row[path_key],
                    f"{target}/{row[path_key]}",
                    expected_sha256=row[sha_key],
                    purpose="reviewed-historical-component",
                )
        return {
            "path": target,
            "source_rows": len(manifest["sources"]),
            "selected": manifest["selected"],
            "connected_component_policy_unchanged": True,
        }

    def f_bank(self, source: Path, target: str) -> dict:
        candidates = read(source / "candidates.json")
        self.file(source, "candidates.json", f"{target}/candidates.json", purpose="source-metadata")
        sheet = next(row for row in candidates if row["id"] == "sheet-f1")
        self.file(
            source,
            "source-sample.ttf",
            f"{target}/source-sample.ttf",
            expected_sha256=sheet["source"]["font_source_sha256"],
        )
        for row in candidates:
            self.file(
                source,
                row["outline"],
                f"{target}/{row['outline']}",
                expected_sha256=row["outline_sha256"],
                purpose="reviewed-derived-outline",
            )
        # These exact colour cells/word crops supplement the unchanged metadata.
        # Full collection sheets and unrelated source candidates stay in old runs.
        for name in (
            "sheet-0/raw-rgb-cell.png",
            "sheet-0/source.json",
            "sheet-1/raw-rgb-cell.png",
            "sheet-1/source.json",
            "support-079/f-original-rgb.png",
            "support-079/word-original-rgb.png",
            "support-079/source.json",
        ):
            self.file(source, name, f"{target}/{name}", purpose="original-source-evidence")
        return {
            "path": target,
            "candidate_ids": [row["id"] for row in candidates],
            "runtime_selection": "balanced; baseline-f0 or sheet-f1 only",
            "unselected_historical_outline_retained_for_exact_loader_contract": True,
        }


def verified_parents(workspace: Path) -> dict[str, Path]:
    result = {key: workspace / "runs" / name for key, name in PARENTS.items()}
    for key, path in result.items():
        verify_run(path)
        if key in PINNED_RECEIPTS and sha256(path / "run.json") != PINNED_RECEIPTS[key]:
            raise ValueError(f"Pinned source receipt changed: {key}")
    approved_base = read(result["round2"] / "reports/approval-receipt.json")["baseline_sha256"]
    if sha256(result["baseline"] / "run.json") != approved_base:
        raise ValueError("First author revision no longer matches its approved parent receipt")
    return result


def export_candidate(author_feedback: Path, *, workspace: Path | None = None) -> Path:
    workspace = workspace_root(workspace)
    parents = verified_parents(workspace)
    feedback = read(author_feedback)
    if (
        feedback.get("candidate_run") != PARENTS["reviewed_delivery"]
        or feedback.get("candidate_run_sha256") != PINNED_RECEIPTS["reviewed_delivery"]
        or not feedback.get("author_text", "").strip()
    ):
        raise ValueError("Author feedback must identify the exact delivered candidate")
    run = Run.create(
        "portable-personal-cpu-assets",
        workspace=workspace,
        seed=1909,
        config={
            "purpose": "private candidate asset export; no rendering or neural training",
            "profile": "personal-cpu-candidate@v1",
            "author_feedback_sha256": sha256(author_feedback),
            "parents": {
                key: {"run_id": path.name, "run_sha256": sha256(path / "run.json")}
                for key, path in parents.items()
            },
            "runtime_archive_limits": {"files": MAX_FILES, "bytes": MAX_BYTES},
        },
    )
    try:
        snapshot_dir = run.path / "logs/source"
        snapshot_dir.mkdir()
        shutil.copyfile(__file__, snapshot_dir / Path(__file__).name)
        atomic_json(
            run.path / "reports/source-identity.json",
            {
                "exporter_sha256": sha256(Path(__file__)),
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "git_status": subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ),
            },
        )
        destination = run.path / "solutions/profile"
        destination.mkdir()
        copier = AssetCopy(destination)
        base = parents["baseline"] / "solutions"
        profiles = {
            "base": copier.profile(base / "profile-v1", "ink/base-profile"),
            "stroke": copier.profile(base / "profile-v3", "ink/stroke-profile"),
        }
        source_banks = {
            "legacy": (base / "historical-glyphs", "ink/banks/legacy"),
            "sessions": (base / "session-glyphs", "ink/banks/sessions"),
            "rules": (
                parents["round2"] / "solutions/structural-sources",
                "ink/banks/rules",
            ),
            "session_o": (parents["session_o"] / "solutions/glyphs", "ink/banks/session-o"),
        }
        inventories = {
            key: copier.scan_bank(source, target) for key, (source, target) in source_banks.items()
        }
        inventories["historical"] = copier.historical_bank(
            base / "new-historical-glyphs", "ink/banks/historical"
        )
        inventories["f_variants"] = copier.f_bank(
            parents["round3"] / "solutions/f-sources", "ink/banks/f-variants"
        )
        recipe_bindings = bindings()
        if profiles["base"]["font_sha256"] != recipe_bindings["font_sha256"]:
            raise ValueError("The measured recipe does not match the base font")
        recipe = {
            "schema": "personal-cpu-recipe/1",
            "recipe": "reviewed-outlines/1",
            "base_profile": profiles["base"]["path"],
            "base_profile_sha256": profiles["base"]["manifest_sha256"],
            "stroke_profile": profiles["stroke"]["path"],
            "stroke_profile_sha256": profiles["stroke"]["manifest_sha256"],
            "calibrated_font_sha256": recipe_bindings["font_sha256"],
            "banks": {key: row["path"] for key, row in inventories.items()},
            "bindings": recipe_bindings,
            "admission": {
                "author_feedback": feedback,
                "author_feedback_sha256": sha256(author_feedback),
                "review_parent_sha256": PINNED_RECEIPTS["reviewed_delivery"],
                "scope": "Broad author satisfaction with delivered pages; not production release",
                "status": "candidate-only",
                "neural_parameter_updates": 0,
            },
        }
        atomic_json(destination / "ink/recipe.json", recipe)
        original = read(base / "profile-v1/profile.json")
        manifest = copy.deepcopy(original)
        manifest.update(
            profile="personal-cpu-candidate",
            version="v1",
            parent=original["profile"] + "@" + original["version"],
            base=original["profile"] + "@" + original["version"],
            created_at=datetime.now(UTC).isoformat(),
            description="Private, opt-in reviewed personal CPU candidate; not production published",
            source_kind="derived",
            distribution="local-only",
            font=f"{profiles['base']['path']}/{original['font']}",
            ink_bundle="ink/recipe.json",
            render=read(parents["baseline"] / "run.json")["config"]["render"],
            build={
                "kind": "portable-reviewed-personal-cpu-assets/1",
                "parents": run.record()["config"]["parents"],
                "source_metadata_preserved": True,
                "original_full_documents": "retained in original evidence runs; not runtime assets",
                "source_context_paths": "historical locators, not required runtime file paths",
                "neural_updates": 0,
                "production_activation": False,
            },
        )
        manifest["assets"] = [
            {
                "path": path.relative_to(destination).as_posix(),
                "sha256": sha256(path),
                "role": "personal-ink"
                if path.relative_to(destination).as_posix() == manifest["ink_bundle"]
                else "font"
                if path.relative_to(destination).as_posix() == manifest["font"]
                else "provenance",
            }
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        ]
        validated = ProfileVersion.model_validate(manifest)
        atomic_json(destination / "profile.json", validated.model_dump(mode="json"))
        loaded = load_snapshot(destination)
        if loaded.manifest.ref != "personal-cpu-candidate@v1":
            raise ValueError("Candidate Profile identity changed")
        files = sorted(path for path in destination.rglob("*") if path.is_file())
        byte_count = sum(path.stat().st_size for path in files)
        if len(files) > MAX_FILES or byte_count > MAX_BYTES:
            raise ValueError("Candidate exceeds existing safe Profile archive limits")
        archive = run.path / "solutions/personal-cpu-candidate-v1.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for path in files:
                z.write(path, path.relative_to(destination).as_posix())
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != {p.relative_to(destination).as_posix() for p in files}:
                raise ValueError("Portable archive inventory differs from Profile")
        atomic_json(run.path / "reports/copied-assets.json", copier.rows)
        atomic_json(run.path / "reports/bank-inventory.json", inventories)
        shutil.copyfile(author_feedback, run.path / "reports/author-feedback.json")
        result = {
            "candidate_profile": str(destination),
            "profile_sha256": loaded.manifest_sha256,
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "file_count": len(files),
            "uncompressed_bytes": byte_count,
            "all_copied_assets_byte_exact": True,
            "embedded_base_profile_identities_exact": True,
            "runtime_only_no_research_code": True,
            "raw_source_crops_and_coordinates_retained": True,
            "original_full_documents_copied": False,
            "runtime_archive_limit_changed": False,
            "real_user_store_modified": False,
            "rendering_acceptance": "not evaluated by asset export",
            "production_activation": False,
            "neural_parameter_updates": 0,
        }
        run.finish("prepared", result=result)
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-feedback", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    print(export_candidate(args.author_feedback, workspace=args.workspace))


if __name__ == "__main__":
    main()
