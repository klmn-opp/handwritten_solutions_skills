"""Build one bounded private CPU candidate increment for reviewed capital Sigma.

The three source crops must already have passed the existing extraction/admission
workflow. This utility preserves all original source files and Profile v1, copies
the full 36-source provenance bank, and changes only the candidate's explicit
scan-role inventory. No renderer, production pointer or neural model is modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from hwsol.personal_ink.bundle import load_recipe
from hwsol.personal_ink.kernels.math_glyph_resolver import ScanGlyph
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.schema import ProfileVersion
from hwsol.profiles.store import ProfileStore, load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.export_personal_cpu_candidate import MAX_BYTES, MAX_FILES, AssetCopy

PARENTS = {
    "assets": (
        "20261120T144402240289Z-portable-personal-cpu-assets-597a21c6",
        "99ef82cef8ac94843b6280c70bf69bdbee362db2313323ade11f3794af5526af",
    ),
    "old_bank": (
        "20261120T135929419362Z-session-o-reviewed-glyph-bank-b803206a",
        "5a58f990f4b062f36725015c25c7144fb93d2218113b5be31a1c243983e27fe6",
    ),
    "sigma_bank": (
        "20261120T150125424902Z-session-o-reviewed-glyph-bank-f6df4ece",
        "a93c05a1c332dcc3d5db5e54e5cc7b73f089c98617ffff4e46728d57c40b82fa",
    ),
}


def read(path: Path) -> dict | list:
    return json.loads(path.read_text("utf-8"))


def snapshot_source(run: Run) -> None:
    folder = run.path / "logs/source"
    folder.mkdir()
    for source in [Path(__file__), Path(__file__).with_name("export_personal_cpu_candidate.py")]:
        shutil.copyfile(source, folder / source.name)


def parents(workspace: Path) -> dict[str, Path]:
    result = {}
    for key, (name, expected) in PARENTS.items():
        path = inside(workspace / "runs", name)
        verify_run(path)
        if sha256(path / "run.json") != expected:
            raise ValueError(f"Pinned parent changed: {key}")
        result[key] = path
    return result


def combine_banks(workspace: Path, parent: dict[str, Path]) -> Path:
    old = parent["old_bank"] / "solutions/glyphs"
    addition = parent["sigma_bank"] / "solutions/glyphs"
    old_rows, new_rows = read(old / "sources.json"), read(addition / "sources.json")
    if len(old_rows) != 33 or len(new_rows) != 3:
        raise ValueError("This increment requires exactly 33 old sources and 3 additions")
    if any(row["character"] != "Σ" for row in new_rows):
        raise ValueError("Only capital Sigma U+03A3 belongs in this bounded increment")
    combined = old_rows + new_rows
    if len({row["id"] for row in combined}) != 36:
        raise ValueError("Source IDs collide")
    for row in combined:
        if row["review_status"] != "agent-accepted" or not row["clean_isolated_ink"]:
            raise ValueError("Unreviewed sources are not admitted")
        if row["original_pixels_edited"]:
            raise ValueError("Source ink was modified")
    run = Run.create(
        "session-o-sigma-combined-source-bank",
        workspace=workspace,
        seed=1909,
        config={
            "purpose": "combine approved source banks; no rendering or neural training",
            "parents": {
                key: {"run": str(parent[key]), "sha256": sha256(parent[key] / "run.json")}
                for key in ("old_bank", "sigma_bank")
            },
            "added_role": "U+03A3 capital Sigma, not U+2211 summation",
            "author_review": False,
            "production_activation": False,
        },
    )
    try:
        snapshot_source(run)
        bank = run.path / "solutions/glyphs"
        copier = AssetCopy(bank)
        deduplicated = []
        for source in (old, addition):
            for path in sorted(p for p in source.rglob("*") if p.is_file()):
                relative = path.relative_to(source).as_posix()
                if relative == "sources.json":
                    continue
                if inside(bank, relative).exists():
                    if sha256(path) != sha256(inside(bank, relative)):
                        raise ValueError(f"Different source files share a path: {relative}")
                    deduplicated.append({"source": str(path), "sha256": sha256(path)})
                else:
                    copier.file(source, relative, relative, purpose="original-source-evidence")
        atomic_json(bank / "sources.json", combined)
        for row in combined:
            if sha256(inside(bank, row["crop"])) != row["crop_sha256"]:
                raise ValueError("Combined source crop changed")
            if sha256(inside(bank, row["parent_300dpi_crop"])) != row["parent_300dpi_sha256"]:
                raise ValueError("Combined parent context changed")
            ScanGlyph.load(bank, row)
        for label, source in (("old", parent["old_bank"]), ("sigma", parent["sigma_bank"])):
            shutil.copyfile(
                source / "solutions/glyphs/sources.json",
                run.path / f"reports/{label}-sources-original.json",
            )
            shutil.copyfile(
                source / "reports/source-approval.json",
                run.path / f"reports/{label}-source-approval.json",
            )
        atomic_json(run.path / "reports/copied-assets.json", copier.rows)
        result = {
            "bank": str(bank),
            "sources_sha256": sha256(bank / "sources.json"),
            "accepted_sources": 36,
            "old_source_records_equal": read(bank / "sources.json")[:33] == old_rows,
            "added_source_records_equal": read(bank / "sources.json")[33:] == new_rows,
            "all_source_and_parent_crop_bytes_exact": True,
            "deduplicated_equal_files": deduplicated,
            "source_pdf_sha256": sha256(bank / "source.pdf"),
            "neural_parameter_updates": 0,
            "author_review": False,
            "production_activation": False,
            "rendering_acceptance": "not evaluated by source merge",
        }
        run.finish("held", result=result)
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


def export_increment(workspace: Path, parent: dict[str, Path], combined_run: Path) -> Path:
    verify_run(combined_run)
    prior = load_snapshot(parent["assets"] / "solutions/profile")
    prior_recipe = load_recipe(prior)
    if prior.manifest.ref != "personal-cpu-candidate@v1":
        raise ValueError("This bounded increment requires the frozen v1 candidate")
    bank = combined_run / "solutions/glyphs"
    rows = read(bank / "sources.json")
    bank_relative = prior_recipe.description.banks.session_o
    if read(prior.directory / bank_relative / "sources.json") != rows[:33]:
        raise ValueError("Portable v1 bank differs from the accepted original 33 sources")
    run = Run.create(
        "portable-personal-cpu-sigma-assets",
        workspace=workspace,
        seed=1909,
        config={
            "purpose": "private capital Sigma source increment; no rendering or neural training",
            "profile": "personal-cpu-candidate@v2",
            "parent_profile": prior.manifest.ref,
            "parent_profile_sha256": prior.manifest_sha256,
            "parent_asset_run": str(parent["assets"]),
            "parent_asset_run_sha256": sha256(parent["assets"] / "run.json"),
            "combined_bank_run": str(combined_run),
            "combined_bank_run_sha256": sha256(combined_run / "run.json"),
            "runtime_archive_limits": {"files": MAX_FILES, "bytes": MAX_BYTES},
        },
    )
    try:
        snapshot_source(run)
        destination = run.path / "solutions/profile"
        copier = AssetCopy(destination)
        mutable = {"profile.json", prior.manifest.ink_bundle, f"{bank_relative}/sources.json"}
        for path in sorted(p for p in prior.directory.rglob("*") if p.is_file()):
            relative = path.relative_to(prior.directory).as_posix()
            if relative not in mutable:
                copier.file(prior.directory, relative, relative, purpose="unchanged-v1-asset")
        copier.file(
            bank, "sources.json", f"{bank_relative}/sources.json", purpose="merged-source-metadata"
        )
        for row in rows:
            for path_key, sha_key in (
                ("crop", "crop_sha256"),
                ("parent_300dpi_crop", "parent_300dpi_sha256"),
            ):
                relative = f"{bank_relative}/{row[path_key]}"
                if inside(destination, relative).exists():
                    if sha256(inside(destination, relative)) != row[sha_key]:
                        raise ValueError("Existing portable original crop changed")
                else:
                    copier.file(
                        bank,
                        row[path_key],
                        relative,
                        expected_sha256=row[sha_key],
                        purpose="original-rgb-source-or-parent",
                    )
        recipe = read(prior.directory / prior.manifest.ink_bundle)
        recipe["scan_roles"] = sorted(set(prior_recipe.description.scan_roles) | {"Σ"})
        recipe["admission"]["increment"] = {
            "role": "Σ",
            "codepoint": "U+03A3",
            "source_count": 3,
            "prior_profile": prior.manifest.ref,
            "prior_profile_sha256": prior.manifest_sha256,
            "combined_bank_run": str(combined_run),
            "combined_bank_run_sha256": sha256(combined_run / "run.json"),
            "combined_bank_sha256": sha256(bank / "sources.json"),
            "source_approval_sha256": sha256(parent["sigma_bank"] / "reports/source-approval.json"),
            "review_kind": "two-Agent-source-review; actual placement pending",
            "author_review": False,
            "neural_parameter_updates": 0,
        }
        atomic_json(destination / prior.manifest.ink_bundle, recipe)
        manifest = copy.deepcopy(read(prior.directory / "profile.json"))
        manifest.update(
            version="v2",
            parent=prior.manifest.ref,
            created_at=datetime.now(UTC).isoformat(),
            description=(
                "Private opt-in CPU candidate v2: three reviewed capital Sigma sources; "
                "placement acceptance pending"
            ),
        )
        manifest["build"]["increment"] = run.record()["config"]
        manifest["build"]["original_parent_crops_packaged"] = 36
        prior_roles = {asset.path: asset.role for asset in prior.manifest.assets}
        manifest["assets"] = [
            {
                "path": path.relative_to(destination).as_posix(),
                "sha256": sha256(path),
                "role": prior_roles.get(path.relative_to(destination).as_posix(), "provenance"),
            }
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        ]
        validated = ProfileVersion.model_validate(manifest)
        atomic_json(destination / "profile.json", validated.model_dump(mode="json"))
        loaded = load_snapshot(destination)
        checked_recipe = load_recipe(loaded)
        if (
            loaded.manifest.ref != "personal-cpu-candidate@v2"
            or "Σ" not in checked_recipe.description.scan_roles
        ):
            raise ValueError("New candidate identity or explicit Sigma role is absent")
        files = sorted(path for path in destination.rglob("*") if path.is_file())
        byte_count = sum(path.stat().st_size for path in files)
        if len(files) > MAX_FILES or byte_count > MAX_BYTES:
            raise ValueError("Candidate exceeds existing safe Profile archive limits")
        archive = run.path / "solutions/personal-cpu-candidate-v2.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in files:
                zipped.write(path, path.relative_to(destination).as_posix())
        with tempfile.TemporaryDirectory(prefix="hwsol-sigma-profile-check-") as temp:
            store = ProfileStore(Path(temp))
            store.import_archive(archive)
            imported = store.resolve(loaded.manifest.ref, candidate=True)
            imported_recipe = load_recipe(imported)
            if (
                imported.manifest_sha256 != loaded.manifest_sha256
                or imported_recipe.sha256 != checked_recipe.sha256
            ):
                raise ValueError("Portable archive round trip changed candidate identity")
        for path in sorted(p for p in prior.directory.rglob("*") if p.is_file()):
            relative = path.relative_to(prior.directory).as_posix()
            if relative not in mutable and sha256(path) != sha256(inside(destination, relative)):
                raise ValueError(f"Protected v1 asset changed in v2: {relative}")
        for path in parent.values():
            verify_run(path)
        atomic_json(run.path / "reports/copied-assets.json", copier.rows)
        atomic_json(
            run.path / "reports/bank-inventory.json",
            {
                "session_o": {
                    "accepted_rows": 36,
                    "new_rows": 3,
                    "original_parent_crops": 36,
                    "manifest_sha256": sha256(bank / "sources.json"),
                    "old_records_exact": True,
                },
                "other_banks_byte_exact_to_v1": True,
            },
        )
        result = {
            "candidate_profile": str(destination),
            "profile_sha256": loaded.manifest_sha256,
            "profile_reference": loaded.manifest.ref,
            "parent_profile": prior.manifest.ref,
            "recipe_sha256": checked_recipe.sha256,
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "file_count": len(files),
            "uncompressed_bytes": byte_count,
            "all_copied_assets_byte_exact": True,
            "embedded_base_profile_identities_exact": True,
            "runtime_only_no_research_code": True,
            "raw_source_crops_and_coordinates_retained": True,
            "original_full_documents_copied": False,
            "original_parent_crops_packaged": 36,
            "runtime_archive_limit_changed": False,
            "real_user_store_modified": False,
            "portable_profile_import_and_recipe_validation": True,
            "rendering_acceptance": "not evaluated by asset export",
            "author_review": False,
            "production_activation": False,
            "neural_parameter_updates": 0,
            "combined_bank_run": str(combined_run),
            "combined_bank_run_sha256": sha256(combined_run / "run.json"),
            "old_runs_verified_unchanged": True,
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
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    workspace = workspace_root(args.workspace)
    source = parents(workspace)
    combined = combine_banks(workspace, source)
    print("Combined source bank:", combined, flush=True)
    print("Portable v2 assets:", export_increment(workspace, source, combined), flush=True)


if __name__ == "__main__":
    main()
