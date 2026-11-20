"""Cumulative CPU builds and a candidate → evaluation → publication lifecycle."""

import json
import shutil
import uuid
from dataclasses import replace
from importlib.metadata import version as package_version
from pathlib import Path

from fontTools.ttLib import TTFont

from hwsol.adapters.handnote import HandnoteAdapter
from hwsol.identity import build_identity
from hwsol.probe import run_probe
from hwsol.profiles.fonts import inspect_font
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.metrics import measure_font
from hwsol.profiles.schema import Asset, ProfileError, ProfileVersion, SampleIndex
from hwsol.profiles.store import ProfileStore, compare, load_snapshot
from hwsol.training.backend import HandnoteCPU, TrainingBackend, merge_variants
from hwsol.training.samples import list_samples, now

EVALUATION_SOURCE = r"""# Profile evaluation

ABCDEFGHIJKLMNOPQRSTUVWXYZ

abcdefghijklmnopqrstuvwxyz 0123456789

The quick brown fox jumps over the lazy dog.

$$x_1^2+\frac{1}{2}=\frac{3}{2},\quad F=ma=6\,\mathrm{N}$$

$$\int_0^1 x^2\,dx=\frac{1}{3},\quad\sum_{k=1}^{n}k=\frac{n(n+1)}{2}$$

$$\alpha+\beta=\sigma,\quad A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$$
"""


def template(output: Path, chars: str | None = None, sets: int = 1) -> dict:
    from handnote.symbols import write_template

    if output.suffix.lower() != ".pdf":
        raise ProfileError("Template output must end in .pdf")
    if output.exists() or output.with_suffix(".json").exists():
        raise ProfileError("Template output already exists")
    if not 1 <= sets <= 8:
        raise ProfileError("Choose 1..8 sets per template; additional rounds are unlimited")
    if chars is not None and (not chars or len(set(chars)) != len(chars) or len(chars) > 512):
        raise ProfileError("Template needs 1..512 distinct characters")
    spec = write_template(output, out_json=output.with_suffix(".json"), chars=chars, sets=sets)
    return {
        "pdf": str(output),
        "template": str(output.with_suffix(".json")),
        "cells": len(spec.cells),
        "sets": spec.sets,
    }


def train(
    store: ProfileStore,
    profile: str,
    backend: TrainingBackend | None = None,
    *,
    include_traces: bool = False,
) -> dict:
    backend = backend or HandnoteCPU()
    folder = store.profile_dir(profile, writable=True)
    if not (folder / "definition.json").is_file():
        raise ProfileError("Import samples or create a Profile before training")
    with store.writer(profile):
        snapshots = store.version_names(profile) + store.version_names(profile, "candidates")
        version = f"v{max([0, *[int(v[1:]) for v in snapshots]]) + 1}"
        published = store.version_names(profile)
        parent = f"{profile}@{published[-1]}" if published else None
        base = load_snapshot(folder / "base")
        definition = json.loads((folder / "definition.json").read_text("utf-8"))
        if definition["base_sha256"] != base.manifest_sha256:
            raise ProfileError("Base snapshot was modified")
        samples = list_samples(store, profile)
        accepted = [(p, s) for p, s in samples if s.state == "accepted"]
        if not accepted:
            raise ProfileError(
                "No accepted samples; archived/quarantined originals remain preserved"
            )
        # Earlier samples win ties; weight affects preference, not invented geometric distortion.
        accepted.sort(
            key=lambda pair: (-pair[1].quality, -pair[1].weight, pair[1].created_at, pair[1].id)
        )
        run_id = "r-" + uuid.uuid4().hex
        run = inside(store.root, f"training-runs/{profile}/{run_id}")
        run.mkdir(parents=True)
        report = {
            "status": "running",
            "profile": profile,
            "version": version,
            "parent": parent,
            "base": base.manifest.ref,
            "started_at": now(),
            "samples": [s.model_dump(mode="json") for _, s in samples],
            "builds": [],
        }
        atomic_json(run / "run.json", report)
        try:
            current = run / "base.ttf"
            shutil.copyfile(base.font, current)
            personal = set(base.manifest.personal_codepoints)
            from hwsol.training.calibration import measure

            styles = list(base.styles)
            font_samples = [(p, s) for p, s in accepted if s.kind != "calibration"]
            for source, sample in accepted:
                if sample.kind == "calibration":
                    styles.append(measure(source / sample.originals[0], sample.id))
            for index, (source, sample) in enumerate(font_samples):
                part = run / f"sample-{index:04d}.ttf"
                result = backend.build(sample, source, part)
                details = inspect_font(part)
                if details["empty_codepoints"]:
                    raise ProfileError(
                        f"Blank printable glyphs in sample {sample.id}: "
                        f"{details['empty_codepoints']}"
                    )
                if not details["glyf"]:
                    if len(font_samples) != 1 or base.manifest.personal_codepoints:
                        raise ProfileError(
                            "CFF OTF can be used alone; cumulative merging needs glyf fonts"
                        )
                    current = part
                    personal = set(details["coverage"])
                else:
                    merged = run / f"merged-{index:04d}.ttf"
                    personal = merge_variants(current, part, merged, personal)
                    current = merged
                report["builds"].append(result)
            stage = run / "candidate"
            stage.mkdir()
            suffix = ".ttf" if inspect_font(current)["glyf"] else ".otf"
            final_font = stage / f"font{suffix}"
            shutil.copyfile(current, final_font)
            # Give modified glyphs a new family name; keep original font/source licenses.
            with TTFont(final_font) as font:
                family = f"Hwsol {profile} {version}"
                for platform_id, encoding, lang in ((3, 1, 0x409), (1, 0, 0)):
                    for field in (1, 3, 4, 6):
                        value = family.replace(" ", "-") if field == 6 else family
                        font["name"].setName(value, field, platform_id, encoding, lang)
                font.save(final_font)
            assets = [Asset(path=final_font.name, sha256=sha256(final_font), role="font")]
            metrics_file = stage / "glyph-metrics.json"
            atomic_json(metrics_file, measure_font(final_font, tuple(sorted(personal))))
            assets.append(
                Asset(path=metrics_file.name, sha256=sha256(metrics_file), role="glyph-metrics")
            )
            rights = [
                (inside(base.directory, a.path), a.role)
                for a in base.manifest.assets
                if a.role in {"license", "provenance"}
            ]
            rights.extend(
                (inside(p, a.path), a.role)
                for p, sample in accepted
                for a in sample.files
                if a.role == "license"
            )
            if styles:
                calibration = stage / "calibration.json"
                atomic_json(calibration, [s.model_dump(mode="json") for s in styles])
                assets.append(
                    Asset(path=calibration.name, sha256=sha256(calibration), role="calibration")
                )
            if include_traces:
                for item in base.manifest.assets:
                    if item.role == "trajectory":
                        dest = inside(stage, item.path)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(inside(base.directory, item.path), dest)
                        assets.append(item)
                for directory, sample in accepted:
                    if sample.has_recorded_strokes:
                        for filename in [
                            *sample.originals,
                            *([sample.template] if sample.template else []),
                        ]:
                            dest = inside(stage, f"traces/{sample.id}/{filename}")
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(inside(directory, filename), dest)
                            assets.append(
                                Asset(
                                    path=dest.relative_to(stage).as_posix(),
                                    sha256=sha256(dest),
                                    role="trajectory",
                                )
                            )
            for source, role in rights:
                filename = f"{role}/{sha256(source)}{source.suffix}"
                dest = inside(stage, filename)
                if any(a.path == filename for a in assets):
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                assets.append(Asset(path=filename, sha256=sha256(dest), role=role))
            # A portable index carries source identifiers and rights, never raw handwriting.
            provenance = stage / "sources.json"
            atomic_json(
                provenance,
                {
                    "base": base.manifest.ref,
                    "base_sha256": base.manifest_sha256,
                    "samples": [
                        {
                            "id": s.id,
                            "license": s.license_label,
                            "kind": s.kind,
                            "quality": s.quality,
                            "weight": s.weight,
                            "state": s.state,
                        }
                        for _, s in samples
                    ],
                },
            )
            assets.append(Asset(path=provenance.name, sha256=sha256(provenance), role="provenance"))
            details = inspect_font(final_font)
            inherited = list(base.manifest.samples)
            indices = {s.id: s for s in inherited}
            for p, sample in samples:
                indices[sample.id] = SampleIndex(
                    id=sample.id,
                    sha256=sha256(p / "sample.json"),
                    included=sample.state == "accepted",
                )
            manifest = ProfileVersion(
                profile=profile,
                version=version,
                parent=parent,
                base=base.manifest.ref,
                created_at=now(),
                description="Cumulative CPU build from retained local samples",
                source_kind="personal-assets",
                distribution="local-only",
                assets=tuple(assets),
                font=final_font.name,
                coverage=tuple(details["coverage"]),
                variants=details["variants"],
                personal_codepoints=tuple(sorted(personal)),
                samples=tuple(indices.values()),
                render=base.manifest.render,
                build={
                    "backend": "handnote-cpu",
                    "identity": build_identity(),
                    "operation": "font-build-and-statistical-calibration; no neural training",
                    "handnote": package_version("handnote"),
                    "hwsol": package_version("handwritten-solutions"),
                    "run_id": run_id,
                    "strategy": "full-history-rebuild",
                    "primary": "quality-weight-oldest",
                    "variant_sampling": "uniform-upstream-no-synthetic-warps",
                    "calibration_pages": len(styles),
                    "trace_assets_included": any(a.role == "trajectory" for a in assets),
                },
            )
            atomic_json(stage / "profile.json", manifest.model_dump(mode="json"))
            load_snapshot(stage)
            destination = folder / "candidates" / version
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ProfileError("Candidate version collision")
            stage.rename(destination)
            report.update(
                status="candidate",
                candidate=manifest.ref,
                candidate_sha256=sha256(destination / "profile.json"),
            )
        except Exception as exc:
            report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise ProfileError(
                f"Build failed; published versions unchanged. Run: {run}: {exc}"
            ) from exc
        finally:
            report["finished_at"] = now()
            atomic_json(run / "run.json", report)
        return {
            "candidate": manifest.ref,
            "samples": len(samples),
            "accepted": len(accepted),
            "run": str(run),
            "published": False,
            "activated": False,
        }


def evaluate(store: ProfileStore, spec: str) -> dict:
    candidate = store.resolve(spec, candidate=True)
    profile = candidate.manifest.profile
    with store.writer(profile):
        # Re-read under the writer lock so publication cannot race evaluation.
        candidate = store.resolve(spec, candidate=True)
        folder = store.profile_dir(profile)
        run = inside(store.root, f"training-runs/{profile}/e-{uuid.uuid4().hex}")
        run.mkdir(parents=True)
        report = {
            "status": "failed",
            "candidate": spec,
            "candidate_sha256": candidate.manifest_sha256,
            "evaluated_at": now(),
            "manual_style_review": "not_performed",
            "run": str(run),
        }
        try:
            details = inspect_font(candidate.font)
            if details["empty_codepoints"]:
                raise ProfileError(f"Empty printable glyphs: {details['empty_codepoints']}")
            if tuple(details["coverage"]) != candidate.manifest.coverage:
                raise ProfileError("Coverage metadata mismatch")
            if details["variants"] != candidate.manifest.variants:
                raise ProfileError("Variant metadata mismatch")
            if candidate.manifest.parent:
                previous = store.resolve(candidate.manifest.parent)
                report["comparison"] = compare(previous, candidate)
                if report["comparison"]["removed_codepoints"]:
                    raise ProfileError("Candidate lost previously published character coverage")
            else:
                report["comparison"] = compare(load_snapshot(folder / "base"), candidate)
            source = run / "standard.md"
            source.write_text(EVALUATION_SOURCE, encoding="utf-8")
            # Upstream caches open font handles. Keep them outside the directory
            # that publish() renames, including on Windows.
            preview_font = run / candidate.font.name
            shutil.copyfile(candidate.font, preview_font)
            render = run_probe(
                source,
                run / "preview",
                preview_font,
                candidate.manifest.render,
                HandnoteAdapter(real_variants=True),
            )
            if candidate.styles:
                from hwsol.jobs.layout import render_one
                from hwsol.jobs.schema import Background

                blob, diagnostics = render_one(
                    EVALUATION_SOURCE,
                    "1",
                    replace(candidate, directory=run),
                    candidate.manifest.render,
                    Background(),
                    None,
                    "combined",
                )
                preview = run / "combined.pdf"
                preview.write_bytes(blob)
                report["combined_preview"] = {
                    "sha256": sha256(preview),
                    "handwriting": diagnostics["handwriting"],
                    "math_fallback_glyphs": diagnostics["math_fallback_glyphs"],
                }
            report.update(
                status="passed",
                glyphs=details,
                rendering={
                    "page_count": render["page_count"],
                    "warnings": render["warnings"],
                    "artifacts": render["artifacts"],
                },
            )
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic_json(run / "evaluation.json", report)
        atomic_json(folder / "evaluations" / f"{candidate.manifest.version}.json", report)
        return report


def publish(store: ProfileStore, spec: str, *, activate: bool = False) -> dict:
    candidate = store.resolve(spec, candidate=True)
    if candidate.manifest.ink_bundle:
        raise ProfileError("Experimental personal ink requires its separate release qualification")
    profile, version = candidate.manifest.profile, candidate.manifest.version
    with store.writer(profile):
        candidate = store.resolve(spec, candidate=True)
        folder = store.profile_dir(profile)
        assessment = folder / "evaluations" / f"{version}.json"
        if not assessment.is_file():
            raise ProfileError("Evaluate the candidate before publishing")
        report = json.loads(assessment.read_text("utf-8"))
        if (
            report.get("status") != "passed"
            or report.get("candidate_sha256") != candidate.manifest_sha256
        ):
            raise ProfileError("Evaluation failed or is stale; re-evaluate this candidate")
        preview = Path(report["run"]) / "preview"
        if not preview.resolve().is_relative_to(store.root):
            raise ProfileError("Evaluation artifacts must be in the local data directory")
        if candidate.styles:
            combined = preview.parent / "combined.pdf"
            if not combined.is_file() or sha256(combined) != report.get("combined_preview", {}).get(
                "sha256"
            ):
                raise ProfileError("Calibrated preview integrity check failed")
        rendering = report.get("rendering", {})
        count = rendering.get("page_count")
        if type(count) is not int or not 1 <= count <= 9999:
            raise ProfileError("Evaluation is missing its page count")
        # Older Windows probes recorded native separators.
        assets = {k.replace(chr(92), "/"): v for k, v in rendering.get("artifacts", {}).items()}
        required = {"probe.pdf", *(f"pages/{i:02d}.png" for i in range(1, count + 1))}
        if set(assets) != required:
            raise ProfileError("Evaluation preview inventory is incomplete")
        for path, expected in assets.items():
            if sha256(inside(preview, path)) != expected:
                raise ProfileError("Evaluation preview integrity check failed")
        destination = folder / "versions" / version
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ProfileError(f"Published version already exists: {spec}")
        candidate.directory.rename(destination)
    if activate:
        store.activate(spec)
    return {"published": spec, "activated": activate}
