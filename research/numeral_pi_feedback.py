"""Bounded real-source numeral/pi audit; no source-ink modification or training."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fontTools.ttLib import TTFont
from handnote.opentype import read_alternates
from handnote.symbols import MARKER_DICT
from handnote.variants import handwritten_outline
from handnote.vector import Outline
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw
from pypdf import PdfWriter

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run
from research.author_feedback_revision import identity, trace
from research.author_feedback_round2 import load_baseline, replay_page
from research.corpus.math_glyph_completion import local_comparison, raster
from research.corpus.math_session_glyphs import outline_bounds
from research.corpus.math_session_polish import body_width
from research.homework_benchmark.render import save_pdf, source_paths
from research.math_feedback_candidate import paired_panel
from research.math_latin_feedback import contacts

ROOT = Path(__file__).resolve().parents[1]
FONT_SHA = "60466bbde4f39280838ac15f1121fc975cf841f42b581f74618eff50c8b2bcd9"
SAMPLES = ["s-c222860c59414473bf606d42eade46b9", "s-6f23c7cfdf304bafa153c2b86c002a3c"]
TRAINING = (
    ROOT / "profiles/realism-studio/training-runs/personal/r-e9ef6e0126b84037ad2633745370d424"
)


def read(path):
    return json.loads(Path(path).read_text())


def source_two(font, destination):
    """Trace all four existing contours to their accepted raw sheets, without relabeling."""
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in range(4):
        sample_index, source_variant = divmod(variant, 2)
        sample = ROOT / "profiles/realism-studio/samples/personal" / SAMPLES[sample_index]
        manifest, template = read(sample / "sample.json"), read(sample / "template.json")
        if manifest["state"] != "accepted":
            raise ValueError("Source is not accepted")
        for f in manifest["files"]:
            if sha256(sample / f["path"]) != f["sha256"]:
                raise ValueError("Raw accepted source hash mismatch")
        source_font = TRAINING / f"sample-{sample_index:04d}.ttf"
        v, c = handwritten_outline(font, 50, variant, 1000)
        original, source_codes = handwritten_outline(source_font, 50, source_variant, 1000)
        if not np.array_equal(c, source_codes) or v.shape != original.shape:
            raise ValueError("Compiled two contour topology mismatch")
        a, b = v[c != MplPath.CLOSEPOLY], original[c != MplPath.CLOSEPOLY]
        deviation = float(
            np.abs((a - a.min(0)) / np.ptp(a, axis=0) - (b - b.min(0)) / np.ptp(b, axis=0)).max()
        )
        if deviation > 0.004:
            raise ValueError("Unverified merged contour lineage")
        raw_path = sample / f"original-{source_variant:03d}.jpg"
        rgb = np.asarray(Image.open(raw_path).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        scale = min(1.0, 1800.0 / max(gray.shape))
        small = (
            cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1
            else gray
        )
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(MARKER_DICT), cv2.aruco.DetectorParameters()
        )
        corners, ids, _ = detector.detectMarkers(small)
        known = {x["id"]: x for x in template["markers"]}
        found = [
            (q.reshape(4, 2) / scale, known[int(i)])
            for q, i in zip(corners, ids.ravel(), strict=True)
            if int(i) in known
        ]
        if len(found) < 2 or any(m["sheet"] != source_variant for _, m in found):
            raise ValueError("Source scan registration lacks reliable sheet identity")
        src = np.concatenate([q for q, _ in found]).astype(np.float32)
        dst = np.array([p for _, m in found for p in m["points"]], dtype=np.float32)
        matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        warped = cv2.warpPerspective(
            rgb,
            matrix,
            tuple(template["page_size_px"]),
            flags=cv2.INTER_LINEAR,
            borderValue=(255, 255, 255),
        )
        cell = next(x for x in template["cells"] if x["char"] == "2")
        x0, y0, x1, y1 = cell["bbox"]
        crop_path = destination / f"two-{variant}-registered-rgb.png"
        Image.fromarray(warped[y0:y1, x0:x1]).save(crop_path)
        shutil.copyfile(raw_path, destination / f"two-{variant}-original.jpg")
        raw_quad = cv2.perspectiveTransform(
            np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32),
            np.linalg.inv(matrix),
        )[0]
        rx0, ry0 = np.floor(raw_quad.min(0)).astype(int)
        rx1, ry1 = np.ceil(raw_quad.max(0)).astype(int)
        raw_crop = destination / f"two-{variant}-original-rgb-crop.png"
        Image.fromarray(rgb).crop((rx0, ry0, rx1, ry1)).save(raw_crop)
        rows.append(
            {
                "character": "2",
                "variant": variant,
                "sample_id": SAMPLES[sample_index],
                "source_variant": source_variant,
                "sample_state": "accepted",
                "original_path": str(raw_path),
                "original_sha256": sha256(raw_path),
                "sample_manifest": manifest,
                "template_cell": cell,
                "template_sha256": sha256(sample / "template.json"),
                "raw_to_registered": matrix.tolist(),
                "registered_to_raw": np.linalg.inv(matrix).tolist(),
                "raw_cell_quad": raw_quad.tolist(),
                "raw_crop_box": [int(rx0), int(ry0), int(rx1), int(ry1)],
                "registered_dpi": template["dpi"],
                "registered_crop": str(crop_path.relative_to(destination.parent)),
                "raw_crop": str(raw_crop.relative_to(destination.parent)),
                "sample_font": str(source_font),
                "sample_font_sha256": sha256(source_font),
                "final_font_sha256": sha256(font),
                "codes_exact": True,
                "normalized_vertex_max_delta": deviation,
                "bounds_em1000": [a.min(0).tolist(), a.max(0).tolist()],
                "body_width_p90_em1000_diagnostic_only": body_width(MplPath(v, c)),
                "new_label": False,
                "original_pixels_edited": False,
            }
        )
    atomic_json(destination / "two-lineage.json", rows)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for k in range(4):
        axes[0, k].imshow(Image.open(destination / f"two-{k}-registered-rgb.png"))
        axes[0, k].set_title(f"2 variant {k}\nRegistered original RGB")
        v, c = handwritten_outline(font, 50, k, 1000)
        axes[1, k].add_patch(PathPatch(MplPath(v, c), color="black", lw=0))
        axes[1, k].set(
            xlim=(-60, 870),
            ylim=(-80, 760),
            aspect="equal",
            title=f"Compiled variant {k}\nFixed 1000-em scale",
        )
        for ax in axes[:, k]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination / "two-real-variants.png", dpi=150)
    plt.close(fig)
    return rows


def audit(path):
    b = load_baseline()
    path = Path(path)
    (path / "logs/source").mkdir(parents=True, exist_ok=True)
    for p in (Path(__file__), ROOT / "research/author_feedback_round2.py"):
        shutil.copyfile(p, path / "logs/source" / p.name)
    source_two(b["profile"].font, path / "solutions/two-sources")
    pi = b["resolver"].glyphs["π"]
    old = b["path"] / "solutions/historical-glyphs"
    new = path / "solutions/pi-source"
    new.mkdir()
    for key in ("crop", "parent_300dpi_crop"):
        source = old / pi.source[key]
        shutil.copyfile(source, new / ("glyph.png" if key == "crop" else "parent.png"))
    atomic_json(new / "source.json", pi.source)
    inventory, counts = [], Counter()
    for case in b["cases"]:
        n = case["page"]
        records = read(b["path"] / "reports" / (case["case_id"] + "-provenance.json"))["candidate"]
        image = Image.open(b["path"] / f"rendered/pages/{n:02d}.png").convert("RGB")
        page_targets = []
        for row in records:
            if row.get("semantic") not in {"2", "π"}:
                continue
            box = row["bounds_px"]
            crop = [
                int(np.floor(box[0])) - 24,
                int(np.floor(box[1])) - 16,
                int(np.ceil(box[2])) + 65,
                int(np.ceil(box[3])) + 18,
            ]
            dest = path / f"comparisons/original-targets/{n:02d}/{row['item_index']:04d}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.crop(crop).save(dest)
            item = {
                "page": n,
                "case_id": case["case_id"],
                "record": copy.deepcopy(row),
                "crop_box": crop,
                "crop": str(dest.relative_to(path)),
            }
            inventory.append(item)
            page_targets.append(item)
            counts[(row["semantic"], row["context"], row["variant"])] += 1
        sheet = Image.new("RGB", (1000, 155 * ((len(page_targets) + 3) // 4)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, item in enumerate(page_targets):
            rr = item["record"]
            x, y = (j % 4) * 250, (j // 4) * 155
            draw.text(
                (x + 5, y + 3),
                f"P{n} item{rr['item_index']} "
                f"{'pi' if rr['semantic'] == 'π' else '2'} v{rr['variant']} {rr['context']}",
                fill="black",
            )
            im = Image.open(path / item["crop"])
            im.thumbnail((240, 125))
            sheet.paste(im, (x + 5, y + 25))
        sheet.save(path / f"comparisons/original-targets/{n:02d}/contact.png")
    atomic_json(path / "reports/all-targets.json", inventory)
    atomic_json(
        path / "reports/counts.json",
        [
            {"semantic": k[0], "context": k[1], "variant": k[2], "count": v}
            for k, v in counts.items()
        ],
    )
    print(path, flush=True)


class NumeralPiFeedback:
    """Select real lighter 2v0 for heavy 2v2/3; explicitly retain unsupported pi."""

    def __init__(self, font):
        self.font = Path(font)
        if sha256(self.font) != FONT_SHA:
            raise ValueError("Requires pinned and traced personal v1 font")
        with TTFont(self.font) as t:
            name = t.getBestCmap()[50]
            self.name, self.gid = name, t.getGlyphID(name)
            self.names = [name, *read_alternates(t).get(name, [])]
        self.decisions = []

    def apply(self, page, renderer, records):
        before_items, before = list(page.items), copy.deepcopy(records)
        result = copy.deepcopy(records)
        self.decisions = []
        for old, row in zip(before, result, strict=True):
            if old.get("semantic") not in {"2", "π"}:
                continue
            i = old["item_index"]
            d = {
                "item_index": i,
                "semantic": old["semantic"],
                "context": old["context"],
                "original_variant": old.get("variant"),
                "applied": False,
            }
            if old["semantic"] == "π":
                d.update(
                    status="explicit-no-reliable-lighter-source",
                    reason=(
                        "Only accepted pi U03C0-p55; its real cap/legs are heavy. "
                        "Existing fit has no confirmed scaling defect. "
                        "No source ink thinned or redrawn."
                    ),
                    source_id=old.get("scan_source", {}).get("id"),
                    existing_transform=old.get("transform_scale_translate"),
                )
            elif old.get("variant") in {0, 1}:
                d.update(
                    status="unchanged-real-lighter-variant", reason="Retain existing accepted v0/v1"
                )
            elif old.get("variant") not in {2, 3} or old.get("font_sha256") != FONT_SHA:
                d.update(
                    status="explicit-original-source-retained",
                    reason="Unsupported variant or font identity",
                )
            else:
                item = before_items[i]
                if item[0] != "path":
                    raise ValueError("Two requires the recorded outline path")
                v, c = handwritten_outline(self.font, 50, 0, 1000)
                # Preserve the old item's orientation/shear; source units cancel in the
                # following uniform fit, including MathText's 100 vs text's 1000 em.
                transformed = Affine2D.from_values(*item[2]).transform(v)
                lo, hi = outline_bounds(transformed, c)
                low, high = np.array(old["bounds_px"]).reshape(2, 2)
                scale = float(np.min((high - low) / (hi - lo)))
                shift = (high + low) / 2 - (hi + lo) * scale / 2
                fitted = transformed * scale + shift
                outline = Outline(fitted, c, stroke=False, digits=4)
                key = ("numeral-pi-feedback", "2", i)
                renderer._outlines[key] = outline
                page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), item[3], 0)
                bounds = outline_bounds(fitted, c).ravel().tolist()
                neighbor = contacts(i, before_items, page, renderer, result)
                errors = []
                if any(x["new_contact"] for x in neighbor):
                    errors.append("new neighboring ink contact")
                if (
                    min(
                        bounds[0] - low[0],
                        bounds[1] - low[1],
                        high[0] - bounds[2],
                        high[1] - bounds[3],
                    )
                    < -0.02
                ):
                    errors.append("outside original ink envelope")
                if min(bounds[0], bounds[1], page.width - bounds[2], page.height - bounds[3]) < 3:
                    errors.append("page boundary clearance below 3 px")
                # Nonsingular uniform affine mapping preserves the accepted source
                # contour's topology. This is not an OCR/recognizability certificate.
                source_contours = int(np.count_nonzero(c == MplPath.MOVETO))
                if scale <= 0 or source_contours != 1 or not np.isfinite(fitted).all():
                    errors.append("unsupported source topology or affine transform")
                d.update(
                    candidate_bounds=bounds,
                    original_bounds=old["bounds_px"],
                    neighbors=neighbor,
                    rejection_reasons=errors,
                    source_contours=source_contours,
                    source_topology_preserved=True,
                )
                if errors:
                    page.items[i] = item
                    d.update(status="rejected-original-retained")
                else:
                    treatment = {
                        "method": (
                            "select authentic 2v0; uniformly fit inside previous ink envelope"
                        ),
                        "original_variant": old["variant"],
                        "selected_variant": 0,
                        "source_font_sha256": FONT_SHA,
                        "source_sample_id": SAMPLES[0],
                        "source_sample_variant": 0,
                        "original_item_matrix": list(item[2]),
                        "post_matrix_uniform_scale": scale,
                        "post_matrix_translation": shift.tolist(),
                        "source_to_output_affine": (
                            Affine2D.from_values(*item[2]).get_matrix()
                            * np.array([[scale, scale, scale], [scale, scale, scale], [1, 1, 1]])
                            + np.array([[0, 0, shift[0]], [0, 0, shift[1]], [0, 0, 0]])
                        ).tolist(),
                        "ink_width_before_after_px": [
                            float(high[0] - low[0]),
                            bounds[2] - bounds[0],
                        ],
                        "ink_height_before_after_px": [
                            float(high[1] - low[1]),
                            bounds[3] - bounds[1],
                        ],
                        "left_right_clearance_added_px": [bounds[0] - low[0], high[0] - bounds[2]],
                        "original_bounds_px": old["bounds_px"],
                        "source_topology_preserved": True,
                        "source_contours": source_contours,
                        "original_pixels_edited": False,
                        "new_label": False,
                        "neural_updates": 0,
                        "author_acceptance": False,
                        "status": "isolated-research-candidate",
                        "no_new_neighbor_contact": True,
                    }
                    row.update(
                        bounds_px=bounds,
                        allowed_bounds_px=old["bounds_px"],
                        variant=0,
                        actual_glyph_id=self.gid,
                        glyph_name=self.name,
                        numeral_pi_feedback=treatment,
                    )
                    d.update(applied=True, status="accepted-for-visual-review", treatment=treatment)
            self.decisions.append(d)
        if identity(before) != identity(result):
            raise ValueError("Glyph semantics/order changed")
        return result


def protect_pixels(before, after, original, candidate, changed, *, page, renderer, before_items):
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape:
        raise ValueError("Page dimensions changed")
    diff = np.any(a != b, axis=2)
    allowed, protected = np.zeros(diff.shape, bool), np.zeros(diff.shape, bool)
    for old, new in zip(original, candidate, strict=True):
        x0, y0, x1, y1 = old["bounds_px"]
        if old["item_index"] in changed:
            if old["semantic"] != "2":
                raise ValueError("Unapproved changed character")
            pad = 2
            target = allowed
        else:
            if old != new:
                raise ValueError("Non-target provenance changed")
            pad = 0
            target = protected
        target[
            max(0, int(np.floor(y0)) - pad) : min(a.shape[0], int(np.ceil(y1)) + pad),
            max(0, int(np.floor(x0)) - pad) : min(a.shape[1], int(np.ceil(x1)) + pad),
        ] = True
    outside = int((diff & ~allowed).sum())
    box_overlap = int((diff & protected).sum())
    other = box_overlap
    if box_overlap:
        # A curved neighbor's rectangular bounds can include the adjacent superscript
        # or numeral's ink. Render the original non-target vectors and unchanged paper
        # separately to protect their actual antialiased pixels, including thin ink.
        # Vector equality was already checked; no target is erased from source images.
        non_target = raster(
            renderer.to_vector_pdf(
                [replace(page, items=[it for i, it in enumerate(before_items) if i not in changed])]
            )
        )
        background = raster(renderer.to_vector_pdf([replace(page, items=[])]))
        protected_ink = np.any(np.asarray(non_target) != np.asarray(background), axis=2)
        other = int((diff & protected_ink).sum())
    if outside or other:
        raise ValueError(f"Non-target pixels changed: outside={outside}, neighbor_ink={other}")
    return {
        "changed_pixels": int(diff.sum()),
        "outside_target_slots": outside,
        "non_target_item_box_overlap_pixels": box_overlap,
        "non_target_ink_pixels": other,
        "exact_non_target_ink_mask_used": bool(box_overlap),
        "changed_targets": len(changed),
        "non_target_records_exact": True,
    }


def candidate(pages, audit_run):
    b = load_baseline()
    run = Run.create(
        "numeral-pi-real-two-candidate",
        config={
            "baseline": str(b["path"]),
            "baseline_run_sha256": b["run_sha256"],
            "source_audit": str(audit_run),
            "pages": pages,
            "render": b["cfg"].model_dump(),
            "neural_updates": 0,
            "production_activation": False,
            "pi_policy": "no reliable lighter accepted source; retain original explicitly",
            "two_policy": "v2/v3 to authentic v0, uniform fit inside old ink envelope",
        },
        seed=b["cfg"].seed,
    )
    print(run.path, flush=True)
    try:
        shutil.copytree(Path(audit_run) / "solutions", run.path / "solutions/source-evidence")
        shutil.copytree(b["path"] / "solutions", run.path / "solutions/baseline")
        atomic_json(run.path / "reports/parent-run.json", read(b["path"] / "run.json"))
        source_hashes = {}
        paths = set(source_paths() + list((ROOT / "research").glob("*.py")))
        for source in paths:
            dest = run.path / "logs/source" / source.resolve().relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            source_hashes[str(source.resolve())] = sha256(source)
        atomic_json(run.path / "reports/source-identity.json", source_hashes)
        summaries, outputs, panels = [], [], []
        for n in pages:
            print(f"page {n}", flush=True)
            r = replay_page(b, n)
            page, renderer, old = r["page"], r["renderer"], r["records"]
            before_items = list(page.items)
            feedback = NumeralPiFeedback(b["profile"].font)
            new = feedback.apply(page, renderer, old)
            if json.loads(json.dumps(trace(renderer))) != r["parser"]:
                raise ValueError("Parser trace changed")
            dest = run.path / f"rendered/pages/{n:02d}.pdf"
            image = save_pdf(renderer.to_vector_pdf([page]), dest)
            outputs.append(dest)
            changed = {d["item_index"] for d in feedback.decisions if d["applied"]}
            if any(
                item != page.items[i] for i, item in enumerate(before_items) if i not in changed
            ):
                raise ValueError("Non-target vector item changed")
            checks = protect_pixels(
                r["image"],
                image,
                old,
                new,
                changed,
                page=page,
                renderer=renderer,
                before_items=before_items,
            )
            case = r["entry"]["case_id"]
            atomic_json(
                run.path / f"reports/{case}-provenance.json",
                {
                    "before": old,
                    "candidate": new,
                    "parser_trace": r["parser"],
                    "decisions": feedback.decisions,
                },
            )
            summary = {
                "page": n,
                "case_id": case,
                "replay_pixels_exact": True,
                "replay_provenance_exact": True,
                "parser_trace_exact": True,
                "checks": checks,
                "decisions": feedback.decisions,
            }
            summaries.append(summary)
            paired_panel(
                r["image"],
                image,
                run.path / f"comparisons/pages/{n:02d}.png",
                f"Page {n}; original pi retained",
                ("Accepted prior candidate", "Real lighter 2v0"),
            )
            for d in feedback.decisions:
                i = d["item_index"]
                box = old[i]["bounds_px"]
                crop = [
                    int(np.floor(box[0])) - 25,
                    int(np.floor(box[1])) - 18,
                    int(np.ceil(box[2])) + 65,
                    int(np.ceil(box[3])) + 20,
                ]
                a, z = r["image"].crop(crop), image.crop(crop)
                folder = run.path / f"comparisons/targets/{n:02d}"
                folder.mkdir(parents=True, exist_ok=True)
                a.save(folder / f"{i:04d}-before.png")
                z.save(folder / f"{i:04d}-candidate.png")
                if d["applied"]:
                    panels.append((n, i, a, z))
        with PdfWriter() as writer:
            for p in outputs:
                writer.append(p)
            writer.write(run.path / "rendered/homework_solutions.pdf")
        if panels:
            folder = run.path / "comparisons/all-changed"
            folder.mkdir()
            local_comparison(folder, panels, labels=("Previous 2", "Real lighter 2v0"))
        atomic_json(run.path / "reports/cases.json", summaries)
        atomic_json(
            run.path / "reports/summary.json",
            {
                "pages": len(pages),
                "targets": sum(len(x["decisions"]) for x in summaries),
                "applied": sum(x["checks"]["changed_targets"] for x in summaries),
                "rejected": [
                    {"page": x["page"], **d}
                    for x in summaries
                    for d in x["decisions"]
                    if d["status"] == "rejected-original-retained"
                ],
                "pi_changed": 0,
                "author_acceptance": False,
                "status": "independent native visual review pending",
            },
        )
        print(run.path, flush=True)
        return run.path
    except Exception as e:
        run.fail(e)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--pages", default="2,6,18")
    args = parser.parse_args()
    if args.audit:
        audit(args.audit)
    elif args.candidate:
        candidate([int(n) for n in args.pages.split(",")], args.source_audit)
