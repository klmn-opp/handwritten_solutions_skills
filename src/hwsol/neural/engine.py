"""Next-generation default backend: frozen personal glyph variation on CPU layout.

The stable personal-CPU page supplies semantics, slot geometry and the teacher ink
targets. The frozen English and mathematical branches replace eligible glyphs; a
deterministic shared-pen/tone correction and integer word spacing finish the page.
No parameter is updated and Torch is not imported.
"""

from __future__ import annotations

import hashlib
import json
import string
from collections import Counter

import numpy as np
from matplotlib import rc_context

from hwsol.adapters.handnote import RENDER_LOCK
from hwsol.jobs.layout import LayoutError, MultiplePages
from hwsol.jobs.schema import Background, Realism
from hwsol.neural import compose, ink, spacing
from hwsol.neural.bundle import load_bundle, load_math_gray
from hwsol.neural.english import EnglishGenerator
from hwsol.neural.math import MathGenerator
from hwsol.personal_ink.kernels.bindings import bind_recipe
from hwsol.personal_ink.runtime import PersonalCpuEngine
from hwsol.profiles.schema import ProfileError

SEED_BASE = 314159
SEED_PAGE = 10000


class NextGenEngine:
    """Frozen next-generation renderer; inherits every stable CPU admission check."""

    def __init__(self, profile, *, qualification: dict | None = None):
        self.profile = profile
        self.qualification = qualification
        self.cpu = PersonalCpuEngine(profile, qualification=qualification)
        self.bundle = load_bundle()
        self.english = EnglishGenerator(self.bundle)
        self.math = MathGenerator(self.bundle)
        self.english_glyphs = self.bundle.glyph_pool()
        self.math_glyphs = self.bundle.math_pool()
        self.ink = ink.load_model()
        self.seed_base = SEED_BASE
        self.manifest_sha256 = hashlib.sha256(
            (self.bundle.root / "manifest.json").read_bytes()
        ).hexdigest()

    # -- frozen draw ------------------------------------------------------
    def _draw_english(self, item):
        label = item["semantic"]
        pool = self.english_glyphs.get(label)
        if not pool:
            return None, None, "no-base-glyph"
        seed = self.seed_base + item["id"]
        source = pool[seed % len(pool)]
        base = self.bundle.english_glyphs["images"][source]
        try:
            generated = self.english.draw(base, label, seed)
        except ValueError:
            return None, None, "model-refused"
        fitted = compose.fit_english(generated, item["stable"])
        if fitted is None:
            return None, source, "fit-rejected"
        return fitted, int(source), "english"

    def _draw_math(self, item):
        label = item["semantic"]
        fallback = item["source_type"] in {"external-font", "inherited-font"}
        pool = self.math_glyphs.get((label, "ordinary"))
        if fallback and not pool:
            pool = [
                row for (char, _), rows in self.math_glyphs.items() if char == label for row in rows
            ]
        anchor = item["stable"]
        source = None
        if pool and (fallback or label in ("ρ", "ϱ")):
            row = pool[(self.seed_base + item["id"]) % len(pool)]
            native = load_math_gray(self.bundle, row)
            try:
                anchored = compose.fit(native, anchor.shape[1], anchor.shape[0])
            except ValueError:
                anchored = None
            if anchored is not None and (anchored < 160).any():
                anchor = anchored
                source = row["id"]
        key = label + "\u241fordinary"
        if key not in self.math.classes or min(anchor.shape) <= 2:
            return None, source, "not-modelled"
        padded = np.pad(anchor, 3, constant_values=255)
        seed = self.seed_base + item["id"]
        try:
            warped, diag = self.math.warp_native(padded, label, seed, "ordinary")
        except ValueError:
            return None, source, "model-refused"
        cropped = warped[3:-3, 3:-3]
        if diag["identity_fallback"] or np.array_equal(cropped, anchor):
            return None, source, "topology-fallback"
        if (
            compose.signature(cropped) != compose.signature(anchor)
            or np.array_equal(cropped, anchor)
            or np.any(warped[:3] < 160)
            or np.any(warped[-3:] < 160)
            or np.any(warped[:, :3] < 160)
            or np.any(warped[:, -3:] < 160)
        ):
            return None, source, "guard-rejected"
        if source is None:
            try:
                if self.math.predict(cropped)["label"] != label:
                    return None, source, "identity-rejected"
            except ValueError:
                return None, source, "identity-error"
        return cropped, source, "math-neural"

    def _draw(self, items):
        replacements = {}
        routes = []
        for item in items:
            if (
                item["context"] == "text"
                and len(item["semantic"]) == 1
                and item["semantic"] in string.ascii_letters
            ):
                shape, source, route = self._draw_english(item)
            elif (
                item["context"] == "mathematics"
                and not item["structure"]
                and not item["matrix_environment"]
            ):
                shape, source, route = self._draw_math(item)
            else:
                shape, source, route = None, None, "layout"
            item["route"] = route
            item["anchor"] = source
            if shape is not None:
                replacements[item["id"]] = shape
            routes.append(dict(id=item["id"], label=item["semantic"], route=route))
        return replacements, routes

    # -- product entry ----------------------------------------------------
    def render_one(
        self,
        source: str,
        number: str,
        profile,
        cfg,
        bg: Background,
        asset,
        realism: Realism = "baseline",
        identity: str | None = None,
    ):
        if bg.kind != "plain" or asset is not None:
            blob, diagnostics = self.cpu.render_one(
                source, number, profile, cfg, bg, asset, realism, identity
            )
            diagnostics["candidate"]["backend"] = "personal-neural@1"
            diagnostics["candidate"]["neural_fallback"] = "non-plain paper uses stable CPU ink"
            diagnostics["handwriting"]["neural_routes"] = {"layout": "stable-cpu"}
            return blob, diagnostics
        if profile.manifest_sha256 != self.cpu.bundle.candidate.manifest_sha256:
            raise ProfileError("Candidate Profile changed during rendering")
        if (
            realism != "baseline"
            or cfg.policy is not None
            or cfg.glyph_metrics != "raw"
            or cfg.math_strokes != "personal"
        ):
            raise ValueError(
                "Personal neural recipe requires baseline realism, raw metrics, personal math "
                "strokes and no renderer policy"
            )
        if any(ord(ch) not in profile.manifest.coverage for ch in number if not ch.isspace()):
            raise ValueError(f"Missing glyph in problem number {number}")
        settings = cfg.model_copy(update={"paper": bg.kind})
        self.seed_base = (
            SEED_BASE
            + int(settings.seed) * SEED_PAGE
            + (int.from_bytes(hashlib.sha256(number.encode()).digest()[:4], "big") % 1_000_000)
        )
        with (
            RENDER_LOCK,
            rc_context(),
            bind_recipe(self.cpu.bundle.description.bindings.model_dump()),
        ):
            try:
                page, renderer, records, diagnostics = self.cpu.compose(
                    compose.prepared_source(source), number, settings
                )
            except MultiplePages as exc:
                raise LayoutError(
                    [{"page_count": exc.count, "reason": "candidate exceeds one problem per page"}]
                ) from exc
            records = compose.substitute_punctuation(page, renderer, records, self.cpu.bundle)
            full, items = compose.item_tiles(page, renderer, records, settings)
            replacements, routes = self._draw(items)
            final, admitted = compose.compose_page(full.shape, items, replacements, full)
            areas = compose.areas_from_page(final, items)
            target = ink.document_target(areas)
            outputs, ink_logs = self.ink.correct_regions(areas, mode="combined", target=target)
            toned = compose.paste_tone(final, areas, outputs)
            contours = compose.check_page_contours(final, toned)
            words = compose.text_word_groups(records, settings)
            text_owners = [area for area in areas if area["kind"] == "text"]
            spacing_log = {"schema": "word-gap-integer-applied/1", "moved_letters": 0}
            if words and {w for word in words for w in word["glyphs"]} >= {
                area["id"] for area in text_owners
            }:
                plan = spacing.plan_spacing(toned, text_owners, words)
                final, spacing_log = spacing.apply_spacing(toned, text_owners, plan)
            else:
                spacing_log["held"] = "incomplete word ownership; stable profile gaps kept"
            yy, xx = np.where(final < 255)
            if not len(xx):
                raise ValueError("Neural page lost all ink")
            bounds = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
            if (
                min(bounds[:2]) < 3
                or bounds[2] > final.shape[1] - 3
                or bounds[3] > final.shape[0] - 3
            ):
                raise ValueError("Personal ink crosses page boundary; page rejected")
        import io
        from pathlib import Path
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            temp = Path(handle.name)
        try:
            compose.write_a4([final], temp)
            blob = temp.read_bytes()
        finally:
            temp.unlink(missing_ok=True)
        from pypdf import PdfReader

        if len(PdfReader(io.BytesIO(blob), strict=True).pages) != 1:
            raise ValueError("Neural page PDF page count changed")
        counts = Counter(item["route"] for item in admitted)
        candidate = {
            "backend": "personal-neural@1",
            "model_bundle": "nextgen-v1",
            "model_manifest_sha256": self.manifest_sha256,
            "neural_weight_updates": 0,
            "glyph_routes": dict(counts),
            "items": len(admitted),
            "ink_target": target,
            "ink_changes": sum(1 for row in ink_logs if row["changed_pixels"]),
            "spacing": spacing_log,
            "page_contours": contours,
            "identity": identity or number,
            "fallback_count": sum(1 for item in admitted if item["route"] == "layout"),
            "source_selection_seed": settings.seed,
            "profile": profile.provenance(),
            "recipe": self.cpu.bundle.description.recipe,
            "source_representation": (
                "compiled-contours"
                if self.cpu.bundle.description.recipe == "compiled-outlines/1"
                else "reviewed-original-crops"
            ),
            "raw_sources_included": self.cpu.bundle.description.recipe == "reviewed-outlines/1",
            "recipe_sha256": self.cpu.bundle.sha256,
            "base_profile": self.cpu.bundle.base.provenance(),
            "local_release_qualification": self.qualification,
            "production_qualified": bool(self.qualification),
            "qualification_scope": (
                "installed compiled author preset and verified release environment"
                if self.qualification
                and self.qualification.get("scope") == "packaged-compiled-author-preset"
                else "this local ProfileStore and exact build; "
                "qualification does not activate defaults"
                if self.qualification
                else "experimental candidate only"
            ),
            "visible_math_items": sum(1 for item in items if item["kind"] == "math"),
            "draw_routes": routes,
            "items_detail": [
                {
                    "id": item["id"],
                    "label": item["semantic"],
                    "kind": item["kind"],
                    "box": item["box"],
                    "route": item["route"],
                    "anchor": item.get("anchor"),
                }
                for item in items
            ],
        }
        diagnostics["candidate"] = candidate
        diagnostics["handwriting"] = {
            "mode": "reviewed-personal-cpu-layout-plus-frozen-neural-variation",
            "glyph_ink": "fill-only",
            "math_size": "body-em/full-size-fractions",
            "planned_lines": renderer.rows,
            "structure_fallbacks": getattr(renderer, "structure_fallbacks", []),
        }
        return blob, json.loads(json.dumps(diagnostics, default=str))
