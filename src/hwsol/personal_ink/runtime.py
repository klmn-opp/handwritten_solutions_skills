"""A generic, explicitly selected CPU candidate using reviewed original outlines.

No run lookup, page-number dispatch, pre-rendered word lookup or model loading is
permitted here. A failed page is rejected by the ordinary atomic job transaction.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from matplotlib import rc_context

from hwsol.adapters.handnote import RENDER_LOCK
from hwsol.config import RenderConfig
from hwsol.jobs.layout import LayoutError, MultiplePages, ink_geometry
from hwsol.jobs.paper import paper_image, visible_ink
from hwsol.jobs.schema import Background, Realism
from hwsol.personal_ink.bundle import load_recipe
from hwsol.personal_ink.kernels.author_feedback_revision import f_collision_check, identity, trace
from hwsol.personal_ink.kernels.bindings import bind_recipe
from hwsol.personal_ink.kernels.cpu_glyph_feedback import CpuGlyphFeedback, layout_with_r_metrics
from hwsol.personal_ink.kernels.f_size_feedback import FSizeFeedback
from hwsol.personal_ink.kernels.f_variation_feedback import FVariationFeedback
from hwsol.personal_ink.kernels.historical_math_candidate import (
    HistoricalMathFeedback,
    load_sources,
)
from hwsol.personal_ink.kernels.math_feedback_candidate import RadicalClearancePolish
from hwsol.personal_ink.kernels.math_glyph_completion import SELECTION
from hwsol.personal_ink.kernels.math_glyph_resolver import MathGlyphResolver, matrix_strokes
from hwsol.personal_ink.kernels.math_latin_feedback import MathLatinFeedback
from hwsol.personal_ink.kernels.math_paired_bars import PersonalPairedBars, source_bar
from hwsol.personal_ink.kernels.math_session_glyphs import SessionGlyphCalibration
from hwsol.personal_ink.kernels.math_session_polish import SessionInkPolish
from hwsol.personal_ink.kernels.numeral_pi_feedback import NumeralPiFeedback
from hwsol.personal_ink.kernels.render import accepted_scans
from hwsol.personal_ink.kernels.session_o_resolver import SessionOResolver
from hwsol.personal_ink.kernels.structural_rule_feedback import StructuralRuleFeedback
from hwsol.personal_ink.validation import audit_new_contacts, capture, validate_session_increment
from hwsol.profiles.schema import ProfileError
from hwsol.profiles.store import LoadedProfile


class PersonalCpuEngine:
    """The same data-driven transform order for every problem and page."""

    def __init__(self, profile: LoadedProfile, *, qualification: dict | None = None):
        self.bundle = load_recipe(profile)
        self.qualification = qualification

    def render_one(
        self,
        source: str,
        number: str,
        profile: LoadedProfile,
        cfg: RenderConfig,
        bg: Background,
        asset: Path | None,
        realism: Realism = "baseline",
        identity: str | None = None,
    ) -> tuple[bytes, dict]:
        if profile.manifest_sha256 != self.bundle.candidate.manifest_sha256:
            raise ProfileError("Candidate Profile changed during rendering")
        if any(ord(ch) not in profile.manifest.coverage for ch in number if not ch.isspace()):
            raise ValueError(f"Missing glyph in problem number {number}")
        if (
            realism != "baseline"
            or cfg.policy is not None
            or cfg.glyph_metrics != "raw"
            or cfg.math_strokes != "personal"
        ):
            raise ValueError(
                "Personal CPU recipe requires baseline realism, raw metrics, personal math "
                "strokes and no renderer policy"
            )
        settings = cfg.model_copy(update={"paper": bg.kind if asset is None else "plain"})
        with RENDER_LOCK, rc_context(), bind_recipe(self.bundle.description.bindings.model_dump()):
            try:
                page, renderer, records, diagnostics = self.compose(source, number, settings)
            except MultiplePages as exc:
                raise LayoutError(
                    [{"page_count": exc.count, "reason": "candidate exceeds one problem per page"}]
                ) from exc
            bounds, _ = ink_geometry(page, renderer)
            if min(bounds[:2]) < 3 or bounds[2] > page.width - 3 or bounds[3] > page.height - 3:
                raise ValueError("Personal ink crosses page boundary; page rejected")
            if asset is not None:
                page.background = paper_image(bg, asset, (page.width, page.height))
                page.paper = None
            pdf = renderer.to_vector_pdf([page])
            if bg.kind != "plain":
                from dataclasses import replace

                if not visible_ink(pdf, renderer.to_vector_pdf([replace(page, items=[])])):
                    raise ValueError("Ink is not visible against the selected paper")
            math = [row for row in records if row["context"] == "mathematics"]
            fallbacks = []
            for row in math:
                if row["source_type"] in {"external-font", "inherited-font"}:
                    reason = row.setdefault(
                        "fallback_reason", "no reviewed compatible personal source admitted"
                    )
                    fallbacks.append(
                        {
                            "item_index": row["item_index"],
                            "semantic": row["semantic"],
                            "codepoint": row.get("codepoint"),
                            "font": row.get("font"),
                            "source_type": row["source_type"],
                            "reason": reason,
                        }
                    )
            return pdf, json.loads(
                json.dumps(
                    {
                        "attempts": [
                            {
                                "page_count": 1,
                                "ink_bounds_px": bounds,
                                "font_size_pt": settings.font_size_pt,
                                "margin_mm": settings.margin_mm,
                                "candidate_automatic_resize": False,
                            }
                        ],
                        "math_fallback_glyphs": fallbacks,
                        "glyph_provenance": records,
                        "formula_groups": renderer.formulas,
                        "matrix_groups": renderer.matrices,
                        "candidate": {
                            "backend": "personal-cpu-qualified"
                            if self.qualification
                            else "personal-cpu-experimental",
                            "local_release_qualification": self.qualification,
                            "recipe": self.bundle.description.recipe,
                            "source_representation": (
                                "compiled-contours"
                                if self.bundle.description.recipe == "compiled-outlines/1"
                                else "reviewed-original-crops"
                            ),
                            "raw_sources_included": (
                                self.bundle.description.recipe == "reviewed-outlines/1"
                            ),
                            "recipe_sha256": self.bundle.sha256,
                            "profile": profile.provenance(),
                            "base_profile": self.bundle.base.provenance(),
                            "problem_identity": identity or number,
                            "visible_math_items": len(math),
                            "math_source_types": dict(Counter(r["source_type"] for r in math)),
                            "fallback_count": len(fallbacks),
                            "source_selection_seed": cfg.seed,
                            "neural_weight_updates": 0,
                            "production_qualified": bool(self.qualification),
                            "qualification_scope": (
                                "installed compiled author preset and verified release environment"
                                if self.qualification
                                and self.qualification.get("scope")
                                == "packaged-compiled-author-preset"
                                else "this local ProfileStore and exact build; "
                                "qualification does not activate defaults"
                                if self.qualification
                                else "experimental candidate only"
                            ),
                            **diagnostics,
                        },
                        "handwriting": {
                            "mode": "reviewed-personal-cpu-recipe",
                            "glyph_ink": "fill-only",
                            "math_size": "body-em/full-size-fractions",
                            "planned_lines": renderer.rows,
                            "structure_fallbacks": getattr(renderer, "structure_fallbacks", []),
                        },
                    }
                )
            )

    def compose(self, source: str, title: str, cfg: RenderConfig) -> tuple:
        """Compose from content alone; the caller owns the render/binding scope."""
        base, strokes = self.bundle.base, self.bundle.strokes
        banks = self.bundle.directories
        scans = accepted_scans(banks["legacy"])
        resolver = MathGlyphResolver(
            {g.source["character"]: g for g in scans}, strokes, selected=SELECTION
        )
        sessions = SessionGlyphCalibration(accepted_scans(banks["sessions"]))
        page, renderer, records = layout_with_r_metrics(source, title, base, cfg)
        raw_geometry = capture(page, renderer, records)
        expected = copy.deepcopy(records)
        for i, (char, _) in matrix_strokes(page, renderer).items():
            expected[i]["semantic"] = char
        parser_before = copy.deepcopy(trace(renderer))
        records = resolver.resolve(page, renderer, records, base)
        records = sessions.apply(page, renderer, records, operator_page=False)
        records = SessionInkPolish(sessions, base.font).apply(page, renderer, records)
        before_items = list(page.items)
        records = CpuGlyphFeedback(base.font).apply(page, renderer, records)
        collision = f_collision_check(page, renderer, before_items, records)
        if any(row["new_contacts"] for row in collision):
            raise ValueError("CPU f calibration introduced a glyph contact")
        clearance = RadicalClearancePolish()
        records = clearance.apply(page, renderer, records)
        records = MathLatinFeedback(base.font).apply(page, renderer, records)
        records = PersonalPairedBars(source_bar(banks["sessions"])).apply(page, renderer, records)
        records = HistoricalMathFeedback(load_sources(banks["historical"]), strokes).apply(
            page, renderer, records, base
        )
        records = NumeralPiFeedback(base.font).apply(page, renderer, records)
        fsize = FSizeFeedback("D-common-source", source=source, title=title)
        records = fsize.apply(page, renderer, records)
        rules = StructuralRuleFeedback(banks["rules"])
        records = rules.apply(page, renderer, records)
        variants = FVariationFeedback(
            banks["f_variants"], source=source, title=title, seed=cfg.seed
        )
        records = variants.apply(page, renderer, records)
        roles = frozenset(self.bundle.description.scan_roles)
        session_o = SessionOResolver(banks["session_o"], seed=cfg.seed, targets=roles)
        before_session = capture(page, renderer, records)
        records = session_o.apply(page, renderer, records, case_id=title)
        source_checks = validate_session_increment(
            before_session, page, renderer, records, banks["session_o"], targets=roles
        )
        contact_audit = audit_new_contacts(raw_geometry, page, renderer, records)
        if contact_audit["unexpected_contacts"]:
            raise ValueError(
                "Personal ink introduced unreviewed contacts: "
                + json.dumps(contact_audit["unexpected_contacts"])
            )
        if identity(expected) != identity(records) or parser_before != trace(renderer):
            raise ValueError("Personal ink changed visible identity or parsed formula structure")
        if len(page.items) != len(records) or [r["item_index"] for r in records] != list(
            range(len(records))
        ):
            raise ValueError("Personal ink item inventory changed")
        return (
            page,
            renderer,
            records,
            {
                "semantic_identity_preserved": True,
                "parser_trace_preserved": True,
                "raw_glyph_sources_preserved": True,
                "radical_clearance_decisions": clearance.decisions,
                "f_size_decisions": fsize.decisions,
                "f_variant_decisions": variants.decisions,
                "session_o_decisions": session_o.decisions,
                "cpu_f_contacts": collision,
                "session_o_geometry_checks": source_checks,
                "whole_recipe_contact_audit": contact_audit,
            },
        )
