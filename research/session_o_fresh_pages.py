# ruff: noqa: E501
"""Fresh explicit solutions exercising reviewed O symbols in varied math contexts.

These are supplementary engineering examples, not a reused independent holdout.
Both arms use the same CPU English, formulas, page layout and white background.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import sympy as sp
from pypdf import PdfWriter

from hwsol.jobs.schema import Background
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.author_feedback_delivery import json_value, snapshot_sources
from research.author_feedback_revision import group_crops, personal_layout, trace
from research.author_feedback_round2_checks import page_geometry
from research.corpus.math_glyph_completion import local_comparison
from research.f_size_feedback import FSizeFeedback
from research.f_variation_feedback import FVariationFeedback
from research.homework_benchmark.render import fallback_summary, save_pdf
from research.math_feedback_candidate import paired_panel
from research.math_latin_feedback import MathLatinFeedback
from research.session_o_baseline import load_baseline, validate_increment
from research.session_o_delivery import verify_source_placements
from research.session_o_resolver import PRIORITY, SessionOResolver

FRESH_TARGETS = PRIORITY | frozenset("*")

CASES = [
    (
        "O1 - Potential and oscillations",
        r"""Find the field and curvature of the potential below. Then evaluate the given oscillation at the specified point.

$$\Phi(\xi,\eta)=\kappa(\xi^2+2\eta^2),\qquad\kappa=\frac{3}{2}$$

Differentiate each coordinate independently. The negative gradient gives the field.

$$\vec E=-\nabla\Phi=(-2\kappa\xi,-4\kappa\eta)$$

$$H=\begin{bmatrix}2\kappa&0\\0&4\kappa\end{bmatrix}=\begin{bmatrix}3&0\\0&6\end{bmatrix}$$

At the requested point the potential is positive and the two field components have opposite signs.

$$\Phi(1,-1)=\frac{9}{2},\qquad\vec E(1,-1)=(-3,6)$$

For the second part use the product and chain rules.

$$\psi(\xi,\eta)=\sin(\pi\xi)\cos(\pi\eta)$$

$$\frac{\partial\psi}{\partial\xi}=\pi\cos(\pi\xi)\cos(\pi\eta)$$

$$\psi\left(\frac{1}{2},0\right)=1,\qquad\frac{\partial\psi}{\partial\xi}\left(\frac{1}{2},0\right)=0$$

For a squared damping ratio of one quarter, take the positive square root.

$$\zeta=\sqrt{\frac{1}{4}}=\frac{1}{2},\qquad\psi_{\xi\eta}=-\pi^2\cos(\pi\xi)\sin(\pi\eta)$$
""",
    ),
    (
        "O2 - Complex inner products",
        r"""Show that the following two vectors form an orthonormal basis. Use conjugation in the first argument of the inner product.

$$u=\frac{1}{\sqrt{2}}\begin{pmatrix}1\\\mathrm{i}\end{pmatrix},\qquad v=\frac{1}{\sqrt{2}}\begin{pmatrix}\mathrm{i}\\1\end{pmatrix}$$

The conjugate of the imaginary unit has the opposite sign. Compute both norms before the cross term.

$$\langle u,u\rangle=\frac{1+(-\mathrm{i})\mathrm{i}}{2}=1$$

$$\langle v,v\rangle=\frac{(-\mathrm{i})\mathrm{i}+1}{2}=1$$

$$\langle u,v\rangle=\frac{\mathrm{i}-\mathrm{i}}{2}=0$$

Normalizing the two arguments preserves their orthogonality.

$$\left\langle\frac{u}{\sqrt{\langle u,u\rangle}},\frac{v}{\sqrt{\langle v,v\rangle}}\right\rangle=0$$

Collect the basis vectors as columns. Direct multiplication gives the identity matrix.

$$U=\frac{1}{\sqrt{2}}\begin{bmatrix}1&\mathrm{i}\\\mathrm{i}&1\end{bmatrix},\qquad U^*U=\begin{bmatrix}1&0\\0&1\end{bmatrix}$$

$$\mathrm{det}(U)=\frac{1-\mathrm{i}^2}{2}=1$$
""",
    ),
    (
        "O3 - Products and likelihood",
        r"""Evaluate two finite products and find the exponential rate estimate for a sample with known sum.

$$P=\prod_{k=1}^{4}\frac{k}{k+1}=\frac{1}{2}\frac{2}{3}\frac{3}{4}\frac{4}{5}=\frac{1}{5}$$

The adjacent factors cancel. A second product telescopes with larger numerators than denominators.

$$Q=\prod_{k=1}^{3}\left(1+\frac{1}{k}\right)=2\frac{3}{2}\frac{4}{3}=4$$

For independent observations from an exponential density, multiply the densities and then take logarithms.

$$L(\lambda)=\prod_{i=1}^{n}\lambda e^{-\lambda x_i}=\lambda^n e^{-\lambda\sum_{i=1}^{n}x_i}$$

$$l(\lambda)=\log L(\lambda)=n\log\lambda-\lambda\sum_{i=1}^{n}x_i$$

$$l'(\lambda)=\frac{n}{\lambda}-\sum_{i=1}^{n}x_i=0,\qquad\hat\lambda=\frac{n}{\sum_{i=1}^{n}x_i}$$

With four observations and a total of six, the estimate and second derivative are

$$\hat\lambda=\frac{2}{3}\approx0.6667,\qquad l''(\hat\lambda)=-9<0.$$

The negative second derivative confirms a strict maximum for a positive rate.
""",
    ),
    (
        "O4 - Circulation and magnetic flux",
        r"""Evaluate the circulation around a counterclockwise circle of radius R. Then differentiate a time-dependent magnetic flux through the same area.

$$I=\oint_C(-y\,dx+x\,dy),\qquad x=R\cos\theta,\quad y=R\sin\theta$$

The parameter runs once around the circle. Differentiate the coordinates and substitute.

$$dx=-R\sin\theta\,d\theta,\qquad dy=R\cos\theta\,d\theta$$

$$I=\int_0^{2\pi}R^2(\sin^2\theta+\cos^2\theta)\,d\theta=2\pi R^2$$

For a radius of three halves, the exact answer and its decimal approximation agree.

$$R=\frac{3}{2},\qquad I=\frac{9\pi}{2}\approx14.1372$$

Now let the uniform perpendicular field have magnitude proportional to the square of time. Its flux and induced circulation are

$$B(t)=\kappa t^2,\qquad\Phi_B(t)=\pi R^2\kappa t^2$$

$$\oint_C\vec E\cdot d\vec r=-\frac{d\Phi_B}{dt}=-2\pi R^2\kappa t.$$

For unit radius and coefficient two, the sign follows the chosen positive circulation.

$$\kappa=2,\quad R=1,\qquad\oint_C\vec E\cdot d\vec r=-4\pi t$$
""",
    ),
]


def solution_checks() -> dict:
    xi, eta, k, theta, radius, t, rate = sp.symbols("xi eta k theta R t rate", real=True)
    phi = sp.Rational(3, 2) * (xi**2 + 2 * eta**2)
    psi = sp.sin(sp.pi * xi) * sp.cos(sp.pi * eta)
    u = sp.Matrix([1, sp.I]) / sp.sqrt(2)
    v = sp.Matrix([sp.I, 1]) / sp.sqrt(2)
    facts = {
        "potential_at_point": phi.subs({xi: 1, eta: -1}) == sp.Rational(9, 2),
        "field_at_point": [-sp.diff(phi, x).subs({xi: 1, eta: -1}) for x in [xi, eta]] == [-3, 6],
        "mixed_derivative": sp.simplify(
            sp.diff(psi, xi, eta) + sp.pi**2 * sp.cos(sp.pi * xi) * sp.sin(sp.pi * eta)
        )
        == 0,
        "orthonormal_basis": (u.conjugate().T * u)[0] == 1
        and (v.conjugate().T * v)[0] == 1
        and (u.conjugate().T * v)[0] == 0,
        "determinant": sp.Matrix.hstack(u, v).det() == 1,
        "telescoping_P": sp.product(k / (k + 1), (k, 1, 4)) == sp.Rational(1, 5),
        "telescoping_Q": sp.product(1 + 1 / k, (k, 1, 3)) == 4,
        "likelihood_curvature": sp.diff(4 * sp.log(rate) - 6 * rate, rate, 2).subs(
            rate, sp.Rational(2, 3)
        )
        == -9,
        "circulation": sp.integrate(radius**2, (theta, 0, 2 * sp.pi)) == 2 * sp.pi * radius**2,
        "flux_derivative": sp.diff(sp.pi * radius**2 * k * t**2, t)
        == 2 * sp.pi * radius**2 * k * t,
        "decimal_rounding": abs(float(9 * sp.pi / 2) - 14.1372) < 0.00005,
    }
    if not all(facts.values()):
        raise ValueError("Independent symbolic solution checks failed")
    return {
        "facts": facts,
        "scope": "Explicit analytic results checked separately from rendering; not automatic LaTeX equivalence proof",
    }


def render_base(base, source: str, title: str):
    page, renderer, records, _, _ = personal_layout(
        source,
        title,
        base["profile"],
        base["cfg"],
        base["resolver"],
        base["sessions"],
        base["polish"],
        Background(),
        None,
        cpu=True,
        clearance=True,
    )
    records = MathLatinFeedback(base["profile"].font).apply(page, renderer, records)
    records = base["bars"].apply(page, renderer, records)
    records = base["historical"].apply(page, renderer, records, base["profile"])
    records = base["round2_digits"].apply(page, renderer, records)
    records = FSizeFeedback("D-common-source", source=source, title=title).apply(
        page, renderer, records
    )
    records = base["round2_rules"].apply(page, renderer, records)
    records = FVariationFeedback(base["f_sources"], source=source, title=title, seed=1909).apply(
        page, renderer, records
    )
    return page, renderer, json_value(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    base = load_baseline()
    run = Run.create(
        "session-o-fresh-four-homework-pages",
        seed=1909,
        config={
            "baseline": str(base["path"]),
            "baseline_sha256": base["run_sha256"],
            "bank": str(args.bank.resolve()),
            "bank_sha256": sha256(args.bank / "sources.json"),
            "render": base["cfg"].model_dump(mode="json"),
            "new_neural_updates": 0,
            "fresh_holdout": False,
            "purpose": "supplementary new content and structure engineering coverage",
        },
    )
    print(run.path, flush=True)
    try:
        snapshot_sources(run, {"base": base["path"]})
        shutil.copytree(args.bank, run.path / "solutions/session-o-glyphs")
        shutil.copytree(base["path"] / "solutions", run.path / "solutions/baseline-assets")
        atomic_json(run.path / "reports/solution-verification.json", solution_checks())
        cases, all_rows, panels = [], [], []
        for number, (title, source) in enumerate(CASES, 1):
            print(number, title, flush=True)
            (run.path / f"solutions/{number:02}.md").write_text(source)
            page, renderer, original = render_base(base, source, title)
            paths = page_geometry(page, renderer)
            before_parser = json_value(trace(renderer))
            before = save_pdf(
                renderer.to_vector_pdf([page]), run.path / f"comparisons/A/{number:02}.pdf"
            )
            resolver = SessionOResolver(
                run.path / "solutions/session-o-glyphs", targets=FRESH_TARGETS
            )
            records = json_value(resolver.apply(page, renderer, original, case_id=title))
            after = save_pdf(
                renderer.to_vector_pdf([page]), run.path / f"rendered/pages/{number:02}.pdf"
            )
            checks = validate_increment(
                paths,
                page,
                renderer,
                original,
                records,
                before,
                after,
                targets=FRESH_TARGETS,
                parser_before=before_parser,
            )
            checks["source_placement"] = verify_source_placements(
                run.path / "solutions/session-o-glyphs", original, records, paths, page, renderer
            )
            boxes = np.asarray([p.get_extents().extents for p in page_geometry(page, renderer)])
            if (
                boxes[:, :2].min() < 3
                or boxes[:, 2].max() > after.width - 3
                or boxes[:, 3].max() > after.height - 3
            ):
                raise ValueError("An accepted page contains ink beyond its boundaries")
            atomic_json(
                run.path / f"reports/{number:02}-provenance.json",
                {
                    "baseline": original,
                    "candidate": records,
                    "parser": before_parser,
                    "decisions": resolver.decisions,
                    "checks": checks,
                },
            )
            paired_panel(
                before,
                after,
                run.path / f"comparisons/pages/{number:02}.png",
                title,
                ("Frozen source pipeline", "Reviewed Session O ink"),
            )
            panels.extend(
                group_crops(
                    original,
                    before_parser,
                    before,
                    records,
                    before_parser,
                    after,
                    run.path / f"comparisons/formulas/{number:02}",
                    number,
                    ("Frozen source pipeline", "Reviewed Session O ink"),
                )
            )
            cases.append(
                {
                    "page": number,
                    "title": title,
                    "checks": checks,
                    "fallback": fallback_summary(records),
                }
            )
            all_rows.extend({"page": number, **r} for r in records)
        for name, folder in [
            ("rendered/homework_solutions.pdf", "rendered/pages"),
            ("comparisons/baseline.pdf", "comparisons/A"),
        ]:
            with PdfWriter() as writer:
                for number in range(1, 5):
                    writer.append(run.path / f"{folder}/{number:02}.pdf")
                writer.write(run.path / name)
        local_comparison(
            run.path / "comparisons/formulas",
            panels,
            labels=("Frozen source pipeline", "Reviewed Session O ink"),
        )
        summary = {
            "pages": 4,
            "changed_items": sum(len(c["checks"]["changed_items"]) for c in cases),
            "font_fallbacks": dict(
                Counter(
                    r["semantic"]
                    for r in all_rows
                    if r["context"] == "mathematics"
                    and r["source_type"] in {"external-font", "inherited-font"}
                )
            ),
            "math_visible_items": sum(r["context"] == "mathematics" for r in all_rows),
            "new_neural_updates": 0,
            "author_acceptance": False,
            "visual_review": "pending independent review of immutable output",
        }
        atomic_json(run.path / "reports/cases.json", cases)
        atomic_json(run.path / "reports/glyph-provenance.json", all_rows)
        atomic_json(run.path / "reports/summary.json", summary)
        run.finish("held", result=summary)
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
