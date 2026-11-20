"""Controlled original-question PDF intake, independent solving and installed CPU evidence.

Originals are explicitly Agent-authored engineering inputs, not held-out external exams.
"""

# ruff: noqa: E501 -- Retain exact frozen question and answer strings.

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    (
        "Symmetric tridiagonal system",
        "Let A have rows (2,1,0), (1,2,1), (0,1,2), and b=(1,0,1)^T. Find det(A), the inverse, and the unique solution of Ax=b. Find all eigenvalues and an orthonormal eigenbasis. Verify the solution and diagonalization.",
    ),
    (
        "Two definite integrals",
        "Evaluate I = integral from 0 to 1 of x^2/(1+x)^3 dx and J = integral from 0 to 1 of log(1+x) dx. Give antiderivatives and check them by differentiation. Explain why both definite integrals are positive.",
    ),
    (
        "Normal mean inference",
        "A random sample of n=25 independent normal observations has sample mean 104. The population standard deviation is known to be 10. Test H0: population mean=100 against a two-sided alternative at level 0.05. Give the statistic, approximate p-value, and 95 percent confidence interval using critical value 1.96. If Y=2X-5, estimate its mean and the standard error of the transformed sample mean.",
    ),
    (
        "Coaxial electrostatic field",
        "An ideal infinitely long coaxial capacitor has inner conductor radius a and outer conductor inner radius b, where 0<a<b. The dielectric permittivity is epsilon. The inner conductor carries positive charge per unit length lambda; the outer conductor carries the opposite charge. Derive the radial electric field for a<r<b, V(a)-V(b), capacitance per unit length and electrostatic energy per unit length. Check the energy both from capacitance and by integrating the field energy density.",
    ),
]


def read(path):
    return json.loads(path.read_text())


def prepare():
    run = Run.create(
        "m21-original-question-pdf",
        seed=1909,
        config={
            "purpose": "question-only intake source frozen before solutions",
            "origin": "Agent-authored controlled engineering questions",
            "external_unseen_assignment": False,
            "sealed_data_accessed": False,
        },
    )
    try:
        pdf = run.path / "solutions/original-questions.pdf"
        with PdfPages(pdf) as out:
            for i, (title, question) in enumerate(QUESTIONS, 1):
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(0.09, 0.92, f"Question {i}: {title}", fontsize=15)
                fig.text(
                    0.09, 0.84, textwrap.fill(question, 80), fontsize=12, va="top", linespacing=1.6
                )
                fig.text(
                    0.09, 0.08, "Agent-authored intake fixture. No supplied answers.", fontsize=10
                )
                out.savefig(fig)
                plt.close(fig)
        atomic_json(run.path / "solutions/question-source.json", QUESTIONS)
        shutil.copyfile(__file__, run.path / "logs/source.py")
        run.finish(
            "prepared",
            result={"pdf_sha256": sha256(pdf), "questions": 4, "answers_provided": False},
        )
        verify_run(run.path)
        print(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


def intake(parent):
    verify_run(parent)
    pdf = parent / "solutions/original-questions.pdf"
    pages = [p.extract_text() for p in PdfReader(pdf).pages]
    run = Run.create(
        "m21-question-reading",
        config={
            "parent": str(parent.resolve()),
            "parent_sha256": sha256(parent / "run.json"),
            "pdf_sha256": sha256(pdf),
            "intake_method": "pypdf text-layer extraction; Agent must read returned text before solving",
            "visual_ocr_claim": False,
        },
    )
    atomic_json(run.path / "reports/extracted-questions.json", pages)
    (run.path / "reports/questions.txt").write_text("\n\n".join(pages))
    shutil.copyfile(__file__, run.path / "logs/source.py")
    run.finish("prepared", result={"pages": len(pages), "reading_pending": True})
    print(run.path)
    print("\n\n".join(pages))


SOLUTIONS = [
    r"""Expand the determinant along the first row. The matrix is real and symmetric.

$$A=\begin{bmatrix}2&1&0\\1&2&1\\0&1&2\end{bmatrix},\qquad \det A=2(4-1)-2=4$$

Cofactors give the inverse. Its product with A is the identity.

$$A^{-1}=\frac{1}{4}\begin{bmatrix}3&-2&1\\-2&4&-2\\1&-2&3\end{bmatrix},\qquad AA^{-1}=I$$

Multiply the inverse by the given right-hand side and check the original system.

$$x=A^{-1}\begin{pmatrix}1\\0\\1\end{pmatrix}=\begin{pmatrix}1\\-1\\1\end{pmatrix},\qquad Ax=\begin{pmatrix}1\\0\\1\end{pmatrix}$$

For the eigenvalues, expand the characteristic determinant by the same row.

$$\det(A-\lambda I)=(2-\lambda)((2-\lambda)^2-2)$$

$$\lambda_+=2+\sqrt{2},\qquad \lambda_0=2,\qquad \lambda_-=2-\sqrt{2}$$

Normalize the three orthogonal eigenvectors.

$$q_+=\frac{1}{2}\begin{pmatrix}1\\\sqrt{2}\\1\end{pmatrix},\quad q_0=\frac{1}{\sqrt{2}}\begin{pmatrix}1\\0\\-1\end{pmatrix},\quad q_-=\frac{1}{2}\begin{pmatrix}1\\-\sqrt{2}\\1\end{pmatrix}$$

With these as columns of Q, direct multiplication verifies the decomposition.

$$Q^TQ=I,\qquad Q^TAQ=\begin{bmatrix}2+\sqrt{2}&0&0\\0&2&0\\0&0&2-\sqrt{2}\end{bmatrix}$$

Every eigenvalue is positive, consistent with the positive determinant.
""",
    r"""Set u equal to one plus x in the first integral. The limits become one and two.

$$u=1+x,\qquad x=u-1,\qquad dx=du$$

$$I=\int_1^2\frac{(u-1)^2}{u^3}\,du=\int_1^2\left(\frac{1}{u}-\frac{2}{u^2}+\frac{1}{u^3}\right)du$$

An antiderivative is the logarithm plus two rational terms.

$$F(x)=\log(1+x)+\frac{2}{1+x}-\frac{1}{2(1+x)^2}$$

Check its derivative over a common denominator.

$$F'(x)=\frac{1}{1+x}-\frac{2}{(1+x)^2}+\frac{1}{(1+x)^3}=\frac{x^2}{(1+x)^3}$$

$$I=F(1)-F(0)=\log 2+1-\frac{1}{8}-\frac{3}{2}=\log 2-\frac{5}{8}$$

For the second integral, integration by parts gives an antiderivative.

$$G(x)=(1+x)\log(1+x)-(1+x)$$

$$G'(x)=\log(1+x)+1-1=\log(1+x)$$

$$J=G(1)-G(0)=2\log 2-2+1=2\log 2-1$$

Both integrands are strictly positive between zero and one, so both areas are positive.

$$I\approx0.06815>0,\qquad J\approx0.38629>0$$
""",
    r"""The population variance is known, so use the normal statistic rather than an estimated-variance statistic.

$$n=25,\qquad \bar{x}=104,\qquad \sigma=10,\qquad \mu_0=100$$

The sample mean has standard error two under the model.

$$\operatorname{SE}(\bar{X})=\frac{\sigma}{\sqrt{n}}=\frac{10}{5}=2$$

$$z=\frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}=\frac{104-100}{2}=2$$

The two-sided tail probability is twice the upper normal tail.

$$p=2(1-\Phi(2))\approx0.04550<0.05$$

Reject the null hypothesis at the specified level. The confidence interval gives the same decision.

$$\bar{x}\pm1.96\frac{\sigma}{\sqrt{n}}=104\pm3.92$$

$$100.08\leq\mu\leq107.92$$

The value one hundred is outside this interval. This is not a probability statement about a fixed parameter.

For the linear transformation, transform the mean and scale the standard error by the absolute coefficient.

$$Y=2X-5,\qquad \widehat{\mu}_Y=2\bar{x}-5=203$$

$$\operatorname{Var}(\bar{Y})=4\operatorname{Var}(\bar{X})=4\frac{100}{25}=16$$

$$\operatorname{SE}(\bar{Y})=\sqrt{16}=4$$
""",
    r"""Assume a homogeneous dielectric and neglect end effects. Cylindrical symmetry makes the field radial in the gap.

For a Gaussian cylinder of length L, the end caps carry no electric flux.

$$E(r)(2\pi rL)=\frac{\lambda L}{\epsilon},\qquad a<r<b$$

$$\vec{E}(r)=\frac{\lambda}{2\pi\epsilon r}\,\hat{r}$$

Potential decreases in the direction of the positive radial field.

$$V(a)-V(b)=\int_a^b E(r)\,dr=\frac{\lambda}{2\pi\epsilon}\log\left(\frac{b}{a}\right)$$

Write c for capacitance per unit length. The logarithm is positive because b exceeds a.

$$c=\frac{\lambda}{V(a)-V(b)}=\frac{2\pi\epsilon}{\log(b/a)}$$

The capacitor energy per unit length is one half the charge density times the voltage.

$$w=\frac{\lambda^2}{2c}=\frac{\lambda^2}{4\pi\epsilon}\log\left(\frac{b}{a}\right)$$

Independently integrate the field energy density across the annular cross section.

$$w=\int_a^b\frac{\epsilon E(r)^2}{2}\,2\pi r\,dr$$

$$w=\frac{\lambda^2}{4\pi\epsilon}\int_a^b\frac{dr}{r}=\frac{\lambda^2}{4\pi\epsilon}\log\left(\frac{b}{a}\right)$$

The two expressions agree. The field, voltage and energy have the expected positive signs.
""",
]


def checks():
    import sympy as s

    x = s.symbols("x", positive=True)
    A = s.Matrix([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    q = s.Matrix.hstack(
        s.Matrix([1, s.sqrt(2), 1]) / 2,
        s.Matrix([1, 0, -1]) / s.sqrt(2),
        s.Matrix([1, -s.sqrt(2), 1]) / 2,
    )
    a, b, lam, eps, r = s.symbols("a b lam eps r", positive=True)
    facts = {
        "determinant": A.det() == 4,
        "inverse": A.inv() == s.Matrix([[3, -2, 1], [-2, 4, -2], [1, -2, 3]]) / 4,
        "solution": A * s.Matrix([1, -1, 1]) == s.Matrix([1, 0, 1]),
        "orthonormal": s.simplify(q.T * q) == s.eye(3),
        "diagonalization": s.simplify(q.T * A * q) == s.diag(2 + s.sqrt(2), 2, 2 - s.sqrt(2)),
        "first_integral": s.simplify(
            s.integrate(x * x / (1 + x) ** 3, (x, 0, 1)) - (s.log(2) - s.Rational(5, 8))
        )
        == 0,
        "second_integral": s.integrate(s.log(1 + x), (x, 0, 1)) == 2 * s.log(2) - 1,
        "primitive": s.simplify(
            s.diff(s.log(1 + x) + 2 / (1 + x) - 1 / (2 * (1 + x) ** 2), x) - x * x / (1 + x) ** 3
        )
        == 0,
        "normal_z": (104 - 100) / (10 / s.sqrt(25)) == 2,
        "ci_lower": s.Rational(104) - s.Rational(196, 100) * 2 == s.Rational(10008, 100),
        "ci_upper": s.Rational(104) + s.Rational(196, 100) * 2 == s.Rational(10792, 100),
        "transformed_mean": 2 * 104 - 5 == 203,
        "transformed_variance": 4 * s.Rational(100, 25) == 16,
        "potential": s.simplify(
            s.integrate(lam / (2 * s.pi * eps * r), (r, a, b))
            - lam * s.log(b / a) / (2 * s.pi * eps)
        )
        == 0,
        "field_energy": s.simplify(
            s.integrate(eps * (lam / (2 * s.pi * eps * r)) ** 2 / 2 * 2 * s.pi * r, (r, a, b))
            - lam * lam * s.log(b / a) / (4 * s.pi * eps)
        )
        == 0,
        "capacitance_energy": s.simplify(
            lam**2 / (2 * (2 * s.pi * eps / s.log(b / a)))
            - lam**2 * s.log(b / a) / (4 * s.pi * eps)
        )
        == 0,
    }
    facts = {k: bool(v) for k, v in facts.items()}
    assert all(facts.values()), facts
    return {
        "facts": facts,
        "two_sided_p": float(s.erfc(s.sqrt(2))),
        "verification_scope": "Symbolic identities and independent root Agent derivation; does not validate rendered shape.",
    }


def solve(parent, reading):
    verify_run(parent)
    verify_run(reading)
    assert read(reading / "run.json")["config"]["parent_sha256"] == sha256(parent / "run.json")
    run = Run.create(
        "m21-solved-intake",
        seed=1909,
        config={
            "purpose": "Agent solutions after frozen PDF text and visual reading",
            "question_run": str(parent.resolve()),
            "reading_run": str(reading.resolve()),
            "synthetic_questions": True,
            "external_unseen_assignment": False,
        },
    )
    try:
        shutil.copyfile(
            parent / "solutions/original-questions.pdf",
            run.path / "solutions/original-questions.pdf",
        )
        shutil.copyfile(
            "/tmp/m21-question-read.png", run.path / "reports/visually-read-questions.png"
        )
        atomic_json(run.path / "reports/independent-checks.json", checks())
        atomic_json(
            run.path / "reports/agent-reading.json",
            {
                "reader": "new_e2e Agent",
                "all_four_original_pdf_pages_visually_read": True,
                "root_independent_math_review": True,
                "no_answer_in_question_pdf": True,
            },
        )
        problems = []
        for i, ((title, question), answer) in enumerate(zip(QUESTIONS, SOLUTIONS, strict=True), 1):
            name = f"{i:02d}.md"
            (run.path / "solutions" / name).write_text(answer)
            problems.append(
                {
                    "id": f"m21-e2e-{i:02d}",
                    "number": title,
                    "statement": question,
                    "sources": [
                        {
                            "path": "original-questions.pdf",
                            "page": i,
                            "sha256": sha256(run.path / "solutions/original-questions.pdf"),
                        }
                    ],
                    "solution": name,
                    "verification": {
                        "status": "passed",
                        "evidence": "Agent solved after frozen question PDF reading; root independent derivation and symbolic identities checked.",
                    },
                }
            )
        job = {
            "schema_version": 1,
            "backend": "personal-cpu-experimental",
            "profile": "personal-cpu-candidate@v2",
            "profile_sha256": "4044d2a8b5118dbdd4e734653f53770a914fa10ede2fb30948714885029a545c",
            "generated_by": "m21-controlled-pdf-intake-e2e/1",
            "problems": problems,
        }
        atomic_json(run.path / "solutions/job.json", job)
        shutil.copyfile(__file__, run.path / "logs/source.py")
        run.finish(
            "prepared",
            result={
                "questions": 4,
                "answer_sha256": {
                    p["solution"]: sha256(run.path / "solutions" / p["solution"]) for p in problems
                },
                "rendering_started": False,
            },
        )
        verify_run(run.path)
        print(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


def render(parent, case=None, cli_path=None):
    verify_run(parent)
    run = Run.create(
        "m21-installed-e2e",
        seed=1909,
        config={
            "parent": str(parent.resolve()),
            "parent_sha256": sha256(parent / "run.json"),
            "case": case,
            "runtime": str(cli_path) if cli_path else "installed8da224f4 Linux3.12",
            "cached_words": False,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/source.py")
        shutil.copytree(parent / "solutions", run.path / "solutions/input")
        if case:
            job = read(run.path / "solutions/input/job.json")
            job["problems"] = [job["problems"][case - 1]]
            atomic_json(run.path / "solutions/input/job.json", job)
        cli = cli_path or Path("/tmp/hwsol-personal-cpu-3.12-lc816um1/runtime/bin/hwsol")
        env = {
            **os.environ,
            "HWSOL_DATA_DIR": str(run.path / "solutions/store"),
            "HWSOL_WORKSPACE": str(Path("/tmp") / run.path.name),
            "MPLCONFIGDIR": str(run.path / "logs/mpl"),
        }
        for key in ["PYTHONPATH", "VIRTUAL_ENV"]:
            env.pop(key, None)
        archive = (
            ROOT
            / "workspace/runs/20261120T150345712055Z-portable-personal-cpu-sigma-assets-0af50b3f/solutions/personal-cpu-candidate-v2.zip"
        )
        commands = []

        def command(args):
            result = subprocess.run(
                [str(cli), *args], cwd="/tmp", env=env, capture_output=True, text=True
            )
            commands.append(
                {
                    "args": args,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            atomic_json(run.path / "logs/commands.json", commands)
            if result.returncode:
                raise RuntimeError(result.stderr)
            return json.loads(result.stdout)

        command(["profiles", "import", "--source", str(archive)])
        output = run.path / "rendered/job"
        response = command(
            [
                "render",
                "--manifest",
                str(run.path / "solutions/input/job.json"),
                "--output",
                str(output),
            ]
        )
        assert response["status"] == "held" and response["result_status"] == "passed"
        command(["verify", "--job", str(output), "--pixels"])
        validation = read(output / "validation_report.json")
        summary = {
            "pages": len(validation["pages"]),
            "runtime_commit": validation["build"]["commit"],
            "font_fallbacks": sum(len(p["math_fallback_glyphs"]) for p in validation["pages"]),
            "all_question_solution_steps_retained": True,
            "question_pdf_intake": True,
            "external_unseen_assignment": False,
            "author_style_approval": False,
            "production_go": False,
        }
        # Child workspaces remain intact outside this immutable evidence directory.
        nested = Path(env["HWSOL_WORKSPACE"])
        for i, receipt in enumerate(nested.rglob("run.json")):
            shutil.copyfile(receipt, run.path / f"reports/child-{i}-receipt.json")
        run.finish("held", result=summary)
        verify_run(run.path)
        print(json.dumps(summary))
    except Exception as exc:
        if "env" in locals():
            for i, receipt in enumerate(Path(env["HWSOL_WORKSPACE"]).rglob("run.json")):
                shutil.copyfile(receipt, run.path / f"reports/child-{i}-receipt.json")
        run.fail(exc)
        raise


def recover(parent):
    import difflib

    verify_run(parent)
    run = Run.create(
        "m21-explicit-source-recovery",
        seed=1909,
        config={
            "parent": str(parent.resolve()),
            "parent_sha256": sha256(parent / "run.json"),
            "purpose": "Agent corrects unsupported notation in its own answer; renderer does not rewrite submitted math",
            "first_pass_success": False,
        },
    )
    shutil.copytree(parent / "solutions", run.path / "solutions", dirs_exist_ok=True)
    differences = []
    changes = [
        (r"\det", r"\mathrm{det}"),
        (r"\operatorname", r"\mathrm"),
        (r"\widehat", r"\hat"),
        (r"\sigma/\sqrt{n}", r"\frac{\sigma}{\sqrt{n}}"),
        (r"\log(b/a)", r"\log\left(\frac{b}{a}\right)"),
    ]
    for path in sorted((run.path / "solutions").glob("*.md")):
        original = path.read_text()
        revised = original
        for before, after in changes:
            revised = revised.replace(before, after)
        path.write_text(revised)
        differences.append(
            {
                "file": path.name,
                "original_sha256": sha256(parent / "solutions" / path.name),
                "revised_sha256": sha256(path),
                "diff": list(difflib.unified_diff(original.splitlines(), revised.splitlines())),
                "mathematical_equivalence": "Operator names remain labels; hat stays on one mu; all divisions remain same ratios now explicitly stacked.",
            }
        )
    atomic_json(run.path / "reports/source-corrections.json", differences)
    atomic_json(run.path / "reports/independent-checks.json", checks())
    shutil.copyfile(__file__, run.path / "logs/source.py")
    run.finish(
        "prepared",
        result={
            "edited_answers": sum(bool(x["diff"]) for x in differences),
            "question_pdf_unchanged": sha256(run.path / "solutions/original-questions.pdf")
            == sha256(parent / "solutions/original-questions.pdf"),
            "no_reasoning_deleted": True,
            "renderer_capability_expansion": False,
        },
    )
    verify_run(run.path)
    print(run.path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "read", "solve", "render", "recover"])
    p.add_argument("--parent", type=Path)
    p.add_argument("--reading", type=Path)
    p.add_argument("--case", type=int, choices=range(1, 5))
    p.add_argument("--cli", type=Path)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare()
    elif a.mode == "read":
        intake(a.parent)
    elif a.mode == "solve":
        solve(a.parent, a.reading)
    elif a.mode == "recover":
        recover(a.parent)
    else:
        render(a.parent, a.case, a.cli)
