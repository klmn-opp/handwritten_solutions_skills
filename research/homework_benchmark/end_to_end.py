"""Agent solutions to four original text inputs, with independent local checks.

This reproducible fixture records an actual Agent solve; it is not an autonomous
solver implementation, handwriting training data, or an image-recognition test.
"""

import json
import shutil
import sys
from pathlib import Path

import sympy as s

from hwsol.jobs.runtime import render_job, verify_job
from hwsol.jobs.schema import Job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, profile_store, verify_run, workspace_root
from research.homework_benchmark.cases import end_to_end_inputs
from research.homework_benchmark.render import launch

ANSWERS = [
    r"""(a) The determinant is $2(4)-(-1)(3)=11$, so the inverse exists.

$$
B^{-1}=\frac{1}{11}\begin{pmatrix}4&1\\-3&2\end{pmatrix}
$$

(b) Multiply the inverse by the given vector.

$$
x=B^{-1}c=\frac{1}{11}\begin{pmatrix}21\\-13\end{pmatrix}
$$

(c) Substitution into both original equations gives

$$
2\frac{21}{11}-\left(-\frac{13}{11}\right)=5
$$

$$
3\frac{21}{11}+4\left(-\frac{13}{11}\right)=1.
$$
""",
    r"""(a) Set $u=\sin x$, so $du=\cos x\,dx$.
At $x=0$, $u=0$; at $x=\pi/2$, $u=1$.

$$
\int_0^{\pi/2}\sin^3x\cos x\,dx=\int_0^1u^3\,du=\frac{1}{4}.
$$

(b) An antiderivative is

$$
F(x)=\frac{\sin^4x}{4}+C.
$$

Differentiating checks the substitution:

$$
F'(x)=\sin^3x\cos x.
$$
""",
    r"""The population standard deviation is unknown. Normality and independence
justify a one-sample t test with $n-1=8$ degrees of freedom.

$$
H_0:\mu=12,\qquad H_1:\mu\ne12.
$$

The estimated standard error is $3/\sqrt{9}=1$.

$$
t=\frac{14-12}{3/\sqrt{9}}=2.
$$

Since $|2|<2.306$, do not reject the null hypothesis at the 5 percent level.
The data do not establish a difference from 12 at that level.

Using the supplied critical value, the 95 percent confidence interval is

$$
14\pm2.306(1)=[11.694,16.306].
$$

The interval contains 12, in agreement with the test decision.
""",
    r"""(a) Convert the measurements to metres:

$$
a=3.00\times10^{-4}\,\mathrm{m},\quad\lambda=6.00\times10^{-7}\,\mathrm{m},\quad L=1.50\,\mathrm{m}.
$$

The first minimum satisfies

$$
\sin\theta=\frac{\lambda}{a}=0.002.
$$

Taking the inverse sine gives $\theta\approx0.0020000013$ radians,
or approximately $0.114592$ degrees.

(b) The two first minima lie on opposite sides of the central maximum.
Since the angle is very small, $\tan\theta\approx\sin\theta\approx\theta$.

$$
W\approx2L\frac{\lambda}{a}=0.00600\,\mathrm{m}=6.00\,\mathrm{mm}.
$$

The exact screen width is $2L\tan\theta\approx0.006000012$ metres.
The relative correction is about $0.0002$ percent, so the approximation
is appropriate for the given precision.
""",
]


def verify_answers() -> list[dict]:
    """Check by a separate computational route from the prose derivations."""
    matrix = s.Matrix([[2, -1], [3, 4]])
    vector = s.Matrix([5, 1])
    inverse = s.Matrix([[4, 1], [-3, 2]]) / 11
    answer = s.Matrix([21, -13]) / 11
    x = s.symbols("x", real=True)
    derivative = s.diff(s.sin(x) ** 4 / 4, x)
    se = s.Rational(3) / s.sqrt(9)
    statistic = (14 - 12) / se
    halfwidth = s.Rational(2306, 1000) * se
    ratio = s.Rational(600, 10**9) / s.Rational(30, 10**5)
    angle = s.asin(ratio)
    exact_width = 3 * s.tan(angle)
    checks = [
        ("inverse identity", matrix * inverse == s.eye(2)),
        ("independent matrix solution", matrix.inv(method="DM") * vector == answer),
        ("both original equations", matrix * answer == vector),
        ("derivative", s.simplify(derivative - s.sin(x) ** 3 * s.cos(x)) == 0),
        (
            "direct integration",
            s.integrate(s.sin(x) ** 3 * s.cos(x), (x, 0, s.pi / 2)) == s.Rational(1, 4),
        ),
        ("integration limits", s.sin(0) == 0 and s.sin(s.pi / 2) == 1),
        ("test statistic", statistic == 2),
        ("test decision", abs(statistic) < s.Rational(2306, 1000)),
        (
            "CI endpoints",
            14 - halfwidth == s.Rational(11694, 1000) and 14 + halfwidth == s.Rational(16306, 1000),
        ),
        ("unit ratio", ratio == s.Rational(2, 1000)),
        ("small-angle width", 3 * ratio == s.Rational(6, 1000)),
        ("exact angle", abs(float(angle) - 0.002000001333335733) < 1e-16),
        ("approximation error", 0 < (float(exact_width) / 0.006 - 1) < 0.00000201),
    ]
    result = [{"check": label, "passed": bool(ok)} for label, ok in checks]
    if not all(row["passed"] for row in result):
        raise ValueError(result)
    return result


def execute() -> Path:
    root = workspace_root()
    run = Run.create(
        "m1-text-end-to-end",
        workspace=root,
        seed=1909,
        config={
            "input_kind": "new Agent-authored original text without reference answers",
            "solve_author": "root Codex Agent; separate from question author",
            "image_pdf_recognition_evaluated": False,
            "new_neural_updates": 0,
            "content_revision": 1,
            "previous_failure": "20261120T103527461026Z-m1-text-end-to-end-e6103b9e",
            "revision_reason": (
                "join a display equation onto one LaTeX source line; semantic content unchanged"
            ),
            "source_sha256": sha256(Path(__file__)),
            "command": "python -m research.homework_benchmark.end_to_end",
        },
    )
    print(run.path, flush=True)
    try:
        protected = {str(root / "latest.json"): sha256(root / "latest.json")}
        store = profile_store(root)
        profile = store.resolve("personal@v1")
        protected[str(profile.directory / "profile.json")] = sha256(
            profile.directory / "profile.json"
        )
        shutil.copyfile(__file__, run.path / "logs/agent-solve-source.py")
        source = Path(__file__).with_name("cases.py")
        shutil.copyfile(source, run.path / "logs/question-source.py")
        statements = end_to_end_inputs()
        atomic_json(run.path / "solutions/original-inputs.json", statements)
        checks = verify_answers()
        atomic_json(run.path / "reports/independent-checks.json", checks)
        evidence = [
            "Inverse product, exact rational solution by symbolic matrix inversion, "
            "and both original equations checked.",
            "Direct symbolic definite integration, derivative of proposed antiderivative, "
            "and limits checked.",
            "Exact t statistic and supplied critical-value comparison, interval endpoints checked; "
            "normal/independent assumptions read from original.",
            "SI conversion and exact inverse sine/tangent compared with small-angle width; "
            "numerical relative error checked.",
        ]
        problems = []
        for i, (item, answer) in enumerate(zip(statements, ANSWERS, strict=True), 1):
            (run.path / f"solutions/source-{i}.txt").write_text(item["statement"] + "\n")
            (run.path / f"solutions/answer-{i}.md").write_text(answer)
            problems.append(
                {
                    "id": item["id"],
                    "number": str(i),
                    "statement": item["statement"],
                    "sources": [{"path": f"source-{i}.txt", "location": item["id"]}],
                    "solution": f"answer-{i}.md",
                    "verification": {"status": "passed", "evidence": evidence[i - 1]},
                }
            )
        job = Job.model_validate(
            {
                "schema_version": 1,
                "generated_by": "OpenAI Codex, independent solutions to four new text problems",
                "profile": profile.manifest.ref,
                "profile_sha256": profile.manifest_sha256,
                "backend": "font-cpu",
                "render": {
                    "dpi": 300,
                    "seed": 1909,
                    "glyph_metrics": "raw",
                    "semantic_layout": True,
                    "math_strokes": "personal",
                },
                "problems": problems,
            }
        )
        manifest = run.path / "solutions/job.json"
        atomic_json(manifest, job.model_dump(mode="json", exclude_unset=True))
        production_output = run.path / "comparisons/production-job"
        render_job(manifest, production_output, store)
        production_check = verify_job(production_output, pixels=True)
        candidate = launch(manifest=manifest)
        verify_run(candidate)
        candidate_result = json.loads((candidate / "run.json").read_text())["result"]
        shutil.copyfile(
            production_output / "homework_solutions.pdf",
            run.path / "rendered/homework_solutions.pdf",
        )
        for p in (production_output / "pages").glob("*.png"):
            shutil.copyfile(p, run.path / "rendered/pages" / p.name)
        for path, digest in protected.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Protected original changed: {path}")
        atomic_json(run.path / "reports/protected-assets.json", protected)
        atomic_json(
            run.path / "reports/candidate-comparison.json",
            {"run": str(candidate), "result": candidate_result},
        )
        run.finish(
            "held",
            result={
                "production_runtime": production_check,
                "candidate_run": str(candidate),
                "independent_answer_checks": "passed",
                "visual_review": "pending separate Agent review",
                "input_scope": "text only, Agent-authored fixtures",
                "torch_loaded": "torch" in sys.modules,
                "production_unchanged": True,
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    execute()
