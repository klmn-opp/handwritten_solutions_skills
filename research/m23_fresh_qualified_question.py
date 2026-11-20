"""Solve and render one frozen controlled text question via installed qualified CPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import sympy as s

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run

ANSWER = r"""The density is nonnegative on its support. Its total mass is one.

$$\int_0^1 3x^2\,dx=1$$

Integrating the first two powers gives the raw moments.

$$\mathrm{E}[X]=\int_0^1 3x^3\,dx=\frac{3}{4}$$

$$\mathrm{E}[X^2]=\int_0^1 3x^4\,dx=\frac{3}{5}$$

Subtract the squared mean to get the variance and take its positive root.

$$\mathrm{Var}(X)=\frac{3}{5}-\left(\frac{3}{4}\right)^2=\frac{3}{80}$$

$$\sigma_X=\sqrt{\frac{3}{80}}=\frac{\sqrt{15}}{20}$$

A linear change shifts the mean and scales the variance by the squared slope.

$$\mathrm{E}[Y]=1-2\left(\frac{3}{4}\right)=-\frac{1}{2}$$

$$\mathrm{Var}(Y)=4\mathrm{Var}(X)=\frac{3}{20}$$

The event reverses the inequality because the slope is negative.

$$\mathrm{P}(Y>0)=\mathrm{P}\left(X<\frac{1}{2}\right)=\int_0^{\frac{1}{2}}3x^2\,dx=\frac{1}{8}$$

For an independent check, the inverse is $x=\frac{1-y}{2}$ with absolute derivative $\frac{1}{2}$.

$$g_Y(y)=\frac{3}{8}(1-y)^2,\qquad -1<y<1$$

Integrating this transformed density over the positive part confirms the result.

$$\int_0^1 g_Y(y)\,dy=\frac{1}{8}$$
"""


def read(p):
    return json.loads(p.read_text())


def execute(run_path: Path, linux: Path, lifecycle: Path):
    run = Run(run_path)
    if run.record()["status"] != "running":
        raise ValueError("Use the fresh frozen question run, never overwrite a completed run")
    verify_run(linux)
    verify_run(lifecycle)
    question = run.path / "solutions/question.txt"
    if sha256(question) != read(run.path / "reports/frozen-question.json")["sha256"]:
        raise ValueError("Original question changed")
    shutil.copyfile(__file__, run.path / "logs/m23_fresh_qualified_question.py")
    try:
        x, y = s.symbols("x y", real=True)
        half = s.Rational(1, 2)
        density = 3 * x**2
        mean = s.integrate(x * density, (x, 0, 1))
        second = s.integrate(x**2 * density, (x, 0, 1))
        variance = second - mean**2
        transformed = density.subs(x, (1 - y) / 2) / 2
        ymean = s.integrate(y * transformed, (y, -1, 1))
        ysecond = s.integrate(y**2 * transformed, (y, -1, 1))
        facts = {
            "mass": s.integrate(density, (x, 0, 1)) == 1,
            "mean": mean == s.Rational(3, 4),
            "second": second == s.Rational(3, 5),
            "variance": variance == s.Rational(3, 80),
            "standard_deviation": s.simplify(s.sqrt(variance) - s.sqrt(15) / 20) == 0,
            "transformed_mass": s.integrate(transformed, (y, -1, 1)) == 1,
            "transformed_mean": ymean == 1 - 2 * mean == -half,
            "transformed_variance": ysecond - ymean**2 == 4 * variance == s.Rational(3, 20),
            "event_both_coordinates": s.integrate(density, (x, 0, half))
            == s.integrate(transformed, (y, 0, 1))
            == s.Rational(1, 8),
        }
        if not all(facts.values()):
            raise ValueError("Independent mathematical identities failed")
        atomic_json(
            run.path / "reports/independent-mathematics.json",
            {
                "facts": facts,
                "question_sha256": sha256(question),
                "method": "Agent derivation; separate direct integrals in X and transformed Y",
                "input_scope": "controlled Agent-authored new text; not an external holdout",
            },
        )
        solution = run.path / "solutions/answer.md"
        solution.write_text(ANSWER)
        qualified = read(lifecycle / "reports/lifecycle.json")["qualification"]
        spec = read(lifecycle / "solutions/input/job.json")["profile"]
        manifest = run.path / "solutions/job.json"
        atomic_json(
            manifest,
            {
                "schema_version": 1,
                "backend": "font-cpu",
                "profile": spec,
                "profile_sha256": qualified["profile_sha256"],
                "generated_by": "m23-fresh-qualified-text-question/1",
                "problems": [
                    {
                        "id": "m23-density-change-01",
                        "number": "A quadratic probability density",
                        "statement": question.read_text().strip(),
                        "sources": [{"path": "question.txt", "sha256": sha256(question)}],
                        "solution": "answer.md",
                        "verification": {
                            "status": "passed",
                            "evidence": "Exact moments and transformed-density check retained",
                        },
                    }
                ],
            },
        )
        state = Path(tempfile.mkdtemp(prefix="hwsol-m23-fresh-question-"))
        shutil.copytree(lifecycle / "reports/store-snapshot", state / "store")
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"HWSOL_DATA_DIR", "HWSOL_WORKSPACE", "PYTHONPATH", "VIRTUAL_ENV"}
        }
        env.update(
            HWSOL_DATA_DIR=str(state / "store"),
            HWSOL_WORKSPACE=str(state / "workspace"),
            MPLCONFIGDIR=str(state / "mpl"),
        )
        runtime = Path(read(linux / "reports/installations.json")[0]["fresh_environment"])
        commands = [
            [
                runtime / "bin/hwsol",
                "render",
                "--manifest",
                manifest,
                "--output",
                run.path / "rendered/homework",
            ],
            [runtime / "bin/hwsol", "verify", "--job", run.path / "rendered/homework", "--pixels"],
        ]
        for i, command in enumerate(commands, 1):
            result = subprocess.run(
                list(map(str, command)), cwd=run.path, env=env, capture_output=True, text=True
            )
            (run.path / f"logs/{i:02d}.txt").write_text(
                result.stdout + "\nSTDERR\n" + result.stderr
            )
            if result.returncode:
                raise ValueError(f"Installed operation failed: logs/{i:02d}.txt")
        receipts = run.path / "logs/cli-receipts"
        receipts.mkdir()
        for child in (state / "workspace/runs").glob("*/run.json"):
            shutil.copyfile(child, receipts / f"{child.parent.name}-receipt.json")
        atomic_json(
            run.path / "reports/execution.json",
            {
                "installed_cli": str(runtime / "bin/hwsol"),
                "linux_install_sha256": sha256(linux / "run.json"),
                "qualification_lifecycle_sha256": sha256(lifecycle / "run.json"),
                "question_sha256": sha256(question),
                "solution_sha256": sha256(solution),
                "source_commit": qualified["build_commit"],
                "visual_review": "pending root inspection",
                "production_activation": False,
                "new_training": False,
            },
        )
        print(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "linux", "lifecycle"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    execute(a.run.resolve(), a.linux.resolve(), a.lifecycle.resolve())
