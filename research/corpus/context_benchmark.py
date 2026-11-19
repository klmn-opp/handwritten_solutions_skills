"""Versioned, backend-independent context cases; no corpus-derived private text in Git."""

import hashlib
import json


def cases() -> list[dict]:
    texts = [
        ("word", "assignment practice solution"),
        ("word", "therefore suppose continuous"),
        ("repeated", "bookkeeper successfully unnecessary"),
        ("long-word", "characteristically incomprehensibility"),
        ("sentence", "The given function is continuous on the closed interval."),
        ("sentence", "Substitute the initial condition and simplify the expression."),
        ("sentence", "The independent columns form a basis for the column space."),
        ("sentence", "We reject the null hypothesis at the stated significance level."),
        ("ambiguity", "A minimum of 10 trials will distinguish l, I, 1, o and 0."),
        ("punctuation", "Check units, signs, and limits. Is the result consistent?"),
        ("repeated", "The little parallel lines illustrate the coefficient pattern."),
        (
            "long-line",
            "After differentiating both sides, collect the common terms "
            "and apply the boundary condition.",
        ),
    ]
    math = [
        ("fraction", r"\frac{x^2+1}{x+1}"),
        ("fraction", r"\frac{1}{1+\frac{1}{x}}"),
        ("integral", r"\int_0^1 x^2\,dx=\frac{1}{3}"),
        ("integral", r"\int x e^x\,dx=(x-1)e^x+C"),
        ("sum", r"\sum_{k=1}^{n}k=\frac{n(n+1)}{2}"),
        ("radical", r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}"),
        ("greek", r"\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2"),
        ("scripts", r"a_{ij}^{(2)}=a_{ij}^{(1)}-l_{i1}a_{1j}"),
        ("matrix", r"A=\begin{bmatrix}1&2\\3&4\end{bmatrix}"),
        ("matrix", r"D=\begin{pmatrix}2&0&0\\0&3&0\\0&0&4\end{pmatrix}"),
        ("vector", r"\vec{F}=q(\vec{E}+\vec{v}\times\vec{B})"),
        ("accent", r"\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i"),
    ]
    sources = [(cat, text) for cat, text in texts]
    sources += [(cat, "$$\n" + tex + "\n$$") for cat, tex in math]
    sources += [
        ("derivation", "$$\nx^2-5x+6=0\n$$\n\n$$\n(x-2)(x-3)=0\n$$\n\n$$\nx=2,3\n$$"),
        ("derivation", "$$\ny=e^{2x}\n$$\n\n$$\ny'=2e^{2x}\n$$\n\n$$\ny''=4e^{2x}\n$$"),
        ("derivation", "$$\nV=IR\n$$\n\n$$\nP=VI=I^2R\n$$"),
        ("limit", "$$\n\\lim_{x\\to0}\\frac{\\sin x}{x}=1\n$$"),
        ("mixed", "The sample mean is $\\bar{x}=12.5$, with $n=20$ observations."),
        (
            "mixed",
            "Compute the determinant explicitly.\n\n$$\n1(4)-2(3)=-2\n$$",
        ),
        (
            "mixed",
            "The current approaches its steady value as time increases.\n\n"
            "$$\ni(t)=\\frac{V}{R}(1-e^{-Rt/L})\n$$",
        ),
        (
            "mixed",
            "The eigenvalues are the roots of the characteristic equation.\n\n"
            "$$\n(2-\\lambda)(3-\\lambda)=0\n$$",
        ),
    ]
    assert len(sources) == 32
    return [
        {
            "id": f"context-v1-{i:03d}",
            "category": cat,
            "source": text,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "origin": "authored context rendering reference; no new Agent solving",
            "constraints": {
                "page_group": (i - 1) // 4,
                "structure_preserved": True,
                "no_clipping": True,
                "visual_transcription": "requires evaluation",
            },
            "split": "development",
            "training_allowed": False,
            "actual_personal_reference": None,
        }
        for i, (cat, text) in enumerate(sources, 1)
    ]


def manifest() -> dict:
    result = {"schema": "context-handwriting-benchmark/1", "version": 1, "cases": cases()}
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result
