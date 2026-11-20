# ruff: noqa: E501
"""New complete homework references; not personal data or model training targets.

The eight holdout cases are evaluation-only. See docs/M1_BENCHMARK_PROTOCOL.md.
No renderer is invoked by this module, including by ``write_inputs``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

VERSION = "homework-m1-v1"
AUTHOR = "OpenAI Codex, newly authored benchmark on 2026-11-20"


def _case(
    identity: str,
    domain: str,
    title: str,
    statement: str,
    solution: str,
    numeric: list[tuple[str, str]],
    semantic: list[str],
    structures: list[str],
    *,
    split: str = "development",
    reject: str | None = None,
    paper: str = "plain",
    dense: bool = False,
) -> dict:
    return {
        "id": identity,
        "split": split,
        "domain": domain,
        "title": title,
        "statement": statement.strip(),
        "solution": solution.strip() + "\n",
        "source": {
            "kind": "agent-authored",
            "author": AUTHOR,
            "revision": VERSION,
            "license": "MIT",
            "historical_assignment": False,
            "unseen_personal_reference": False,
        },
        "expected": {
            "outcome": "reject" if reject else "render",
            "error_contains": reject,
            "page_count": 0 if reject else 1,
            "structures": structures,
        },
        "background": {"kind": paper, **({"path": "custom-paper.png"} if paper == "image" else {})},
        "density": "dense" if dense else "normal",
        "numeric_checks": [{"lhs": a, "rhs": b} for a, b in numeric],
        "semantic_checks": semantic,
        "isolation": {
            "model_training": False,
            "renderer_tuning": split == "development",
            "writer_generalization_claim": False,
        },
    }


_CASES = [
    _case(
        "m1-la-01-rref-null",
        "linear-algebra",
        "RREF, free variables and null space",
        "Let A=[[1,2,-1,3],[2,4,1,9],[-1,-2,2,-2]]. "
        "(a) Compute its reduced row echelon form. (b) Give its rank, nullity and a null-space "
        "basis. (c) Verify the basis against the original matrix, distinguishing free variables "
        "from pivot variables.",
        r"""(a) First subtract twice the first row from the second, and add the first row to the third.

$$A=\begin{pmatrix}1&2&-1&3\\2&4&1&9\\-1&-2&2&-2\end{pmatrix}$$

The last two rows become $(0,0,3,3)$ and $(0,0,1,1)$. Their dependence leaves two pivots.

$$R=\begin{pmatrix}1&2&0&4\\0&0&1&1\\0&0&0&0\end{pmatrix}$$

(b) Write $x_2=s$ and $x_4=t$. The pivot equations give $x_1=-2s-4t$ and $x_3=-t$.

$$\vec{x}=s\begin{pmatrix}-2\\1\\0\\0\end{pmatrix}+t\begin{pmatrix}-4\\0\\-1\\1\end{pmatrix}$$

Thus the rank is two and the nullity is two. The two displayed vectors are independent because their free coordinates are independent.

(c) Multiplying each vector by the original matrix gives the zero vector. This checks the original equations, not only the reduced system.

$$\mathrm{rank}(A)+\mathrm{nullity}(A)=2+2=4$$""",
        [("1*(-2)+2", "0"), ("2*(-4)+1*(-1)+9", "0"), ("4+2*(-1)-2", "0")],
        ["RREF has pivot columns one and three", "Both proposed null vectors are annihilated by A"],
        ["matrix-bracket", "subscript", "vector"],
        dense=True,
    ),
    _case(
        "m1-la-02-orthogonal-projection",
        "linear-algebra",
        "Projection and residual energy",
        "Let u=(1,2,-1), v=(2,0,2), b=(2,1,3). (a) Project b onto span{u,v}. "
        "(b) Find the residual and squared distance. (c) Explain why the minimizer is unique.",
        r"""(a) The spanning vectors are nonzero and orthogonal, since $u^{\mathrm{T}}v=0$. Their squared lengths are six and eight.

$$p=\frac{u^{\mathrm{T}}b}{u^{\mathrm{T}}u}u+\frac{v^{\mathrm{T}}b}{v^{\mathrm{T}}v}v=\frac{1}{6}u+\frac{5}{4}v$$

$$p=\begin{pmatrix}8/3\\1/3\\7/3\end{pmatrix},\qquad r=b-p=\begin{pmatrix}-2/3\\2/3\\2/3\end{pmatrix}$$

(b) The residual is orthogonal to both spanning vectors. Consequently it is orthogonal to every vector in the subspace.

$$u^{\mathrm{T}}r=0,\qquad v^{\mathrm{T}}r=0,\qquad r^{\mathrm{T}}r=\frac{4}{3}$$

(c) For any other point $p+w$ in the subspace, the cross term vanishes.

$$\Vert b-(p+w)\Vert^2=\Vert r\Vert^2+\Vert w\Vert^2\ge\frac{4}{3}$$

Equality holds only when $w=0$. This proves uniqueness without assuming that the original three coordinates are independent.""",
        [("1*2+2*0-1*2", "0"), ("4/9+4/9+4/9", "4/3")],
        ["Projection coefficients 1/6 and 5/4", "Residual orthogonal to span, squared norm 4/3"],
        ["matrix-bracket", "fraction-bar", "norm"],
    ),
    _case(
        "m1-la-03-two-projectors",
        "linear-algebra",
        "Spectral projectors and matrix powers",
        "For A=[[5,2],[2,2]], (a) obtain an orthonormal eigenbasis, "
        "(b) find the two spectral projectors, and (c) express A^n for nonnegative integers n and A's positive square root.",
        r"""(a) The characteristic polynomial separates into two distinct positive roots.

$$\mathrm{det}(A-\lambda I)=(5-\lambda)(2-\lambda)-4=(\lambda-6)(\lambda-1)$$

$$u=\frac{1}{\sqrt{5}}\begin{pmatrix}2\\1\end{pmatrix},\qquad v=\frac{1}{\sqrt{5}}\begin{pmatrix}-1\\2\end{pmatrix}$$

Direct multiplication gives $Au=6u$, $Av=v$ and $u^{\mathrm{T}}v=0$.

(b) The orthogonal projectors are the outer products.

$$P=\frac{1}{5}\begin{pmatrix}4&2\\2&1\end{pmatrix},\qquad Q=\frac{1}{5}\begin{pmatrix}1&-2\\-2&4\end{pmatrix}$$

They satisfy $P+Q=I$, $PQ=0$, $P^2=P$ and $Q^2=Q$.

(c) Powers and the positive square root follow from these identities.

$$A^n=6^nP+Q,\qquad A^{1/2}=\sqrt{6}P+Q$$

Squaring the last expression recovers $6P+Q=A$. Both eigenvalues of the proposed square root are positive.""",
        [("5*2-4", "6"), ("4/5+1/5", "1"), ("6**2*4/5+1/5", "29")],
        [
            "Eigenvalues 6 and 1 with orthonormal columns",
            "Projectors idempotent, orthogonal and sum to I",
        ],
        ["matrix-bracket", "radical-sign", "superscript"],
    ),
    _case(
        "m1-la-04-jordan-recurrence",
        "linear-algebra",
        "A repeated eigenvalue and recurrence",
        "Let A=[[2,1],[0,2]] and x_(n+1)=A x_n with x_0=(1,3). "
        "(a) Write A=2I+N and show N^2=0. (b) Derive A^n and x_n for n>=0. "
        "(c) Evaluate x_4 and the determinant of a product of m copies of A.",
        r"""(a) The strictly upper triangular part is nilpotent.

$$N=\begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad N^2=0$$

(b) All binomial terms containing two or more copies of N vanish. The two remaining terms are sufficient even though the matrix does not have two independent eigenvectors.

$$A^n=2^nI+n2^{n-1}N=\begin{pmatrix}2^n&n2^{n-1}\\0&2^n\end{pmatrix}$$

For $n=0$ the off-diagonal entry is zero. Multiplication by the initial vector gives

$$x_n=\begin{pmatrix}2^n+3n2^{n-1}\\3\,2^n\end{pmatrix},\qquad x_4=\begin{pmatrix}112\\48\end{pmatrix}$$

(c) Determinants multiply, so the result depends on the number of factors rather than on a new diagonalization.

$$\mathrm{det}(A^m)=\prod_{k=1}^{m}\mathrm{det}(A)=4^m$$

The empty product at $m=0$ is one, consistent with the identity matrix. Repeated multiplication also gives $x_1=(5,6)$, an independent check of the recurrence formula.""",
        [("2**4+3*4*2**3", "112"), ("3*2**4", "48"), ("2*1+3", "5")],
        ["Nilpotent part squares to zero", "Matrix power and recurrence hold for nonnegative n"],
        ["matrix-bracket", "product", "superscript"],
        paper="grid",
        dense=True,
    ),
    _case(
        "m1-la-05-hermitian",
        "linear-algebra",
        "Complex orthogonality and inverse",
        "Let H=[[3,i],[-i,3]], where i^2=-1. (a) Find its eigenvalues and normalized eigenvectors. "
        "(b) Check their complex inner product. (c) Find H inverse and verify it by multiplication.",
        r"""(a) The matrix equals its conjugate transpose. Its characteristic equation is

$$(3-\lambda)^2-1=0,\qquad \lambda_1=2,\quad\lambda_2=4$$

$$u=\frac{1}{\sqrt{2}}\begin{pmatrix}-\mathrm{i}\\1\end{pmatrix},\qquad v=\frac{1}{\sqrt{2}}\begin{pmatrix}\mathrm{i}\\1\end{pmatrix}$$

(b) Conjugating the first vector is essential. Without conjugation, the usual real dot-product calculation would not test complex orthogonality.

$$\langle u,v\rangle=\frac{\mathrm{i}\,\mathrm{i}+1}{2}=0,\qquad\langle u,u\rangle=\langle v,v\rangle=1$$

(c) The determinant is eight, and the inverse is

$$H^{-1}=\frac{1}{8}\begin{pmatrix}3&-\mathrm{i}\\\mathrm{i}&3\end{pmatrix}$$

The diagonal entries of $HH^{-1}$ are $(9-1)/8=1$, while both off-diagonal entries cancel. This verifies the inverse and the signs of its imaginary entries.""",
        [("(9-1)/8", "1"), ("2*4", "8")],
        ["Conjugate inner product, not real dot product", "Eigenvalues 2,4 and exact inverse"],
        ["matrix-bracket", "radical-sign", "inner-product"],
        split="holdout",
    ),
    _case(
        "m1-la-06-rank-compatibility",
        "linear-algebra",
        "Rank factorization and consistency",
        "Let A=[[1,0,2],[0,1,-1],[2,1,3]]. (a) Factor A as B C with B having independent columns. "
        "(b) Characterize every right-hand side b for which Ax=b is consistent. "
        "(c) Solve the system for b=(3,-2,4) and state the nullity.",
        r"""(a) The first two columns are independent and the third is twice the first minus the second.

$$B=\begin{pmatrix}1&0\\0&1\\2&1\end{pmatrix},\qquad C=\begin{pmatrix}1&0&2\\0&1&-1\end{pmatrix},\qquad A=BC$$

(b) The third row of A is twice the first plus the second. Therefore consistency requires, and is guaranteed by,

$$b_3=2b_1+b_2$$

The left-null vector $(-2,-1,1)$ gives the same compatibility condition.

(c) Here $4=2(3)-2$, so the system is consistent. Set the free coordinate $x_3=t$.

$$x=\begin{pmatrix}3-2t\\-2+t\\t\end{pmatrix},\qquad t\in\mathbb{R}$$

$$\mathrm{Null}(A)=\left\{t\begin{pmatrix}-2\\1\\1\end{pmatrix}:t\in\mathbb{R}\right\}$$

There are two pivots and one free coordinate. Thus rank two plus nullity one equals the number of columns, three.""",
        [("2*3-2", "4"), ("2*(-2)+1+3", "0")],
        ["A=BC, rank 2", "Consistency iff b3=2b1+b2", "Parametric solution for all real t"],
        ["matrix-bracket", "set-brace", "membership"],
        split="holdout",
    ),
    _case(
        "m1-ca-01-two-substitutions",
        "calculus",
        "Rational and logarithmic integrals",
        "(a) Evaluate the integral of x/(x^2+4)^2 from 0 to 2. "
        "(b) Evaluate the integral of x ln(1+x^2) from 0 to 1. "
        "For each integral show the transformed bounds and verify the antiderivative by differentiation.",
        r"""(a) With $u=x^2+4$, the differential is $du=2x\,dx$ and the endpoints become four and eight.

$$\int_0^2\frac{x}{(x^2+4)^2}\,dx=\frac{1}{2}\int_4^8u^{-2}\,du=\frac{1}{16}$$

An antiderivative is $-1/[2(x^2+4)]$. Its derivative has the positive sign required by the original nonnegative integrand.

(b) Now use $v=1+x^2$. The endpoints are one and two, and integration by parts gives $\int\ln v\,dv=v\ln v-v$.

$$\int_0^1x\ln(1+x^2)\,dx=\frac{1}{2}\int_1^2\ln v\,dv=\ln 2-\frac{1}{2}$$

$$F(x)=\frac{(1+x^2)\ln(1+x^2)-(1+x^2)}{2}$$

Differentiating F cancels the two plain x terms and leaves $x\ln(1+x^2)$. Keeping the constant contribution at the lower endpoint avoids an incorrect answer of $\ln2-1$.""",
        [("(1/4-1/8)/2", "1/16"), ("(2-1)/2", "1/2")],
        [
            "Both antiderivatives differentiate to original integrands",
            "Both transformed bounds retained",
        ],
        ["integral", "fraction-bar", "logarithm"],
    ),
    _case(
        "m1-ca-02-ellipse-integral",
        "calculus",
        "Double integration over an ellipse",
        "Let D be x^2/4+y^2<=1 and f(x,y)=x^2+4y^2. "
        "(a) Compute the integral of f over D using an elliptical polar substitution. "
        "(b) Find the area of D and the average of f. State the Jacobian and integration limits.",
        r"""Use $x=2r\cos\theta$ and $y=r\sin\theta$, with $0\le r\le1$ and $0\le\theta\le2\pi$. The absolute Jacobian is $2r$.

$$x^2+4y^2=4r^2,\qquad dx\,dy=2r\,dr\,d\theta$$

(a) Both the integrand and the area element must be transformed.

$$\iint_D(x^2+4y^2)\,dx\,dy=\int_0^{2\pi}\int_0^1 8r^3\,dr\,d\theta=4\pi$$

(b) Replacing the integrand by one gives the area.

$$|D|=\int_0^{2\pi}\int_0^1 2r\,dr\,d\theta=2\pi$$

$$f_{\mathrm{avg}}=\frac{4\pi}{2\pi}=2$$

The original integrand lies between zero and four on the ellipse, so the average of two is consistent with those bounds. Omitting the Jacobian would change the radial weight and give an incorrect average.""",
        [("8/4*2", "4"), ("4/2", "2")],
        ["Absolute Jacobian 2r", "Integral 4*pi, area 2*pi, average 2"],
        ["double-integral", "integral", "fraction-bar"],
        paper="dots",
    ),
    _case(
        "m1-ca-03-piecewise-density",
        "calculus",
        "Normalization and a piecewise distribution",
        "Let f(x)=k x(2-x) for 0<x<2 and zero elsewhere. "
        "(a) Normalize f. (b) Express its distribution function as one exact LaTeX cases environment. "
        "(c) Find the mean and variance. This capability test must retain the native cases environment.",
        r"""(a) Integrating the quadratic over its support determines the constant.

$$1=k\int_0^2x(2-x)\,dx=\frac{4k}{3},\qquad k=\frac{3}{4}$$

(b) The distribution function is continuous and includes both outside intervals.

$$F(x)=\begin{cases}0&x\le0\\3x^2/4-x^3/4&0<x<2\\1&x\ge2\end{cases}$$

(c) Symmetry about one gives the mean, and direct moments verify it.

$$E[X]=\frac{3}{4}\int_0^2x^2(2-x)\,dx=1$$

$$E[X^2]=\frac{3}{4}\int_0^2x^3(2-x)\,dx=\frac{6}{5},\qquad\mathrm{Var}(X)=\frac{1}{5}$$

At zero and two the middle polynomial equals the outside limits. Its derivative on the support is the normalized density. The native piecewise notation is part of the required content and must not disappear from an accepted page.""",
        [("(3/4)*(4/3)", "1"), ("6/5-1", "1/5")],
        [
            "Density normalized and nonnegative",
            "CDF continuous at both endpoints",
            "Mean 1, variance 1/5",
        ],
        ["piecewise", "integral", "fraction-bar"],
        reject="Unsupported LaTeX environment",
    ),
    _case(
        "m1-ca-04-taylor-series",
        "calculus",
        "Taylor remainder and a telescoping sum",
        "(a) Compute the limit as x tends to zero of (exp(x)-1-x-x^2/2)/x^3. "
        "(b) Bound the remainder after the cubic term for |x|<=1/2. "
        "(c) Evaluate the infinite sum of 1/[n(n+2)] for n from 1 upward by finite partial sums.",
        r"""(a) Taylor expansion retains the cubic term before division.

$$e^x=1+x+\frac{x^2}{2}+\frac{x^3}{6}+R_4(x)$$

$$\lim_{x\to0}\frac{e^x-1-x-x^2/2}{x^3}=\frac{1}{6}$$

(b) The fourth derivative is $e^x$. On the specified interval it is at most $e^{1/2}$, so the Lagrange remainder satisfies

$$|R_4(x)|\le\frac{e^{1/2}|x|^4}{24},\qquad |x|\le\frac{1}{2}$$

This also implies $R_4(x)/x^3\to0$ and justifies the limit.

(c) Partial fractions make cancellation explicit without subtracting divergent series.

$$\frac{1}{n(n+2)}=\frac{1}{2}\left(\frac{1}{n}-\frac{1}{n+2}\right)$$

$$S_N=\frac{3}{4}-\frac{1}{2(N+1)}-\frac{1}{2(N+2)}$$

$$\sum_{n=1}^{\infty}\frac{1}{n(n+2)}=\frac{3}{4}$$

The two tail terms are positive, so every finite partial sum is smaller than its limit. The first partial sum is one third, providing a quick independent endpoint check.""",
        [("3/4-1/4-1/6", "1/3"), ("1/2*(1-1/3)", "1/3")],
        [
            "Taylor limit 1/6 with stated fourth-order remainder",
            "Finite partial sums telescope before limit",
        ],
        ["limit", "summation", "absolute-value", "subscript"],
        dense=True,
    ),
    _case(
        "m1-ca-05-critical-points",
        "calculus",
        "Stationary points and shifted coordinates",
        "For f(x,y)=x^3+xy+y^2, (a) find every stationary point, "
        "(b) classify each using the Hessian, and (c) expand f about its local minimum using x=1/6+xi and y=-1/12+eta.",
        r"""(a) The two first derivatives must vanish simultaneously.

$$3x^2+y=0,\qquad x+2y=0$$

Substitution gives $x(3x-1/2)=0$. Thus the stationary points are $(0,0)$ and $(1/6,-1/12)$.

(b) The Hessian is

$$H=\begin{pmatrix}6x&1\\1&2\end{pmatrix}$$

At the origin its determinant is negative, so the point is a saddle. At the second point the leading minors are one and one, both positive, giving a strict local minimum.

$$f(1/6,-1/12)=-\frac{1}{432}$$

(c) Put $x=1/6+\xi$ and $y=-1/12+\eta$. Direct expansion gives

$$f(x,y)=-\frac{1}{432}+\frac{\xi^2}{2}+\xi\eta+\eta^2+\xi^3$$

There are no linear terms, as required at a stationary point. The positive quadratic part controls sufficiently small perturbations, while the cubic term prevents a claim of a global minimum.""",
        [("(1/6)**3+(1/6)*(-1/12)+(-1/12)**2", "-1/432"), ("1*2-1", "1")],
        [
            "Exactly two real stationary points",
            "Saddle and strict local minimum",
            "Exact xi/eta expansion",
        ],
        ["matrix-bracket", "greek", "superscript"],
        split="holdout",
    ),
    _case(
        "m1-ca-06-improper-log",
        "calculus",
        "An endpoint singularity and a parameter family",
        "(a) Evaluate the improper integral of ln(x)/sqrt(x) from 0 to 1 and establish absolute convergence. "
        "(b) For p>0 evaluate I(p)=integral_0^1 x^(p-1) ln(x) dx. "
        "(c) Compute the limit of n I(n) as n tends to infinity.",
        r"""(a) Set $x=t^2$ before taking the lower endpoint limit.

$$\int_0^1\frac{\ln x}{\sqrt{x}}\,dx=4\int_0^1\ln t\,dt=-4$$

The endpoint term $t\ln t$ tends to zero. Since the logarithm is nonpositive on this interval, the absolute integral equals four.

(b) For $p>0$, integration by parts with a positive lower cutoff gives

$$\int_{\epsilon}^1x^{p-1}\ln x\,dx=\left[\frac{x^p\ln x}{p}-\frac{x^p}{p^2}\right]_{\epsilon}^{1}$$

As $\epsilon\to0$, both $\epsilon^p$ and $\epsilon^p\ln\epsilon$ vanish. Therefore

$$I(p)=-\frac{1}{p^2}$$

(c) It follows that $nI(n)=-1/n$, which tends to zero from below. Substitution of $p=1/2$ recovers the value in part (a), checking the parameter formula independently.""",
        [("-1/(1/2)**2", "-4"), ("10*(-1/10**2)", "-1/10")],
        ["Endpoint cutoff and absolute convergence retained", "I(p)=-1/p^2 for p>0"],
        ["integral", "radical-sign", "limit", "greek"],
        split="holdout",
    ),
    _case(
        "m1-st-01-normal-likelihood",
        "statistics",
        "Normal estimates and a known-variance interval",
        "The independent normal observations are 2,2,5,7. (a) Find the maximum-likelihood mean and variance. "
        "(b) Find the unbiased sample variance. (c) In a separate known-sigma model with sigma=3, "
        "construct a 95 percent mean interval using z=1.96. Keep the variance assumptions distinct.",
        r"""(a) The likelihood is a product over the four independent observations.

$$L(\mu,\sigma^2)=\prod_{i=1}^{4}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

The sum is sixteen, so $\hat\mu=4$. The squared deviations sum to eighteen.

$$\hat\sigma^2=\frac{18}{4}=\frac{9}{2}$$

(b) Dividing by three instead gives the unbiased sample variance $s^2=6$. These two estimators use different denominators and answer different questions.

(c) For the separate model in which sigma is known to be three, the standard error of the mean is $3/\sqrt{4}=3/2$.

$$\mu:\quad4\pm1.96\frac{3}{2},\qquad[1.06,6.94]$$

The normal critical value is justified here by the supplied known variance. It must not be substituted for a Student critical value merely because a sample variance has also been computed.""",
        [("(2+2+5+7)/4", "4"), ("4+4+1+9", "18"), ("4-1.96*3/2", "1.06"), ("4+1.96*3/2", "6.94")],
        [
            "MLE denominator n, unbiased denominator n-1",
            "Known variance interval is a separate model",
        ],
        ["product", "fraction-bar", "radical-sign", "hat-accent"],
        dense=True,
    ),
    _case(
        "m1-st-02-weighted-fit",
        "statistics",
        "Weighted regression and covariance",
        "Fit y=a+bx to observations (-1,0),(0,1),(2,5) with weights 1,2,1. "
        "Assume independent errors with variance sigma^2/w_i. (a) Solve the weighted normal equations. "
        "(b) Verify weighted residual orthogonality and find weighted SSE. (c) State the coefficient covariance matrix.",
        r"""(a) The weights enter both cross-products. The normal equations are

$$\begin{pmatrix}4&1\\1&5\end{pmatrix}\begin{pmatrix}\hat a\\\hat b\end{pmatrix}=\begin{pmatrix}7\\10\end{pmatrix}$$

$$\hat a=\frac{25}{19},\qquad\hat b=\frac{33}{19}$$

(b) The residuals in observation order are $(8,-6,4)/19$. Their weighted sum and weighted cross-product with x vanish.

$$\sum_iw_i e_i=\frac{8-12+4}{19}=0,\qquad\sum_iw_ix_i e_i=\frac{-8+8}{19}=0$$

$$SSE_w=\frac{64+72+16}{19^2}=\frac{8}{19}$$

(c) Under the stated variance model, weighted least squares has covariance

$$\mathrm{Cov}(\hat a,\hat b)=\frac{\sigma^2}{19}\begin{pmatrix}5&-1\\-1&4\end{pmatrix}$$

Thus the slope standard error, when sigma is known, is $2\sigma/\sqrt{19}$. Interpreting the weights as inverse relative variances is essential; arbitrary numerical weights alone would not justify this covariance formula.""",
        [("4*25/19+33/19", "7"), ("25/19+5*33/19", "10"), ("(64+72+16)/19**2", "8/19")],
        [
            "Exact normal equation solution",
            "Weighted, not unweighted residual checks",
            "Covariance follows stated heteroscedastic model",
        ],
        ["matrix-bracket", "summation", "hat-accent", "radical-sign"],
        dense=True,
    ),
    _case(
        "m1-st-03-contingency",
        "statistics",
        "Independence in a two-by-three table",
        "The observed table is [[12,18,30],[18,12,30]]. "
        "(a) Find expected counts under independence. (b) Compute Pearson's chi-square statistic and degrees of freedom. "
        "(c) Compare with 5.991 at alpha=0.05, and give the asymptotic p-value using the df=2 tail formula exp(-x/2).",
        r"""(a) Each row total is sixty. Column totals are thirty, thirty and sixty, with grand total 120.

$$E=\begin{pmatrix}15&15&30\\15&15&30\end{pmatrix}$$

Every expected count is large enough for the usual chi-square approximation in this exercise.

(b) Four cells contribute $9/15$ and the two last-column cells contribute zero.

$$\chi^2=\sum_{i=1}^{2}\sum_{j=1}^{3}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}=\frac{12}{5}=2.4$$

$$\nu=(2-1)(3-1)=2$$

(c) Since $2.4<5.991$, do not reject independence at the stated level. This is a failure to reject, not proof that the variables are independent.

$$p=\exp(-2.4/2)=e^{-6/5}\approx0.3010$$

The tail probability exceeds $\alpha=0.05$, agreeing with the critical-value decision. Both conclusions use the same asymptotic reference distribution.""",
        [("60*30/120", "15"), ("4*9/15", "12/5"), ("(2-1)*(3-1)", "2")],
        ["Expected counts computed from marginal totals", "Chi-square 12/5, df=2, non-rejection"],
        ["matrix-bracket", "summation", "greek", "subscript"],
        paper="image",
    ),
    _case(
        "m1-st-04-annotated-posterior",
        "statistics",
        "Bernoulli likelihood and a beta posterior",
        "Independent Bernoulli outcomes are 1,0,1,1,0. Use a uniform prior for p on (0,1). "
        "(a) Obtain the MLE. (b) Normalize the posterior and find its mean and mode. "
        "Retain an exact underbrace annotation naming the likelihood factor in the posterior expression.",
        r"""(a) There are three successes and two failures. Maximizing the log likelihood gives

$$\ln L(p)=3\ln p+2\ln(1-p),\qquad\hat p=\frac{3}{5}$$

(b) The uniform prior is constant on the parameter interval, so the posterior is proportional to the likelihood.

$$f(p| x)=60\underbrace{p^3(1-p)^2}_{\mathrm{likelihood}},\qquad0<p<1$$

The integral of $p^3(1-p)^2$ from zero to one is $1/60$, establishing normalization. This is the beta distribution with parameters four and three.

$$E[p| x]=\frac{4}{7},\qquad p_{\mathrm{mode}}=\frac{3}{5}$$

The mean and mode are different summaries, even though the mode agrees with the MLE under this prior. The likelihood annotation is required source content; a renderer must explicitly refuse it if unsupported.""",
        [("1/4-2/5+1/6", "1/60"), ("60*(1/5-2/6+1/7)", "4/7")],
        ["Normalized beta(4,3) posterior", "Posterior mean 4/7 and interior mode 3/5"],
        ["underbrace", "subscript", "fraction-bar"],
        reject="Unsupported LaTeX command: underbrace",
    ),
    _case(
        "m1-st-05-paired-inference",
        "statistics",
        "Paired differences and uncertainty",
        "Six independent normally distributed paired differences, defined as after minus before, "
        "are 1,2,4,5,3,3. (a) Compute the mean and unbiased variance of the differences. "
        "(b) Test zero mean at the two-sided 5 percent level using critical value 2.571. "
        "(c) Give the corresponding confidence interval and explain the direction of the effect.",
        r"""(a) Analysis uses the differences as the observations. Their sum is eighteen and their squared deviations from three sum to ten.

$$\bar d=3,\qquad s_d^2=\frac{10}{6-1}=2,\qquad\mathrm{SE}(\bar d)=\sqrt{\frac{2}{6}}=\frac{1}{\sqrt{3}}$$

(b) The null hypothesis is $H_0:\mu_d=0$. The Student statistic and its degrees of freedom are

$$T=\frac{3}{1/\sqrt{3}}=3\sqrt{3},\qquad\nu=5$$

Because $3\sqrt{3}>2.571$, reject the null at the stated level.

(c) The interval is

$$\mu_d:\quad3\pm\frac{2.571}{\sqrt{3}}$$

Both endpoints are positive. Since the differences were defined as after minus before, the estimated change is an increase. Treating the twelve original measurements as independent would discard the pairing and would require information not supplied here.""",
        [("(1+2+4+5+3+3)/6", "3"), ("(4+1+1+4)/5", "2"), ("3**2*3", "27")],
        [
            "Paired, independent differences with df5",
            "Two-sided reject and positive confidence endpoints",
        ],
        ["bar-accent", "radical-sign", "subscript", "plus-minus"],
        split="holdout",
    ),
    _case(
        "m1-st-06-exponential-scale",
        "statistics",
        "Exponential scale and rate estimation",
        "Independent exponential observations with mean theta are 1,2,2,3,7. "
        "(a) Find the MLE and variance of the sample-mean estimator of theta. "
        "(b) Find the rate MLE and an unbiased rate estimator. "
        "(c) Use the chi-square pivot with 10 degrees of freedom and supplied quantiles 3.247 and 20.483 for a 95 percent scale interval.",
        r"""(a) The sample size is five and the sum is fifteen. Differentiating the log likelihood gives

$$\ln L(\theta)=-5\ln\theta-\frac{15}{\theta},\qquad\hat\theta=3$$

The sample mean is unbiased and has variance $\theta^2/5$. Its plug-in variance estimate is $9/5$.

(b) Invariance gives the rate MLE $\hat\lambda=1/3$. For a gamma sum with shape five, the expectation of the reciprocal is $\lambda/4$.

$$\hat\lambda_U=\frac{4}{15}$$

This last estimator is unbiased; the rate MLE itself is not.

(c) The pivot is $2\sum_iX_i/\theta$, with a chi-square distribution having ten degrees of freedom. Reversing the inequalities when solving for the positive scale gives

$$\theta\in\left[\frac{30}{20.483},\frac{30}{3.247}\right]$$

The upper chi-square quantile produces the lower scale endpoint. The interval depends on the exponential model and the supplied quantiles, not on a normal approximation for this small sample.""",
        [("(1+2+2+3+7)/5", "3"), ("3**2/5", "9/5"), ("(5-1)/15", "4/15")],
        [
            "Scale MLE 3, rate MLE 1/3",
            "Unbiased rate 4/15",
            "Chi-square inverse interval endpoints ordered correctly",
        ],
        ["hat-accent", "fraction-bar", "membership", "tall-bracket"],
        split="holdout",
        dense=True,
    ),
    _case(
        "m1-ph-01-potential-field",
        "physics",
        "Potential, field and work",
        "In a charge-free region the electrostatic potential is Phi=kappa(x^2-2y^2+z^2). "
        "Coordinates are in metres and kappa=3 V/m^2. "
        "(a) Find E and its divergence. (b) Evaluate E and its magnitude at (1,-1,2). "
        "(c) Find the work done by the field on a 2 C charge moved from the origin to that point.",
        r"""(a) The electric field is the negative potential gradient.

$$\vec E=-\nabla\Phi=(-2\kappa x,4\kappa y,-2\kappa z)$$

$$\nabla\cdot\vec E=-2\kappa+4\kappa-2\kappa=0,\qquad\nabla\times\vec E=0$$

The zero divergence is consistent with zero charge density in the region.

(b) At the stated point and parameter value,

$$\vec E=(-6,-12,-12)\ \mathrm{V/m},\qquad|\vec E|=\sqrt{36+144+144}=18\ \mathrm{V/m}$$

(c) The potential difference is nine volts. Work by the field is the negative change in electric potential energy.

$$\Delta\Phi=3(1-2+4)=9\ \mathrm{V},\qquad W=-q\Delta\Phi=-18\ \mathrm{J}$$

An external agent moving the charge quasistatically would do positive eighteen joules of work. Keeping the two agents distinct prevents a sign reversal.""",
        [("-6+12-6", "0"), ("36+144+144", "18**2"), ("-2*3*(1-2+4)", "-18")],
        ["Field is negative gradient, divergence and curl zero", "Field work -18 J"],
        ["vector", "nabla", "radical-sign", "greek", "units"],
    ),
    _case(
        "m1-ph-02-rlc-phase",
        "physics",
        "AC phase, power and damping",
        "A series circuit has R=12 ohm, L=0.08 H, C=1/800 F, and an rms source of 48 V at omega=50 rad/s. "
        "(a) Find impedance and current magnitude/phase. (b) Find real and reactive power. "
        "(c) Find the undamped frequency, damping ratio and damped natural frequency.",
        r"""(a) The inductive and capacitive reactances are four and sixteen ohms.

$$Z=R+\mathrm{i}\left(\omega L-\frac{1}{\omega C}\right)=12-12\mathrm{i}\ \Omega$$

$$|Z|=12\sqrt{2}\ \Omega,\qquad I_{\mathrm{rms}}=2\sqrt{2}\ \mathrm{A}$$

The impedance angle is $-\pi/4$. Therefore current leads the voltage by $\pi/4$.

(b) The real power is positive, while the reactive power is negative because the net reactance is capacitive.

$$P=I_{\mathrm{rms}}^2R=96\ \mathrm{W},\qquad Q=I_{\mathrm{rms}}^2(-12)=-96\ \mathrm{var}$$

(c) For the unforced series-circuit equation,

$$\omega_0=\frac{1}{\sqrt{LC}}=100\ \mathrm{rad/s},\qquad\alpha=\frac{R}{2L}=75\ \mathrm{s^{-1}}$$

$$\zeta=\frac{\alpha}{\omega_0}=\frac{3}{4},\qquad\omega_d=\sqrt{\omega_0^2-\alpha^2}=25\sqrt{7}\ \mathrm{rad/s}$$

The damping ratio is less than one, so the natural transient is underdamped. This natural frequency is distinct from the imposed source frequency of fifty radians per second.""",
        [("50*0.08", "4"), ("1/(50*(1/800))", "16"), ("8*12", "96"), ("100**2-75**2", "25**2*7")],
        [
            "Capacitive impedance and leading current",
            "P96, Q-96",
            "Damping ratio 3/4, natural frequencies distinct",
        ],
        ["radical-sign", "fraction-bar", "greek", "subscript", "units"],
        dense=True,
    ),
    _case(
        "m1-ph-03-faraday-loop",
        "physics",
        "Closed-loop induction and orientation",
        "A circular conducting loop of radius 0.20 m and resistance 2 ohm lies in the xy-plane. "
        "An external magnetic field B(t)=0.5 t^2 tesla points along positive z for t>=0. "
        "Take the positive loop direction counterclockwise viewed from positive z. "
        "(a) Find magnetic flux and signed emf. (b) Find current at t=3 s and interpret its direction. Neglect self-inductance.",
        r"""(a) The positive surface normal is along positive z, consistent with the stated loop orientation. The field is uniform across the area.

$$\Phi_B(t)=B(t)\pi r^2=\frac{\pi t^2}{50}\ \mathrm{Wb}$$

Faraday's law determines the signed circulation.

$$\oint\vec E\cdot d\vec r=-\frac{d\Phi_B}{dt}=-\frac{\pi t}{25}\ \mathrm{V}$$

(b) At three seconds the emf is negative relative to the specified counterclockwise direction.

$$\mathrm{emf}(3)=-\frac{3\pi}{25}\ \mathrm{V},\qquad I(3)=-\frac{3\pi}{50}\ \mathrm{A}$$

The actual current is clockwise as viewed from positive z. Its induced field opposes the increase in outward magnetic flux. Because self-inductance is neglected, Ohm's law can be applied directly to the instantaneous emf.""",
        [("0.5*0.20**2", "1/50"), ("(3/25)/2", "3/50")],
        [
            "Counterclockwise positive orientation and positive z normal",
            "Signed emf -pi*t/25",
            "Clockwise current at t3",
        ],
        ["closed-integral", "vector", "fraction-bar", "units"],
    ),
    _case(
        "m1-ph-04-quantum-well",
        "physics",
        "Normalized well state and position spread",
        "A particle of mass m occupies the ground state of an infinite well on 0<x<L, "
        "with psi(x)=sqrt(2/L) sin(pi x/L). (a) Verify normalization. "
        "(b) Derive its energy using the stationary Schrodinger equation. "
        "(c) Give mean position and position uncertainty. Retain the exact hbar command in the energy formula.",
        r"""(a) The integral of the squared sine over one half-period is $L/2$, so

$$\int_0^L|\psi(x)|^2\,dx=\frac{2}{L}\frac{L}{2}=1$$

(b) Twice differentiating the wave function gives $\psi''=-(\pi/L)^2\psi$. The stationary equation inside the well is

$$-\frac{\hbar^2}{2m}\psi''=E\psi,\qquad E=\frac{\pi^2\hbar^2}{2mL^2}$$

(c) Reflection symmetry about the centre gives the mean position. The second moment follows by integrating the squared wave function with $x^2$.

$$\langle x\rangle=\frac{L}{2},\qquad\langle x^2\rangle=L^2\left(\frac{1}{3}-\frac{1}{2\pi^2}\right)$$

$$\Delta x=L\sqrt{\frac{1}{12}-\frac{1}{2\pi^2}}$$

The expression under the square root is positive. The reduced Planck constant is part of the physical equation and must be rendered or explicitly refused, never silently removed.""",
        [("1/3-1/4", "1/12"), ("2*(1/2)", "1")],
        [
            "Normalized wave function",
            "Energy pi^2*hbar^2/(2*m*L^2)",
            "Position variance L^2*(1/12-1/(2*pi^2))",
        ],
        ["hbar", "greek", "integral", "inner-product", "radical-sign"],
        reject="Unsupported LaTeX command: hbar",
    ),
    _case(
        "m1-ph-05-two-lenses",
        "physics",
        "Two successive thin lenses",
        "Two converging thin lenses in air are 60 cm apart. Their focal lengths are 12 cm and 8 cm. "
        "An object is 18 cm to the left of the first lens. "
        "(a) Locate the first image. (b) Treat it as the object for the second lens and find the final image. "
        "(c) Give total transverse magnification and orientation, using real-object and real-image distances as positive.",
        r"""(a) Apply the thin-lens equation separately to each lens.

$$\frac{1}{12}=\frac{1}{18}+\frac{1}{s_1'},\qquad s_1'=36\ \mathrm{cm}$$

The first image lies thirty-six centimetres to the right of the first lens.

(b) It is therefore twenty-four centimetres to the left of the second lens, which makes it a real object for that lens.

$$s_2=60-36=24\ \mathrm{cm},\qquad\frac{1}{8}=\frac{1}{24}+\frac{1}{s_2'}$$

$$s_2'=12\ \mathrm{cm}$$

The final image is seventy-two centimetres to the right of the first lens.

(c) Multiply the signed transverse magnifications.

$$M_1=-\frac{36}{18}=-2,\qquad M_2=-\frac{12}{24}=-\frac{1}{2},\qquad M=M_1M_2=1$$

The two inversions cancel. The final image is real, upright relative to the original object, and equal in transverse size within the thin-lens approximation.""",
        [("1/18+1/36", "1/12"), ("1/24+1/12", "1/8"), ("(-36/18)*(-12/24)", "1")],
        [
            "First image at36cm, second object distance24cm",
            "Final image at72cm fromfirst, total magnification+1",
        ],
        ["fraction-bar", "subscript", "units"],
        split="holdout",
        paper="lined",
    ),
    _case(
        "m1-ph-06-travelling-wave",
        "physics",
        "Wave speed, tension and energy transport",
        "A travelling transverse string wave is psi(x,t)=A cos(kx-omega t+phi), "
        "where A=0.003 m, k=4 pi per metre, omega=80 pi per second, and the mass per unit length is 0.020 kg/m. "
        "(a) Find wavelength, frequency and propagation speed. (b) Find maximum transverse speed and acceleration. "
        "(c) Find string tension and mean transmitted power.",
        r"""(a) The phase $kx-\omega t+\phi$ propagates in the positive x direction.

$$\lambda=\frac{2\pi}{k}=\frac{1}{2}\ \mathrm{m},\qquad f=\frac{\omega}{2\pi}=40\ \mathrm{Hz},\qquad v=\frac{\omega}{k}=20\ \mathrm{m/s}$$

(b) Differentiate the displacement with respect to time while holding position fixed.

$$\frac{\partial\psi}{\partial t}=A\omega\sin(kx-\omega t+\phi)$$

$$v_{y,\max}=0.24\pi\ \mathrm{m/s},\qquad a_{y,\max}=19.2\pi^2\ \mathrm{m/s^2}$$

Transverse particle motion is distinct from the propagation of a fixed phase.

(c) For mass density $\mu=0.020\ \mathrm{kg/m}$, the ideal-string relation gives

$$T=\mu v^2=8\ \mathrm{N}$$

$$\bar P=\frac{1}{2}\mu\omega^2A^2v=\frac{36\pi^2}{3125}\ \mathrm{W}$$

The mean power is independent of phase phi. Doubling amplitude would quadruple this power, while leaving the propagation speed unchanged when tension and density remain fixed.""",
        [
            ("80/4", "20"),
            ("0.003*80", "0.24"),
            ("0.003*80**2", "19.2"),
            ("0.020*20**2", "8"),
            ("0.5*0.020*80**2*0.003**2*20", "36/3125"),
        ],
        [
            "Wave propagation and particle motion distinguished",
            "Tension8N and mean power36*pi^2/3125W",
        ],
        ["greek", "partial-derivative", "subscript", "bar-accent", "units"],
        split="holdout",
        dense=True,
    ),
]


def cases(split: str = "development") -> list[dict]:
    """Return independent records without changing the frozen in-module inventory."""
    if split not in {"development", "holdout", "all"}:
        raise ValueError("Expected development, holdout or all")
    return copy.deepcopy([c for c in _CASES if split == "all" or c["split"] == split])


def end_to_end_inputs() -> list[dict]:
    """Separate original text inputs, intentionally without reference solutions."""
    prompts = [
        (
            "m1-e2e-la",
            "linear-algebra",
            "Let B=[[2,-1],[3,4]] and c=(5,1). "
            "(a) Compute B inverse. (b) Solve Bx=c. (c) Check the answer in both original equations.",
        ),
        (
            "m1-e2e-ca",
            "calculus",
            "(a) Evaluate the integral of sin(x)^3 cos(x) from 0 to pi/2. "
            "(b) Find an antiderivative and differentiate it to check the substitution. "
            "State the new integration limits explicitly.",
        ),
        (
            "m1-e2e-st",
            "statistics",
            "Nine independent observations from a normal population "
            "have sample mean 14 and sample standard deviation 3. Test H0: mu=12 against a "
            "two-sided alternative at 5 percent using the supplied critical t value 2.306. "
            "Give the test statistic, degrees of freedom, decision and confidence interval.",
        ),
        (
            "m1-e2e-ph",
            "physics",
            "A single slit of width 0.30 mm is illuminated normally by "
            "light of wavelength 600 nm. A screen is 1.50 m away. "
            "(a) Calculate the first diffraction minimum angle using sin(theta)=lambda/a. "
            "(b) Find the approximate central-maximum width using the small-angle approximation. "
            "Show the conversion to metres and state why that approximation is appropriate.",
        ),
    ]
    return [
        {
            "id": identity,
            "domain": domain,
            "statement": statement,
            "source": {"kind": "agent-authored", "author": AUTHOR, "revision": VERSION},
            "input_format": "text",
            "reference_solution_supplied": False,
        }
        for identity, domain, statement in prompts
    ]


def verify_references() -> dict:
    """Execute source-level exact checks; not a rendered-text or visual verifier."""
    import sympy as sp

    from hwsol.jobs.checks import arithmetic

    evidence: dict[str, list[str]] = {c["id"]: [] for c in _CASES}

    def checked(identity: str, name: str, actual: object, expected: object) -> None:
        delta = actual - expected
        ok = (
            delta.applyfunc(sp.simplify) == sp.zeros(*delta.shape)
            if isinstance(delta, sp.MatrixBase)
            else sp.simplify(delta) == 0
        )
        if not ok:
            raise ValueError(
                f"Reference verification failed: {identity}: {name}: {actual} != {expected}"
            )
        evidence[identity].append(name)

    for row in _CASES:
        for n, check in enumerate(row["numeric_checks"], 1):
            if arithmetic(check["lhs"]) != arithmetic(check["rhs"]):
                raise ValueError(f"Bad numeric check {row['id']} #{n}")
            evidence[row["id"]].append(f"runtime rational equality {n}")

    M, R = sp.Matrix, sp.Rational
    a = M([[1, 2, -1, 3], [2, 4, 1, 9], [-1, -2, 2, -2]])
    checked(
        _CASES[0]["id"], "RREF exact", a.rref()[0], M([[1, 2, 0, 4], [0, 0, 1, 1], [0, 0, 0, 0]])
    )
    checked(
        _CASES[0]["id"],
        "null basis exact",
        a * M([[-2, -4], [1, 0], [0, -1], [0, 1]]),
        sp.zeros(3, 2),
    )
    u, v, b = M([1, 2, -1]), M([2, 0, 2]), M([2, 1, 3])
    p = (u.dot(b) / u.dot(u)) * u + (v.dot(b) / v.dot(v)) * v
    checked(_CASES[1]["id"], "projection", p, M([R(8, 3), R(1, 3), R(7, 3)]))
    checked(_CASES[1]["id"], "distance squared", (b - p).dot(b - p), R(4, 3))
    a = M([[5, 2], [2, 2]])
    p, q = M([[4, 2], [2, 1]]) / 5, M([[1, -2], [-2, 4]]) / 5
    checked(_CASES[2]["id"], "spectral reconstruction", 6 * p + q, a)
    checked(_CASES[2]["id"], "positive root squares to A", (sp.sqrt(6) * p + q) ** 2, a)
    n = sp.symbols("n", integer=True, nonnegative=True)
    a = M([[2, 1], [0, 2]])
    pn = M([[2**n, n * 2 ** (n - 1)], [0, 2**n]])
    checked(_CASES[3]["id"], "power recurrence", a * pn, pn.subs(n, n + 1))
    checked(_CASES[3]["id"], "power at zero", pn.subs(n, 0), sp.eye(2))
    h = M([[3, sp.I], [-sp.I, 3]])
    eigen = M([[-sp.I, sp.I], [1, 1]]) / sp.sqrt(2)
    checked(_CASES[4]["id"], "complex eigenpairs", h * eigen, eigen * sp.diag(2, 4))
    checked(_CASES[4]["id"], "unitary eigenbasis", eigen.conjugate().T * eigen, sp.eye(2))
    checked(_CASES[4]["id"], "inverse product", h * M([[3, -sp.I], [sp.I, 3]]) / 8, sp.eye(2))
    a = M([[1, 0, 2], [0, 1, -1], [2, 1, 3]])
    checked(
        _CASES[5]["id"],
        "rank factorization",
        M([[1, 0], [0, 1], [2, 1]]) * M([[1, 0, 2], [0, 1, -1]]),
        a,
    )
    t = sp.symbols("t", real=True)
    checked(
        _CASES[5]["id"],
        "complete parametric solution",
        a * M([3 - 2 * t, -2 + t, t]),
        M([3, -2, 4]),
    )
    checked(_CASES[5]["id"], "left-null condition", M([[-2, -1, 1]]) * a, sp.zeros(1, 3))
    x, y, xi, eta = sp.symbols("x y xi eta", real=True)
    checked(
        _CASES[6]["id"],
        "rational definite integral",
        sp.integrate(x / (x * x + 4) ** 2, (x, 0, 2)),
        R(1, 16),
    )
    checked(
        _CASES[6]["id"],
        "log definite integral",
        sp.integrate(x * sp.log(1 + x * x), (x, 0, 1)),
        sp.log(2) - R(1, 2),
    )
    r, theta = sp.symbols("r theta", real=True)
    xy = M([2 * r * sp.cos(theta), r * sp.sin(theta)])
    checked(_CASES[7]["id"], "elliptical Jacobian", xy.jacobian([r, theta]).det(), 2 * r)
    checked(
        _CASES[7]["id"],
        "ellipse integral",
        sp.integrate(8 * r**3, (r, 0, 1), (theta, 0, 2 * sp.pi)),
        4 * sp.pi,
    )
    density = R(3, 4) * x * (2 - x)
    for power, value in [(0, 1), (1, 1), (2, R(6, 5))]:
        checked(
            _CASES[8]["id"],
            f"density moment {power}",
            sp.integrate(x**power * density, (x, 0, 2)),
            value,
        )
    checked(_CASES[8]["id"], "CDF derivative", sp.diff(3 * x * x / 4 - x**3 / 4, x), density)
    checked(
        _CASES[9]["id"],
        "Taylor limit",
        sp.limit((sp.exp(x) - 1 - x - x * x / 2) / x**3, x, 0),
        R(1, 6),
    )
    k = sp.symbols("k", integer=True, positive=True)
    checked(
        _CASES[9]["id"], "convergent sum", sp.summation(1 / (k * (k + 2)), (k, 1, sp.oo)), R(3, 4)
    )
    f = x**3 + x * y + y * y
    points = sp.solve([sp.diff(f, x), sp.diff(f, y)], [x, y])
    if set(points) != {(0, 0), (R(1, 6), -R(1, 12))}:
        raise ValueError("Unexpected stationary points")
    evidence[_CASES[10]["id"]].append("all stationary points exact")
    checked(
        _CASES[10]["id"],
        "shifted polynomial",
        f.subs({x: xi + R(1, 6), y: eta - R(1, 12)}).expand(),
        -R(1, 432) + xi * xi / 2 + xi * eta + eta * eta + xi**3,
    )
    pvar = sp.symbols("p", positive=True)
    xp = sp.symbols("xp", positive=True)
    log_anti = 2 * sp.sqrt(xp) * sp.log(xp) - 4 * sp.sqrt(xp)
    checked(
        _CASES[11]["id"],
        "improper log antiderivative",
        sp.diff(log_anti, xp),
        sp.log(xp) / sp.sqrt(xp),
    )
    checked(
        _CASES[11]["id"],
        "improper log endpoint limits",
        log_anti.subs(xp, 1) - sp.limit(log_anti, xp, 0, dir="+"),
        -4,
    )
    cutoff = sp.symbols("epsilon", positive=True)
    anti = x**pvar * sp.log(x) / pvar - x**pvar / pvar**2
    checked(
        _CASES[11]["id"], "parameter antiderivative", sp.diff(anti, x), x ** (pvar - 1) * sp.log(x)
    )
    checked(_CASES[11]["id"], "cutoff limit", sp.limit(anti.subs(x, cutoff), cutoff, 0, dir="+"), 0)
    values = [sp.Integer(z) for z in [2, 2, 5, 7]]
    mean = sum(values) / 4
    checked(_CASES[12]["id"], "normal mean", mean, 4)
    checked(_CASES[12]["id"], "normal SSE", sum((v - mean) ** 2 for v in values), 18)
    design, weight, response = M([[1, -1], [1, 0], [1, 2]]), sp.diag(1, 2, 1), M([0, 1, 5])
    gram = design.T * weight * design
    beta = gram.inv() * design.T * weight * response
    checked(_CASES[13]["id"], "weighted solution", beta, M([R(25, 19), R(33, 19)]))
    residual = response - design * beta
    checked(
        _CASES[13]["id"], "weighted orthogonality", design.T * weight * residual, sp.zeros(2, 1)
    )
    checked(_CASES[13]["id"], "weighted SSE", (residual.T * weight * residual)[0], R(8, 19))
    observed = M([[12, 18, 30], [18, 12, 30]])
    expected = M([[15, 15, 30], [15, 15, 30]])
    checked(
        _CASES[14]["id"],
        "Pearson statistic",
        sum((o - e) ** 2 / e for o, e in zip(observed, expected, strict=True)),
        R(12, 5),
    )
    posterior = 60 * x**3 * (1 - x) ** 2
    checked(_CASES[15]["id"], "posterior integral", sp.integrate(posterior, (x, 0, 1)), 1)
    checked(_CASES[15]["id"], "posterior mean", sp.integrate(x * posterior, (x, 0, 1)), R(4, 7))
    checked(_CASES[15]["id"], "interior mode derivative", sp.diff(posterior, x).subs(x, R(3, 5)), 0)
    d = [sp.Integer(z) for z in [1, 2, 4, 5, 3, 3]]
    checked(_CASES[16]["id"], "paired unbiased variance", sum((v - 3) ** 2 for v in d) / 5, 2)
    if not 3 * sp.sqrt(3) > R(2571, 1000):
        raise ValueError("Paired test decision mismatch")
    evidence[_CASES[16]["id"]].append("test statistic exceeds supplied positive critical value")
    lam = sp.symbols("lambda", positive=True)
    sum_density = lam**5 * x**4 * sp.exp(-lam * x) / sp.factorial(4)
    checked(
        _CASES[17]["id"],
        "unbiased rate identity",
        sp.integrate(4 / x * sum_density, (x, 0, sp.oo)),
        lam,
    )
    z, kappa = sp.symbols("z kappa", real=True)
    potential = kappa * (x * x - 2 * y * y + z * z)
    field = -M([sp.diff(potential, c) for c in [x, y, z]])
    checked(
        _CASES[18]["id"],
        "field value",
        field.subs({x: 1, y: -1, z: 2, kappa: 3}),
        M([-6, -12, -12]),
    )
    checked(
        _CASES[18]["id"],
        "zero divergence",
        sum(sp.diff(field[i], c) for i, c in enumerate([x, y, z])),
        0,
    )
    resistance, inductance, capacitance, omega = sp.Integer(12), R(2, 25), R(1, 800), sp.Integer(50)
    impedance = resistance + sp.I * (omega * inductance - 1 / (omega * capacitance))
    checked(_CASES[19]["id"], "impedance", impedance, 12 - 12 * sp.I)
    checked(
        _CASES[19]["id"],
        "damping ratio",
        resistance / 2 * sp.sqrt(capacitance / inductance),
        R(3, 4),
    )
    flux = sp.pi * t * t / 50
    checked(_CASES[20]["id"], "Faraday sign and derivative", -sp.diff(flux, t), -sp.pi * t / 25)
    length = sp.symbols("L", positive=True)
    psi = sp.sqrt(2 / length) * sp.sin(sp.pi * x / length)
    checked(_CASES[21]["id"], "well normalization", sp.integrate(psi**2, (x, 0, length)), 1)
    checked(
        _CASES[21]["id"],
        "well second moment",
        sp.integrate(x * x * psi**2, (x, 0, length)),
        length**2 * (R(1, 3) - 1 / (2 * sp.pi**2)),
    )
    checked(
        _CASES[21]["id"],
        "well second derivative",
        sp.diff(psi, x, 2),
        -((sp.pi / length) ** 2) * psi,
    )
    checked(_CASES[22]["id"], "first lens image", 1 / (R(1, 12) - R(1, 18)), 36)
    checked(_CASES[22]["id"], "second lens image", 1 / (R(1, 8) - R(1, 24)), 12)
    amplitude, wave_omega, density, speed = R(3, 1000), 80 * sp.pi, R(1, 50), sp.Integer(20)
    checked(
        _CASES[23]["id"],
        "mean wave power",
        density * wave_omega**2 * amplitude**2 * speed / 2,
        36 * sp.pi**2 / 3125,
    )
    return {
        "version": VERSION,
        "status": "passed",
        "scope": "source-level exact arithmetic and selected symbolic derivations; no rendered-content or author-style claim",
        "independent_human_review": "not_performed",
        "cases": [
            {"id": row["id"], "checks": evidence[row["id"]], "status": "passed"} for row in _CASES
        ],
    }


def write_inputs(
    dest: Path, split: str = "development", profile: str = "personal@v1"
) -> list[Path]:
    """Create new existing-Job-schema inputs, refusing overwrite and never rendering."""
    from hwsol.jobs.schema import Job

    selected = cases(split)
    verify_references()
    destinations = [dest / row["id"] for row in selected]
    if any(p.exists() for p in destinations):
        raise FileExistsError("Benchmark case directories cannot be overwritten")
    manifests = []
    for row, folder in zip(selected, destinations, strict=True):
        folder.mkdir(parents=True)
        (folder / "problem.txt").write_text(row["statement"] + "\n", encoding="utf-8")
        (folder / "solution.md").write_text(row["solution"], encoding="utf-8")
        if row["background"]["kind"] == "image":
            from PIL import Image, ImageDraw

            paper = Image.new("RGB", (1240, 1754), (255, 253, 248))
            draw = ImageDraw.Draw(paper)
            for y in range(140, 1650, 70):
                draw.line((65, y, 1175, y), fill=(224, 231, 235), width=1)
            paper.save(folder / "custom-paper.png", dpi=(150, 150))
        job = {
            "schema_version": 1,
            "backend": "font-cpu",
            "realism": "combined",
            "generated_by": AUTHOR,
            "language": "en",
            "profile": profile,
            "background": row["background"],
            "render": {
                "dpi": 300,
                "seed": 271120,
                "font_size_pt": 11,
                "line_spacing": 1.6,
                "glyph_metrics": "raw",
                "semantic_layout": True,
                "math_strokes": "personal",
            },
            "problems": [
                {
                    "id": row["id"],
                    "number": row["id"],
                    "statement": row["statement"],
                    "sources": [
                        {
                            "path": "problem.txt",
                            "location": "newly Agent-authored M1 benchmark",
                            "sha256": hashlib.sha256(
                                (folder / "problem.txt").read_bytes()
                            ).hexdigest(),
                        }
                    ],
                    "solution": "solution.md",
                    "verification": {
                        "status": "passed",
                        "evidence": "Exact source-level arithmetic and symbolic checks executed by cases.verify_references; rendered content and author style are evaluated separately.",
                        "checks": row["numeric_checks"],
                    },
                }
            ],
        }
        validated = Job.model_validate(job)
        path = folder / "job.json"
        path.write_text(
            json.dumps(validated.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (folder / "case.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifests.append(path)
    return manifests


def inventory() -> dict:
    """Stable inventory without exposing held problem contents in a summary."""
    assert len(_CASES) == 24 and len({c["id"] for c in _CASES}) == 24
    assert Counter(c["split"] for c in _CASES) == {"development": 16, "holdout": 8}
    assert Counter(c["domain"] for c in _CASES) == dict.fromkeys(
        ["linear-algebra", "calculus", "statistics", "physics"], 6
    )
    return {
        "version": VERSION,
        "cases": 24,
        "development": 16,
        "holdout": 8,
        "expected": dict(Counter(c["expected"]["outcome"] for c in _CASES)),
        "end_to_end_original_inputs": 4,
        "source": "newly Agent-authored; not original historical homework or new writer samples",
    }
