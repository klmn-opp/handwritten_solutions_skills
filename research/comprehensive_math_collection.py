"""A blank, broad university math/physics collection booklet, separate from old O.

Printed prompts describe intended writing. They are never verified handwriting
labels; actual scans must retain their own content, RGB pixels, and coordinates.
"""

import logging
import shutil
import unicodedata
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from PIL import Image
from pypdf import PdfWriter

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root


def row(latex, symbols, use, roles="usual handwritten forms", *, kind="math", **extra):
    return {
        "intended_latex": latex,
        "symbols": [
            {"unicode": c, "codepoint": f"U+{ord(c):04X}", "name": unicodedata.name(c)}
            for c in dict.fromkeys(symbols)
            if not c.isspace()
        ],
        "use": use,
        "variant_roles": roles.split("; "),
        "writing_instruction": (
            "Write the displayed content once in your normal handwriting, with clear separation. "
            "Do not trace the printed contours. Preserve the displayed mathematical structure."
        ),
        "render_kind": kind,
        "verified_label": None,
        "review_status": "not-collected",
        **extra,
    }


def catalog():
    pages = []

    def page(title, rows, *, pitch=92, prompt_size=14):
        number = len(pages) + 1
        pages.append(
            {
                "session": f"O-FULL-{number:02d}",
                "page": number,
                "title": title,
                "row_pitch_pt": pitch,
                "prompt_size_pt": prompt_size,
                "rows": [
                    {"id": f"O-FULL-P{number:02d}-R{i:02d}", **r} for i, r in enumerate(rows, 1)
                ],
            }
        )

    page(
        "Greek lowercase: all 24 letters",
        [
            row(r"\alpha\quad\beta\quad\gamma\quad\delta", "αβγδ", "alpha / beta / gamma / delta"),
            row(
                r"\varepsilon\quad\zeta\quad\eta\quad\theta", "εζηθ", "epsilon / zeta / eta / theta"
            ),
            row(r"\iota\quad\kappa\quad\lambda\quad\mu", "ικλμ", "iota / kappa / lambda / mu"),
            row(
                r"\nu\quad\xi\quad o\quad\pi",
                "νξοπ",
                "nu / xi / omicron / pi",
                "Greek omicron has the conventional o-shaped mathematical form",
            ),
            row(r"\rho\quad\sigma\quad\tau\quad\upsilon", "ρστυ", "rho / sigma / tau / upsilon"),
            row(r"\varphi\quad\chi\quad\psi\quad\omega", "φχψω", "phi / chi / psi / omega"),
            row(
                r"\alpha_1+\beta_2=\gamma\quad\theta(t)=\omega t+\varphi",
                "αβγθωφ",
                "Greek letters in formulas",
            ),
        ],
    )
    page(
        "Greek uppercase: all 24 letters",
        [
            row(
                r"A\quad B\quad\Gamma\quad\Delta",
                "ΑΒΓΔ",
                "Alpha / Beta / Gamma / Delta",
                "Greek Alpha and Beta conventionally share Latin A and B shapes",
            ),
            row(
                r"E\quad Z\quad H\quad\Theta",
                "ΕΖΗΘ",
                "Epsilon / Zeta / Eta / Theta",
                "Greek Epsilon Zeta Eta conventionally share E Z H shapes",
            ),
            row(
                r"I\quad K\quad\Lambda\quad M",
                "ΙΚΛΜ",
                "Iota / Kappa / Lambda / Mu",
                "Greek Iota Kappa Mu conventionally share I K M shapes",
            ),
            row(
                r"N\quad\Xi\quad O\quad\Pi",
                "ΝΞΟΠ",
                "Nu / Xi / Omicron / Pi",
                "Greek Nu and Omicron conventionally share N and O shapes",
            ),
            row(
                r"P\quad\Sigma\quad T\quad\Upsilon",
                "ΡΣΤΥ",
                "Rho / Sigma / Tau / Upsilon",
                "Greek Rho and Tau conventionally share P and T shapes",
            ),
            row(
                r"\Phi\quad X\quad\Psi\quad\Omega",
                "ΦΧΨΩ",
                "Phi / Chi / Psi / Omega",
                "Greek Chi conventionally shares Latin X shape",
            ),
            row(
                r"\Delta x\quad\Sigma_{ij}\quad\Phi_B\quad\Omega_0",
                "ΔΣΦΩ",
                "Change / covariance / flux / frequency",
            ),
        ],
    )
    page(
        "Greek variants and easily confused pairs",
        [
            row(
                r"\epsilon\quad\varepsilon\quad e",
                "ϵεe",
                "Lunate epsilon / open epsilon / Latin e",
                "epsilon U+03F5; varepsilon U+03B5; use your usual distinct variants",
                printed_latex=r"ϵ\quad ε\quad e",
                printed_mapping=(
                    "STIX MathText aliases epsilon/varepsilon to the same open glyph; "
                    "explicit U+03F5 and U+03B5 preserve the intended distinct variant roles"
                ),
            ),
            row(
                r"\phi\quad\varphi\quad\Phi",
                "ϕφΦ",
                "Straight phi / curly phi / capital Phi",
                "phi U+03D5; varphi U+03C6; capital Phi U+03A6",
            ),
            row(
                r"\theta\quad\vartheta\quad\Theta", "θϑΘ", "Theta / alternate theta / capital Theta"
            ),
            row(r"\rho\quad\varrho\quad p", "ρϱp", "Rho / alternate rho / Latin p"),
            row(r"\pi\quad\varpi\quad\Pi", "πϖΠ", "Pi / alternate pi / capital Pi"),
            row(r"\sigma\quad\varsigma\quad\Sigma", "σςΣ", "Sigma / final sigma / capital Sigma"),
            row(
                r"\kappa\quad\varkappa\quad\nu\quad v",
                "κϰνv",
                "Kappa variants / Greek nu / Latin v",
            ),
        ],
    )
    page(
        "Digits and lowercase Latin: part 1",
        [
            row(r"0\quad1\quad2\quad3\quad4", "01234", "Individual digits"),
            row(r"5\quad6\quad7\quad8\quad9", "56789", "Individual digits"),
            row(r"0.05\quad-1.20\quad3.1416", "0123456.−", "Decimals / negative sign / zero"),
            row(r"a\quad b\quad c\quad d\quad e", "abcde", "Lowercase Latin variables"),
            row(r"f\quad g\quad h\quad i\quad j", "fghij", "Lowercase Latin variables"),
            row(r"k\quad l\quad m\quad n\quad o", "klmno", "Lowercase Latin variables"),
            row(r"p\quad q\quad r\quad s\quad t", "pqrst", "Lowercase Latin variables"),
        ],
    )
    page(
        "Latin letters: remaining lowercase and capitals",
        [
            row(r"u\quad v\quad w\quad x\quad y\quad z", "uvwxyz", "Remaining lowercase variables"),
            row(r"A\quad B\quad C\quad D\quad E", "ABCDE", "Capital Latin variables"),
            row(r"F\quad G\quad H\quad I\quad J", "FGHIJ", "Capital Latin variables"),
            row(r"K\quad L\quad M\quad N\quad O", "KLMNO", "Capital Latin variables"),
            row(r"P\quad Q\quad R\quad S\quad T", "PQRST", "Capital Latin variables"),
            row(r"U\quad V\quad W\quad X\quad Y\quad Z", "UVWXYZ", "Remaining capital variables"),
            row(
                r"0\quad O\quad o\quad1\quad I\quad l",
                "0Oo1Il",
                "Zero / letters O,o / one / letters I,l",
            ),
        ],
    )
    page(
        "Numbers and variables in real contexts",
        [
            row(r"x_0=0\quad x_1=1\quad x_2=2", "x012=", "Digits with subscripts"),
            row(r"2x+3y=7\quad z=-4", "xyz2347+=−", "Short equations"),
            row(r"f(x)=x^2\quad g(t)=e^{-t}", "fgxte2=−", "Functions and exponents"),
            row(r"r=0.5\quad d=2r\quad k=10", "rdk0125.=", "r / d / k in natural spacing"),
            row(r"f_i\quad f_{\rm avg}\quad F=ma", "fiavgFma=", "f in small scripts and full size"),
            row(r"3.0\times10^{-4}\quad6.02\times10^{23}", "012346.×−", "Scientific notation"),
            row(
                r"5\,\mathrm{m/s}\quad12\,\mathrm{V}\quad4\,\Omega",
                "1245msVΩ/",
                "Units and variable spacing",
            ),
        ],
    )
    page(
        "Arithmetic and basic relations",
        [
            row(r"+\quad-\quad\pm\quad\mp", "+−±∓", "Plus / minus / plus-minus / minus-plus"),
            row(r"\times\quad\cdot\quad *\quad\div", "×⋅*÷", "Multiplication variants / division"),
            row(
                r"=\quad\ne\quad\equiv\quad :=",
                "=≠≡:=",
                "Equality / inequality / identity / definition",
            ),
            row(r"<\quad>\quad\leq\quad\geq", "<>≤≥", "Ordered comparisons"),
            row(
                r"\ll\quad\gg\quad\lesssim\quad\gtrsim",
                "≪≫≲≳",
                "Much less / greater / approximate bounds",
            ),
            row(
                r"\approx\quad\simeq\quad\sim\quad\cong",
                "≈≃∼≅",
                "Approximation / asymptotic / congruence",
            ),
            row(
                r"\propto\quad\infty\quad!\quad\%",
                "∝∞!%",
                "Proportionality / infinity / factorial / percent",
            ),
        ],
    )
    page(
        "Geometry and algebraic operators",
        [
            row(
                r"\perp\quad\parallel\quad\angle",
                "⊥∥∠",
                "Perpendicular / parallel / angle",
                printed_latex=r"⊥\quad\parallel\quad\angle",
                printed_mapping="Explicit U+22A5 for the catalog perpendicular role",
            ),
            row(r"30^{\circ}\quad90^{\circ}\quad180^{\circ}", "01389°", "Angle degrees"),
            row(
                r"\circ\quad\bullet\quad\star",
                "∘•⋆",
                "Composition / dot / star",
                printed_latex=r"\circ\quad•\quad\star",
                printed_mapping="Explicit U+2022 round bullet; avoid MathText U+2219 dot alias",
            ),
            row(
                r"\oplus\quad\otimes\quad\odot", "⊕⊗⊙", "Direct sum / tensor product / circled dot"
            ),
            row(
                r"a\mid b\quad a\nmid b\quad a\equiv b\pmod{n}",
                "abn∣∤≡",
                "Divisibility / modular arithmetic",
                printed_latex=r"a\mid b\quad a\nmid b\quad a\equiv b\;(\mathrm{mod}\;n)",
                printed_mapping="Explicit equivalent printing of unsupported MathText pmod macro",
            ),
            row(r"A^T\quad A^{-1}\quad A^{\dagger}", "AT1−†", "Transpose / inverse / adjoint"),
            row(
                r"\det A\quad\operatorname{tr}A\quad\operatorname{rank}A",
                "Adetrank",
                "Matrix operators in handwriting",
            ),
        ],
    )
    page(
        "Sets, membership, and number systems",
        [
            row(r"\in\quad\notin\quad\ni", "∈∉∋", "Membership / non-membership / contains"),
            row(
                r"\subset\quad\subseteq\quad\nsubseteq", "⊂⊆⊈", "Subset / inclusion / not included"
            ),
            row(
                r"\supset\quad\supseteq\quad\subsetneq",
                "⊃⊇⊊",
                "Superset / inclusion / strict subset",
            ),
            row(
                r"\cup\quad\cap\quad\setminus\quad\triangle",
                "∪∩∖△",
                "Union / intersection / difference / symmetric difference",
            ),
            row(
                r"\emptyset\quad\varnothing\quad\mathcal{P}(A)",
                "∅𝒫A",
                "Empty set (twice) / power set",
                "emptyset and varnothing are synonymous print forms; "
                "supply your natural empty-set variants if you use them",
                printed_mapping=(
                    "Both empty-set macros print U+2205; not two distinct font variants"
                ),
            ),
            row(
                r"\mathbb{N}\quad\mathbb{Z}\quad\mathbb{Q}\quad\mathbb{R}\quad\mathbb{C}",
                "ℕℤℚℝℂ",
                "Natural / integer / rational / real / complex",
                "Use your own blackboard-letter habit; "
                "keep these distinct from ordinary Latin letters",
            ),
            row(
                r"|A|\quad A^c\quad A\times B\quad A\triangle B",
                "ABc|×△",
                "Cardinality / complement / product / difference",
            ),
        ],
    )
    page(
        "Logic and arrows",
        [
            row(
                r"\forall\quad\exists\quad\nexists\quad\neg",
                "∀∃∄¬",
                "For all / exists / not exists / negation",
            ),
            row(
                r"\land\quad\lor\quad\top\quad\bot",
                "∧∨⊤⊥",
                "And / or / true / false",
                printed_latex=r"\wedge\quad\vee\quad\top\quad\bot",
                printed_mapping="Logical land/lor use equivalent wedge/vee MathText aliases",
            ),
            row(
                r"\to\quad\leftarrow\quad\leftrightarrow",
                "→←↔",
                "Forward / reverse / two-way arrows",
            ),
            row(
                r"\Rightarrow\quad\Leftarrow\quad\Leftrightarrow",
                "⇒⇐⇔",
                "Implication and equivalence",
            ),
            row(
                r"\mapsto\quad\hookrightarrow\quad\twoheadrightarrow",
                "↦↪↠",
                "Maps to / inclusion / surjection",
            ),
            row(
                r"\uparrow\quad\downarrow\quad\nearrow\quad\searrow",
                "↑↓↗↘",
                "Vertical and diagonal arrows",
            ),
            row(
                r"\vdash\quad\models\quad\therefore\quad\because",
                "⊢⊨∴∵",
                "Derivation / models / therefore / because",
                printed_latex=r"\vdash\quad⊨\quad\therefore\quad\because",
                printed_mapping="Explicit catalog U+22A8 semantic consequence reference",
            ),
        ],
    )
    page(
        "Summation, products, and integrals",
        [
            row(
                r"\sum\quad\sum_{i=1}^{n}x_i",
                "∑inx1",
                "Sum: isolated and indexed",
                row_height_pt=104,
            ),
            row(
                r"\prod\quad\prod_{k=1}^{n}a_k",
                "∏kna1",
                "Product: isolated and indexed",
                row_height_pt=104,
            ),
            row(r"\int\quad\int_0^1 f(x)\,dx", "∫01fxd", "Integral: isolated and bounded"),
            row(
                r"\iint_D f\,dx\,dy\quad\iiint_V g\,dV", "∬∭DVfgdxy", "Double and triple integrals"
            ),
            row(r"\oint_C\mathbf{F}\cdot d\mathbf{r}", "∮CFdr⋅", "Closed curve integral"),
            row(
                r"∯_S\mathbf{E}\cdot d\mathbf{S}\quad∰_V f\,dV",
                "∯∰SEVfd⋅",
                "Closed surface and closed triple operators",
                "Optional less-frequent operator forms; use your normal course notation",
            ),
            row(r"\int_{-\infty}^{\infty}e^{-x^2}\,dx", "∫∞exd2−", "Tall integral and limits"),
        ],
    )
    page(
        "Differentials and physical notation",
        [
            row(
                r"dx\quad dy\quad dt\quad\partial x\quad\partial t",
                "dxyt∂",
                "Ordinary and partial differentials",
            ),
            row(r"\frac{df}{dx}\quad\frac{d^2x}{dt^2}", "dfxt2", "First and second derivatives"),
            row(
                r"\frac{\partial f}{\partial x}\quad\frac{\partial^2 f}{\partial x\partial y}",
                "∂fxy2",
                "Partial derivatives",
            ),
            row(
                r"\nabla f\quad\nabla\cdot\mathbf{F}\quad\nabla\times\mathbf{F}",
                "∇fF⋅×",
                "Gradient / divergence / curl",
            ),
            row(
                r"\Delta x\quad\delta x\quad\delta_{ij}",
                "Δδxij",
                "Finite change / variation / Kronecker delta",
            ),
            row(
                r"\lim_{x\to0}f(x)\quad\lim_{n\to\infty}a_n",
                "limxfna0→∞",
                "Limits and approach arrows",
            ),
            row(
                r"\hbar\quad\ell\quad\Re z\quad\Im z",
                "ℏℓℜℑz",
                "Reduced Planck constant / ell / real and imaginary parts",
                printed_latex=r"ℏ\quad\ell\quad\Re z\quad\Im z",
                printed_mapping=(
                    "Explicit U+210F Planck symbol; MathText hbar otherwise uses U+0127"
                ),
            ),
        ],
    )
    page(
        "Delimiters and norms at useful heights",
        [
            row(
                r"(x)\quad[x]\quad\{x\}\quad\langle x,y\rangle",
                "xy()[]{}⟨⟩",
                "Short delimiters and angle brackets",
            ),
            row(
                r"\left(\frac{a+b}{c+d}\right)\quad\left[\frac{1}{1+x^2}\right]",
                "abcdx12+()[]",
                "Tall round and square delimiters",
            ),
            row(
                r"\left\{\frac{x^2+1}{x+1}\right\}\quad\left\langle\frac{u}{v},w\right\rangle",
                "xuvw12+{},⟨⟩",
                "Tall braces and angle brackets",
            ),
            row(
                r"|x|\quad\|x\|\quad\left|\frac{a}{b}\right|\quad\left\|\frac{u}{v}\right\|",
                "xabuv|‖",
                "Absolute value and paired norm bars",
                "Short bars; tall bars; paired norm strokes as you naturally write them",
            ),
            row(
                r"\lfloor x\rfloor\quad\lceil x\rceil\quad\left[\frac{a}{b},\frac{c}{d}\right)",
                "xabcd⌊⌋⌈⌉[)",
                "Floor / ceiling / interval endpoints",
            ),
        ],
        pitch=126,
        prompt_size=18,
    )
    page(
        "Radicals: short, long, tall, and nested",
        [
            row(
                r"\sqrt{2}\quad\sqrt{x}\quad\sqrt{a+b}",
                "√2xab+",
                "Short radicals: ordinary writing",
            ),
            row(r"\sqrt{x_1^2+x_2^2+x_3^2+x_4^2}", "√x1234+", "Long radical over a full sum"),
            row(r"\sqrt{\frac{a^2+b^2}{c^2+d^2}}", "√abcd2+", "Tall radical over a fraction"),
            row(
                r"\sqrt[3]{x}\quad\sqrt[4]{a+b}\quad\sqrt[n]{y}",
                "√xyabn34+",
                "Cube / fourth / general index roots",
            ),
            row(r"\sqrt{1+\sqrt{1+x^2}}", "√1x2+", "Nested radicals with clear joins"),
        ],
        pitch=126,
        prompt_size=18,
    )
    page(
        "Vectors, accents, and functional symbols",
        [
            row(r"\vec{x}\quad\vec{v}\quad\vec{F}", "xvF→", "Vector arrows over letters"),
            row(
                r"\hat{x}\quad\hat{n}\quad\widehat{AB}",
                "xnABˆ",
                "Hat: single letter and wider span",
            ),
            row(
                r"\bar{x}\quad\bar{z}\quad\overline{AB}",
                "xzAB¯",
                "Bar: single letter and wider span",
            ),
            row(
                r"\dot{x}\quad\ddot{x}\quad\dot{\theta}\quad\ddot{\theta}",
                "xθ˙¨",
                "First and second time derivatives",
            ),
            row(r"f'\quad f''\quad f'''\quad x'", "fx′″‴", "Primes: one, two, and three"),
            row(
                r"\tilde{x}\quad\widetilde{AB}\quad\breve{x}", "xAB˜˘", "Tilde: short / long; breve"
            ),
            row(
                r"\mathcal{F}\quad\mathcal{L}\quad\mathcal{H}\quad\mathcal{E}",
                "ℱℒℋℰ",
                "Calligraphic transform / space / energy letters",
            ),
        ],
    )
    page(
        "Subscripts, superscripts, and stacked indices",
        [
            row(r"x_i\quad x_{ij}\quad a_{n+1}", "xij an1+", "Single and multiple subscripts"),
            row(
                r"x^2\quad x^n\quad x^{-1}\quad e^{i\theta}", "xn2−1eiθ", "Superscripts and powers"
            ),
            row(
                r"a_i^2\quad b_{n+1}^{k-1}\quad x_{i,j}^{(t)}",
                "abxinkjt12+−(),",
                "Simultaneous upper and lower scripts",
            ),
            row(r"T_{ij}^{(k)}\quad H_0\quad H_1", "TijkH01()", "Tensor and statistical indices"),
            row(r"{}^{14}_{6}\mathrm{C}\quad{}^{A}_{Z}X", "146CAZX", "Left-hand isotope scripts"),
            row(
                r"\binom{n}{k}\quad P(A\mid B)",
                "nkPAB()∣",
                "Binomial coefficient / conditional probability",
            ),
            row(
                r"\sum_{i=1}^{n}x_i^2\quad\int_a^b f(x)\,dx",
                "∑∫in1x2abfd",
                "Operator limits and variable scripts",
                row_height_pt=104,
            ),
        ],
    )
    page(
        "Fractions and multi-level short formulas",
        [
            row(r"\frac{1}{2}\quad\frac{a+b}{c-d}", "12abcd+−", "Simple and algebraic fractions"),
            row(r"\frac{\frac{a}{b}}{\frac{c}{d}}", "abcd", "Fraction of fractions"),
            row(r"\frac{1}{1+\frac{1}{1+x}}", "1x+", "Short continued fraction"),
            row(
                r"\frac{-b\pm\sqrt{b^2-4ac}}{2a}",
                "abc±√24−",
                "Quadratic formula: connected structure",
            ),
            row(
                r"\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}",
                "∑inx¯12−()",
                "Variance: wide numerator and fraction bar",
            ),
        ],
        pitch=126,
        prompt_size=18,
    )
    page(
        "Matrices, determinants, and piecewise structure",
        [
            row(
                r"A=\begin{bmatrix}a&b\\c&d\end{bmatrix}",
                "Aabcd=[]",
                "Two-by-two square matrix",
                kind="matrix",
                matrix=[["a", "b"], ["c", "d"]],
                delimiters="[]",
                prefix="A=",
            ),
            row(
                r"B=\begin{pmatrix}1&0&x\\0&1&y\\0&0&1\end{pmatrix}",
                "Bxy01=()",
                "Three-by-three round matrix",
                kind="matrix",
                matrix=[["1", "0", "x"], ["0", "1", "y"], ["0", "0", "1"]],
                delimiters="()",
                prefix="B=",
            ),
            row(
                r"\begin{vmatrix}a&b\\c&d\end{vmatrix}=ad-bc",
                "abcd|=−",
                "Two-by-two determinant",
                kind="matrix",
                matrix=[["a", "b"], ["c", "d"]],
                delimiters="||",
                suffix="=ad-bc",
            ),
            row(
                r"f(x)=\begin{cases}x^2,&x\geq0\\-x,&x<0\end{cases}",
                "fx20≥<−()={",
                "Two-branch piecewise function",
                kind="cases",
                cases=[["x^2,", "x\\geq0"], ["-x,", "x<0"]],
                prefix="f(x)=",
            ),
        ],
        pitch=158,
    )
    page(
        "Additional notation for university assignments",
        [
            row(
                r"\ldots\quad\cdots\quad\vdots\quad\ddots",
                "…⋯⋮⋱",
                "Horizontal / centred / vertical / diagonal ellipses",
            ),
            row(
                r"\to\quad\longrightarrow\quad\leftrightarrow\quad\longleftrightarrow\quad\Longleftrightarrow",
                "→↔⟶⟷⟺",
                "Short and long single / double-headed arrows",
            ),
            row(
                r"\hat{\imath}\quad\hat{\jmath}\quad\hat{k}",
                "ıȷkˆ",
                "Cartesian basis vectors: i-hat / j-hat / k-hat",
            ),
            row(
                r"\mathbb{E}[X]\quad\mathbb{P}(A)",
                "𝔼ℙXA[]()",
                "Expectation and probability: blackboard letters",
            ),
            row(
                r"\mathcal{N}(\mu,\sigma^2)\quad\operatorname{Var}(X)\quad\operatorname{Cov}(X,Y)",
                "𝒩μσ2VarCovXY(),",
                "Normal distribution / variance / covariance",
            ),
            row(
                r"\overbrace{x_1+\cdots+x_n}^{n\;\mathrm{terms}}\quad"
                r"\underbrace{a+\cdots+a}_{m\;\mathrm{terms}}",
                "xn1am+⋯",
                "Overbrace and underbrace with explanatory labels",
                kind="braces",
                braces=[
                    {"body": r"x_1+\cdots+x_n", "label": r"n\;\mathrm{terms}", "position": "over"},
                    {"body": r"a+\cdots+a", "label": r"m\;\mathrm{terms}", "position": "under"},
                ],
                printed_mapping=(
                    "Controlled vector braces with exact MathText body/label subexpressions"
                ),
                row_height_pt=145,
            ),
        ],
        pitch=99,
    )
    return {
        "schema": "comprehensive-math-collection/1",
        "session": "O-FULL",
        "scope": (
            "Practical university mathematics and physics; not every mathematical Unicode symbol"
        ),
        "prompt_truth": (
            "intended only; actual handwriting, labels, RGB and transforms require review"
        ),
        "existing_symbols_may_be_recollected": True,
        "source_scan_suggestion": "source_pdf/session-O_full_source.pdf",
        "pages": pages,
    }


PAGE_SIZE = (595.44, 841.68)
PRINT_SIZE = 14


class PromptRenderer:
    """Print standard references with auditable math fonts and explicit structures."""

    def __init__(self, ax):
        self.ax = ax
        self.parser = MathTextParser("path")
        self.fonts = {}
        self.parses = []

    def measure(self, latex, size=PRINT_SIZE):
        parsed = self.parser.parse(
            f"${latex}$", dpi=72, prop=FontProperties(family="DejaVu Sans", size=size)
        )
        glyphs = []
        for glyph in parsed.glyphs:
            if len(glyph) == 6:
                font, glyph_size, codepoint, glyph_id, x, y = glyph
            elif len(glyph) == 5:
                font, glyph_size, codepoint, x, y = glyph
                glyph_id = font.get_char_index(codepoint)
            else:
                raise ValueError("Unrecognized MathText glyph tuple; cannot audit printed source")
            if font.get_char_index(codepoint) == 0:
                raise ValueError(f"Missing printed glyph U+{codepoint:04X}: {latex}")
            path = Path(font.fname).resolve()
            if str(path) not in self.fonts:
                self.fonts[str(path)] = sha256(path)
            glyphs.append(
                {
                    "font": str(path),
                    "codepoint": int(codepoint),
                    "glyph_id": int(glyph_id),
                    "size_pt": float(glyph_size),
                    "x_pt": float(x),
                    "y_pt": float(y),
                }
            )
        self.parses.append(
            {
                "latex": latex,
                "size_pt": size,
                "glyphs": glyphs,
                "width_pt": parsed.width,
                "height_pt": parsed.height,
            }
        )
        return float(parsed.width), float(parsed.height)

    def math(self, latex, x, top, size=PRINT_SIZE):
        self.measure(latex, size)
        return self.ax.text(x, top, f"${latex}$", fontsize=size, va="top", color="black")

    def path(self, vertices, codes):
        patch = PathPatch(MplPath(vertices, codes), fill=False, color="black", linewidth=0.8)
        self.ax.add_patch(patch)
        return patch

    def delimiter(self, char, x, bottom, top, *, opening):
        direction = 1 if opening else -1
        if char in "[]":
            vertices = [
                (x + 6 * direction, top),
                (x, top),
                (x, bottom),
                (x + 6 * direction, bottom),
            ]
            codes = [MplPath.MOVETO] + [MplPath.LINETO] * 3
        elif char == "|":
            vertices, codes = [(x, top), (x, bottom)], [MplPath.MOVETO, MplPath.LINETO]
        elif char in "()":
            vertices = [
                (x + 6 * direction, top),
                (x - 3 * direction, top - 8),
                (x - 3 * direction, bottom + 8),
                (x + 6 * direction, bottom),
            ]
            codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 3
        elif char == "{":
            middle = (top + bottom) / 2
            vertices = [
                (x + 7, top),
                (x, top),
                (x + 2, middle + 6),
                (x - 4, middle),
                (x + 2, middle - 6),
                (x, bottom),
                (x + 7, bottom),
            ]
            codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 6
        else:
            raise ValueError(f"Unsupported controlled delimiter {char}")
        return self.path(vertices, codes)

    def matrix(self, item, x, top):
        cells = item.get("matrix", item.get("cases"))
        prefix, suffix = item.get("prefix", ""), item.get("suffix", "")
        col_width = [max(self.measure(r[c])[0] for r in cells) for c in range(len(cells[0]))]
        row_height = 23
        height = len(cells) * row_height
        artists = []
        if prefix:
            width = self.measure(prefix)[0]
            artists.append(self.math(prefix, x, top - height / 2 + 9))
            x += width + 10
        opening = item.get("delimiters", "{")[0]
        artists.append(self.delimiter(opening, x, top - height, top, opening=True))
        cursor = x + 14
        for col, width in enumerate(col_width):
            for r, values in enumerate(cells):
                w = self.measure(values[col])[0]
                artists.append(
                    self.math(values[col], cursor + (width - w) / 2, top - r * row_height - 3)
                )
            cursor += width + 20
        right = cursor - 11
        if item["render_kind"] == "matrix":
            artists.append(
                self.delimiter(item["delimiters"][1], right, top - height, top, opening=False)
            )
            right += 12
        if suffix:
            artists.append(self.math(suffix, right, top - height / 2 + 9))
        return artists

    def horizontal_brace(self, left, right, y, direction):
        middle = (left + right) / 2
        vertices = [
            (left, y - direction * 4),
            (left, y),
            (left + 4, y),
            (left + 7, y),
            (middle - 7, y),
            (middle - 2, y),
            (middle - 2, y + direction * 4),
            (middle, y + direction * 4),
            (middle + 2, y + direction * 4),
            (middle + 2, y),
            (middle + 7, y),
            (right - 7, y),
            (right - 4, y),
            (right, y),
            (right, y - direction * 4),
        ]
        codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO]
        codes += [MplPath.CURVE4] * 6 + [MplPath.LINETO] + [MplPath.CURVE4] * 3
        return self.path(vertices, codes)

    def braces(self, item, x, top):
        artists = []
        for example in item["braces"]:
            body, label = example["body"], example["label"]
            width, _ = self.measure(body)
            label_width, _ = self.measure(label, 11)
            if example["position"] == "over":
                artists.append(self.math(label, x + (width - label_width) / 2, top, 11))
                artists.append(self.horizontal_brace(x, x + width, top - 20, 1))
                artists.append(self.math(body, x, top - 27))
            else:
                artists.append(self.math(body, x, top - 4))
                artists.append(self.horizontal_brace(x, x + width, top - 30, -1))
                artists.append(self.math(label, x + (width - label_width) / 2, top - 38, 11))
            x += max(width, label_width) + 70
        return artists


def bounds_pt(artists, canvas, ax):
    box = Bbox.union([a.get_window_extent(canvas.get_renderer()) for a in artists])
    return box.transformed(ax.transData.inverted()).extents.tolist()


def render_page(page, destination, total):
    fig = Figure(figsize=(PAGE_SIZE[0] / 72, PAGE_SIZE[1] / 72), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, PAGE_SIZE[0]), ylim=(0, PAGE_SIZE[1]))
    ax.axis("off")
    headers = [
        ax.text(
            42,
            808,
            f"Personal Math Writing / {page['session']} / {page['page']} of {total}",
            fontsize=13,
            va="top",
        ),
        ax.text(42, 786, "COMPREHENSIVE UNIVERSITY MATH / PHYSICS SAMPLES", fontsize=9, va="top"),
        ax.text(42, 769, page["title"], fontsize=10, va="top"),
        ax.text(
            42,
            752,
            "Write naturally on the blank line. Use your usual pen; do not trace the print.",
            fontsize=8,
            va="top",
        ),
        ax.text(
            42,
            42,
            "Keep original colour. Scan all pages together; "
            "the printed prompt is not a verified label.",
            fontsize=7.4,
            va="top",
        ),
    ]
    renderer = PromptRenderer(ax)
    elements, cursor = [], 730
    for item in page["rows"]:
        pitch = item.get("row_height_pt", page["row_pitch_pt"])
        labels = [
            ax.text(42, cursor, item["id"], fontsize=7.5, va="top"),
            ax.text(154, cursor, item["use"], fontsize=7.7, va="top"),
        ]
        if item["render_kind"] == "math":
            formula = [
                renderer.math(
                    item.get("printed_latex", item["intended_latex"]),
                    62,
                    cursor - 17,
                    page["prompt_size_pt"],
                )
            ]
        elif item["render_kind"] in {"matrix", "cases"}:
            formula = renderer.matrix(item, 62, cursor - 17)
        elif item["render_kind"] == "braces":
            formula = renderer.braces(item, 62, cursor - 17)
        else:
            raise ValueError("Unsupported collection prompt; never emit bare LaTeX")
        elements.append((item, cursor, pitch, labels, formula))
        cursor -= pitch
    canvas.draw()
    checks = []
    for item, top, pitch, labels, formula in elements:
        label_box, math_box = bounds_pt(labels, canvas, ax), bounds_pt(formula, canvas, ax)
        if not (
            42 - 1e-6 <= label_box[0] < label_box[2] <= 554
            and math_box[0] >= 41
            and math_box[2] <= 554
            and math_box[3] <= label_box[1] - 2
        ):
            raise ValueError(
                f"Printed prompt/annotation overlaps or clips: {item['id']} "
                f"label={label_box}, prompt={math_box}"
            )
        blank = [42, top - pitch + 10, 553, math_box[1] - 8]
        minimum_height = 40 if item["render_kind"] in {"matrix", "cases", "braces"} else 25
        if blank[3] - blank[1] < minimum_height or blank[1] < 60:
            raise ValueError(f"Insufficient natural writing space: {item['id']} {blank}")
        ax.plot([42, 553], [blank[1] + 4] * 2, linewidth=0.45, color="#c5c5c5")
        check = {
            "id": item["id"],
            "label_bounds_pdf_pt": label_box,
            "printed_prompt_bounds_pdf_pt": math_box,
            "answer_bbox_pdf_points": blank,
            "answer_height_pt": blank[3] - blank[1],
            "no_label_prompt_overlap": True,
            "no_page_clipping": True,
            "printed_math_missing_glyphs": 0,
        }
        item.update(check)
        item["printed_math_policy"] = (
            "MathText/STIX reference font"
            if item["render_kind"] == "math"
            else "controlled vector structure plus exact MathText/STIX subexpressions"
        )
        checks.append(check)
    for box in [bounds_pt(headers, canvas, ax)]:
        if box[0] < 40 or box[2] > 555 or box[1] < 25 or box[3] > 818:
            raise ValueError("Collection header/footer clipping")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"session-{page['session']}"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.clear()
    return {
        "page": page["page"],
        "session": page["session"],
        "rows": checks,
        "fonts": renderer.fonts,
        "printed_math_parses": renderer.parses,
        "pdf": path.with_suffix(".pdf").name,
        "png": path.with_suffix(".png").name,
    }


def launch():
    source = catalog()
    root = workspace_root()
    protected_paths = [
        root / "latest.json",
        root / "runs/20261120T105153402369Z-m1-optional-math-session-o-137ebfeb/run.json",
    ]
    protected = {str(p): sha256(p) for p in protected_paths}
    run = Run.create(
        "comprehensive-math-collection-o-full",
        config={
            "purpose": "blank author collection booklet; practical broad math/physics coverage",
            "pages": len(source["pages"]),
            "source_pdf_originals_read_or_written": False,
            "old_session_o_preserved": True,
            "actual_new_handwriting_samples": 0,
            "neural_updates": 0,
            "prompt_labels_verified": False,
            "source_sha256": sha256(Path(__file__)),
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/source.py")
        atomic_json(run.path / "reports/protected-assets.json", protected)
        captures = []

        class Capture(logging.Handler):
            def emit(self, record):
                captures.append(record.getMessage())

        handler = Capture(logging.WARNING)
        logger = logging.getLogger("matplotlib")
        logger.addHandler(handler)
        try:
            with mpl.rc_context(
                {
                    "mathtext.fontset": "stix",
                    "font.family": "DejaVu Sans",
                    "pdf.fonttype": 42,
                    "text.usetex": False,
                }
            ):
                reports = []
                for page in source["pages"]:
                    print(f"page {page['page']:02d}: {page['title']}", flush=True)
                    reports.append(
                        render_page(
                            page, run.path / "solutions/collection/pages", len(source["pages"])
                        )
                    )
        finally:
            logger.removeHandler(handler)
        atomic_json(run.path / "reports/font-warnings.json", captures)
        if captures:
            raise ValueError(f"Printed math font warning(s): {captures}")
        collection = run.path / "solutions/collection"
        with PdfWriter() as writer:
            for report in reports:
                writer.append(collection / "pages" / report["pdf"])
            writer.write(collection / "session-O_full.pdf")
        atomic_json(collection / "catalog.json", source)
        atomic_json(run.path / "reports/print-validation.json", reports)
        fonts = {key: value for r in reports for key, value in r["fonts"].items()}
        atomic_json(run.path / "reports/printed-fonts.json", fonts)
        text = [
            "# Session O-FULL: blank university mathematics / physics booklet",
            "",
            "19 pages; 123 prompts. Recollecting familiar symbols is intentional. This is not "
            "a claim to cover every mathematical Unicode symbol.",
            "",
            "Write each example naturally once in the blank area, using your usual pen. "
            "Do not trace the computer font. Keep spaces between separate samples. "
            "Use the normal size implied by each formula. You may add natural variants "
            "within the same row if there is room; do not force unfamiliar forms.",
            "",
            "Scan all completed pages in colour as source_pdf/session-O_full_source.pdf. "
            "Keep the original scan unchanged. The blank printed template must stay outside "
            "the original handwriting corpus. Retain page order and row IDs.",
            "",
            "Printed Unicode/LaTeX describes intended content only. Actual writing must be "
            "reviewed, with original PDF, 300-DPI RGB crops, coordinates/transforms and any "
            "different actual labels retained. No new handwriting or training is claimed.",
            "",
        ]
        for page in source["pages"]:
            text.extend([f"## {page['session']}: {page['title']}", ""])
            for item in page["rows"]:
                codes = ", ".join(f"{s['unicode']} ({s['codepoint']})" for s in item["symbols"])
                text.extend(
                    [
                        f"- **{item['id']}** — {item['use']}. `{item['intended_latex']}`",
                        f"  Unicode roles: {codes}. Variants: {'; '.join(item['variant_roles'])}.",
                    ]
                )
            text.append("")
        (collection / "README.md").write_text("\n".join(text), encoding="utf-8")
        # Overview only; all source page PNGs remain at 300 DPI and original PDF geometry.
        for start in range(0, len(reports), 4):
            canvas = Image.new("RGB", (1240, 1754), "#dddddd")
            for slot, report in enumerate(reports[start : start + 4]):
                with Image.open(collection / "pages" / report["png"]) as original:
                    preview = original.convert("RGB")
                    preview.thumbnail((610, 867), Image.Resampling.LANCZOS)
                    canvas.paste(preview, ((slot % 2) * 620 + 5, (slot // 2) * 877 + 5))
            canvas.save(run.path / f"comparisons/overview-{start // 4 + 1:02d}.png")
        if any(sha256(Path(p)) != value for p, value in protected.items()):
            raise ValueError("Existing collection or production receipt changed")
        result = {
            "pages": len(reports),
            "prompt_rows": sum(len(p["rows"]) for p in source["pages"]),
            "actual_new_handwriting_samples": 0,
            "neural_updates": 0,
            "printed_math_missing_glyphs": 0,
            "font_warnings": [],
            "all_prompt_note_bounds_checked": True,
            "page_clipping": False,
            "minimum_answer_height_pt": min(
                r["answer_height_pt"] for p in reports for r in p["rows"]
            ),
            "pdf": "solutions/collection/session-O_full.pdf",
            "author_prompt_truth": "not-collected",
            "visual_review": "pending independent Agent",
        }
        run.finish("held", result=result)
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    launch()
