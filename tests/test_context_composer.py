import io

import numpy as np
import pypdfium2 as pdfium
import pytest
from handnote.vector import VectorPage, write_pdf

from research.corpus.context_composer import mask_outline


def test_vectorization_preserves_dot_hole_and_disconnected_ink():
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:60, 20:60] = True
    mask[30:50, 30:50] = False
    mask[10, 40] = True
    mask[65:75, 35:40] = True
    outline = mask_outline(mask)
    page = VectorPage(96, 96, (0, 0, 0))
    page.items.append(("path", ("test",), (1, 0, 0, 1, 0, 0), 255, 0))
    blob = write_pdf([page], {("test",): outline}, dpi=72)
    with pdfium.PdfDocument(io.BytesIO(blob)) as doc:
        raw = doc[0].render(scale=1).to_pil().convert("L")
    actual = np.asarray(raw) < 128
    assert np.array_equal(actual, mask)


def test_empty_generated_outline_rejected():
    with pytest.raises(ValueError, match="Empty"):
        mask_outline(np.zeros((96, 96), dtype=bool))


def test_all_context_cases_use_supported_math_and_holdout_is_not_training(tmp_path):
    from hwsol.jobs.layout import validate_markdown
    from hwsol.profiles.store import ProfileStore
    from research.corpus.context_benchmark import manifest
    from research.corpus.context_homework import complete_homework

    profile = ProfileStore(tmp_path / "data").resolve()
    suite = manifest()
    assert len(suite["cases"]) == 32
    assert len({r["id"] for r in suite["cases"]}) == 32
    for row in suite["cases"]:
        assert row["training_allowed"] is False
        assert validate_markdown(row["source"], profile)
    for source in complete_homework().values():
        assert validate_markdown(source, profile)


def test_worked_homework_arithmetic_and_calculus():
    import sympy as s

    a = s.Matrix([[1, 1, 1], [2, -1, 1], [1, 2, -1]])
    assert a.inv() * s.Matrix([6, 3, 2]) == s.Matrix([1, 2, 3])
    x, t = s.symbols("x t", positive=True)
    assert s.integrate(x**-3, (x, 1, s.oo)) == s.Rational(1, 2)
    assert s.diff(-1 / (2 * x**2), x) == x**-3
    current = 2 * (1 - s.exp(-2 * t))
    assert s.simplify(3 * s.diff(current, t) + 6 * current - 12) == 0
    assert current.subs(t, 0) == 0
    residuals = [s.Rational(1, 6), -s.Rational(1, 3), s.Rational(1, 6)]
    assert sum(residuals) == 0
    assert sum(v**2 for v in residuals) == s.Rational(1, 6)
