import sympy as s

from research.corpus.hard_homework import cases


def test_spectral_decomposition_and_regression_are_correct():
    a = s.Matrix([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    vectors = [
        s.Matrix([1, -s.sqrt(2), 1]) / 2,
        s.Matrix([1, 0, -1]) / s.sqrt(2),
        s.Matrix([1, s.sqrt(2), 1]) / 2,
    ]
    for lam, u in zip([2 - s.sqrt(2), 2, 2 + s.sqrt(2)], vectors, strict=True):
        assert s.simplify(a * u - lam * u) == s.zeros(3, 1)
    q = s.Matrix.hstack(*vectors)
    assert s.simplify(q.T * q) == s.eye(3)
    x = s.Matrix([[1, t] for t in [-2, -1, 0, 1, 2]])
    y = s.Matrix([-3, -3, 4, 3, 9])
    beta = (x.T * x).inv() * x.T * y
    e = y - x * beta
    assert beta == s.Matrix([2, 3]) and (e.T * e)[0] == 14
    assert ((x.T * x).inv() * s.Rational(14, 3))[1, 1] == s.Rational(7, 15)


def test_parameter_integral_and_rlc_initial_value_problem():
    a, b, t = s.symbols("a b t", positive=True)
    assert s.simplify(-s.diff(b / (a * a + b * b), a) - 2 * a * b / (a * a + b * b) ** 2) == 0
    assert s.integrate(s.exp(-2 * t) * s.sin(3 * t), (t, 0, s.oo)) == s.Rational(3, 13)
    assert s.integrate(t * s.exp(-2 * t) * s.sin(3 * t), (t, 0, s.oo)) == s.Rational(12, 169)
    w = s.sqrt(3) / 2
    q = 3 * (1 - s.exp(-t / 2) * (s.cos(w * t) + s.sin(w * t) / s.sqrt(3)))
    i = 2 * s.sqrt(3) * s.exp(-t / 2) * s.sin(w * t)
    assert s.simplify(s.diff(q, t) - i) == 0
    assert s.simplify(s.diff(q, t, 2) + i + q - 3) == 0
    assert q.subs(t, 0) == 0 and i.subs(t, 0) == 0
    assert len(cases()) == 4
    assert all(source.count("$$") >= 8 for source in cases().values())


def test_all_hard_cases_use_supported_latex_and_exact_text_groups(tmp_path):
    from hwsol.jobs.layout import validate_markdown
    from hwsol.profiles.store import ProfileStore
    from research.corpus.mixed_ink_pages import text_groups

    profile = ProfileStore(tmp_path / "profiles").resolve()
    for title, source in cases().items():
        blocks = validate_markdown(source, profile)
        texts = [title] + [v for b in blocks for kind, v in b.segments if kind == "t"]
        records = [
            {"context": "text", "semantic": ch, "item_index": i}
            for i, ch in enumerate("".join("".join(t.split()) for t in texts))
        ]
        groups = text_groups(source, title, records, profile)
        assert "".join(g["text"] for g in groups) == "".join(r["semantic"] for r in records)
        assert sorted(i for g in groups for i in g["items"]) == list(range(len(records)))


def test_changed_neighbours_cannot_meet_inside_an_original_gap():
    import numpy as np

    from research.corpus.mixed_ink_pages import collision_reason

    original = np.array([[10, 10, 20, 20], [40, 10, 50, 20]], dtype=float)
    first = np.array([18, 10, 32, 20])
    second = np.array([30, 10, 43, 20])
    assert collision_reason(first, original, [0], 100, 100) is None
    assert collision_reason(second, original, [1], 100, 100) is None
    occupied = original.copy()
    occupied[0] = first
    assert collision_reason(second, occupied, [1], 100, 100) == "would_overlap_neighbour"


def test_method_label_is_printable_without_annotation_support():
    import io

    import numpy as np
    import pypdfium2 as pdfium
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject

    from research.corpus.mixed_ink_pages import label_pdf

    writer = PdfWriter()
    page = writer.add_blank_page(595.2, 841.92)
    content = DecodedStreamObject()
    content.set_data(b"0 g 100 100 20 20 re f")
    page.replace_contents(content)
    raw = io.BytesIO()
    writer.write(raw)
    writer.close()
    labeled = label_pdf(raw.getvalue(), "expanded-data")
    reader = PdfReader(io.BytesIO(labeled))
    assert len(reader.pages) == 1
    assert "expanded-data" in reader.pages[0].extract_text()
    with pdfium.PdfDocument(raw.getvalue()) as doc:
        page = doc[0]
        original = np.asarray(page.render(scale=1).to_pil().convert("L"))
        page.close()
    with pdfium.PdfDocument(labeled) as doc:
        page = doc[0]
        pixels = np.asarray(page.render(scale=1, draw_annots=False).to_pil().convert("L"))
        page.close()
    assert (pixels[-30:] < 100).any()
    assert (original[:-30] < 100).any()
    assert np.array_equal(pixels[:-30], original[:-30])
