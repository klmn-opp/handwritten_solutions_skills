"""Corpus safety tests use synthetic fixtures, never counted as personal research data."""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter

from research.corpus.dataset import export_dataset, frozen_splits, split_for
from research.corpus.ingest import import_file
from research.corpus.store import Conflict, Corpus
from research.corpus.train import validate_dataset
from research.corpus.validation import latex_structure


@pytest.fixture
def corpus(tmp_path):
    image = Image.new("RGB", (120, 90), "white")
    ImageDraw.Draw(image).line([(10, 60), (50, 10), (90, 70)], fill="black", width=4)
    path = tmp_path / "source.png"
    image.save(path)
    c = Corpus(tmp_path / "corpus")
    import_file(c, path, course="fixture-only", batch="fixture-session")
    return c, path


def sample(c):
    return c.propose(
        c.rows("page")[0]["page_id"],
        [5, 5, 100, 80],
        content_type="glyph",
        label="A",
        role="personal-handwriting",
    )


def verify(c, s, **changes):
    return c.review(
        s["sample_id"],
        s["revision"],
        {
            "review_status": "verified",
            "privacy_checked": True,
            "quality_status": "clean",
            **changes,
        },
        "test-fixture-reviewer",
        {
            "actual_transcription": True,
            "crop_complete": True,
            "writer_checked": True,
            "human_review": True,
        },
    )


def test_source_immutable_dedup_and_provenance(corpus):
    c, path = corpus
    before = path.read_bytes()
    d = import_file(c, path, course="fixture", batch="second-import")
    assert d["deduplicated"] and len(c.rows("document")) == 1
    assert path.read_bytes() == before == c.path(d["original"]).read_bytes()
    assert c.path(d["original"]).stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match="provenance"):
        import_file(c, path, course="fixture", batch="x", domain="model-generated")
    assert len(c.history(d["document_id"])) >= 2


def test_atomic_assets_and_traversal(corpus):
    c, _ = corpus
    with pytest.raises(ValueError):
        c.path("../../etc/passwd")
    content = b"{}"
    assert c.asset(content, ".json") == c.asset(content, ".json")
    p = c.path(c.asset(content, ".json"))
    p.chmod(0o644)
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corruption"):
        c.asset(content, ".json")


def test_recognition_never_verifies_and_requires_matching_crop(corpus):
    c, _ = corpus
    s = sample(c)
    r = {
        "engine": "test-fixture",
        "version": "0",
        "text": "WRONG",
        "raw": {},
        "input_sha256": s["crop_sha256"],
    }
    c.add_recognition(s["sample_id"], r)
    assert c.get("sample", s["sample_id"])["candidate_text"] == "A"
    assert c.get("sample", s["sample_id"])["verified_text"] is None
    with pytest.raises(ValueError, match="match"):
        c.add_recognition(s["sample_id"], {**r, "input_sha256": "fake"})
    approved = verify(c, s)
    c.add_recognition(s["sample_id"], {**r, "text": "SECOND WRONG CANDIDATE"})
    assert c.get("sample", s["sample_id"]) == approved


def test_formula_missing_selection_preserves_reviewed_and_existing_candidates(corpus):
    from research.corpus.candidate_jobs import select_missing

    c, _ = corpus
    page_id = c.rows("page")[0]["page_id"]
    a = c.propose(page_id, [1, 1, 20, 20])
    b = c.propose(page_id, [21, 1, 40, 20])
    done = c.propose(page_id, [41, 1, 60, 20])
    word = c.propose(page_id, [61, 1, 80, 20], content_type="english-word")
    c.review(done["sample_id"], 0, {"review_status": "rejected"}, "fixture")
    c.add_recognition(
        b["sample_id"],
        {
            "engine": "unimernet",
            "version": "fixture",
            "text": "x",
            "raw": {},
            "input_sha256": b["crop_sha256"],
        },
    )
    assert select_missing(c, page_id) == [a["sample_id"]]
    assert word["sample_id"] not in select_missing(c, page_id)
    with pytest.raises(KeyError):
        select_missing(c, "nonexistent")


def test_formula_jobs_bound_concurrency_and_leave_labels_untouched(corpus, monkeypatch, tmp_path):
    import threading
    import time

    from research.corpus.candidate_jobs import CandidateJobs

    c, _ = corpus
    s = sample(c)
    jobs = CandidateJobs(c)
    monkeypatch.setattr(jobs, "available", lambda: True)
    started, release = threading.Event(), threading.Event()

    def fixture_recognize(*args, **kwargs):
        assert kwargs["candidate_count"] == 3
        started.set()
        assert release.wait(3)
        return tmp_path / "fixture-only-not-inference"

    monkeypatch.setattr("research.corpus.candidate_jobs.recognize", fixture_recognize)
    j = jobs.start(sample_id=s["sample_id"])
    assert started.wait(2)
    try:
        with pytest.raises(ValueError, match="运行"):
            jobs.start(sample_id=s["sample_id"])
        with pytest.raises(ValueError, match="Choose"):
            jobs.start(sample_id=s["sample_id"], page_id=s["page_id"])
    finally:
        release.set()
    for _ in range(100):
        if jobs.get(j["id"])["status"] != "running":
            break
        time.sleep(0.01)
    assert jobs.get(j["id"])["status"] == "complete"
    assert c.get("sample", s["sample_id"]) == s


def test_candidate_count_rejected_before_loading_model(corpus):
    from research.corpus.recognition import recognize

    c, _ = corpus
    for count in [0, 4, True, "3"]:
        with pytest.raises(ValueError, match="Candidate count"):
            recognize(c, "unimernet", [], candidate_count=count)


def test_formula_job_failure_is_visible_and_preserves_labels(corpus, monkeypatch):
    from research.corpus.candidate_jobs import CandidateJobs

    c, _ = corpus
    s = sample(c)
    jobs = CandidateJobs(c)
    jobs.jobs["fixture"] = {"id": "fixture", "status": "running", "count": 1}

    def fail(*args, **kwargs):
        raise RuntimeError("fixture CUDA unavailable")

    monkeypatch.setattr("research.corpus.candidate_jobs.recognize", fail)
    jobs._work("fixture", [s["sample_id"]])
    assert jobs.get("fixture")["status"] == "failed"
    assert "CUDA unavailable" in jobs.get("fixture")["error"]
    assert c.get("sample", s["sample_id"]) == s


def test_review_gates_conflicts_and_restart(corpus):
    c, _ = corpus
    s = sample(c)
    with pytest.raises(ValueError, match="attestation"):
        c.review(s["sample_id"], 0, {"review_status": "verified"}, "tester")
    approved = verify(c, s)
    assert approved["verified_text"] == "A"
    with pytest.raises(Conflict):
        c.review(s["sample_id"], 0, {"candidate_text": "B"}, "tester")
    restarted = Corpus(c.root)
    updated = restarted.review(
        s["sample_id"], 1, {"review_status": "needs-review", "candidate_text": "B"}, "tester"
    )
    assert updated["verified_text"] is None and len(restarted.history(s["sample_id"])) == 3


def test_merge_split_transaction_and_lineage(corpus):
    c, _ = corpus
    parent = verify(c, sample(c))
    with pytest.raises(ValueError):
        c.reshape(
            [{"id": parent["sample_id"], "revision": 1}],
            [[0, 0, 20, 20], [20, 20, 50, 40]],
            "tester",
        )
    assert len(c.rows("sample")) == 1
    children = c.reshape(
        [{"id": parent["sample_id"], "revision": 1}], [[5, 5, 40, 80], [40, 5, 100, 80]], "tester"
    )
    assert all(
        s["verified_text"] is None and s["parents"] == [parent["sample_id"]] for s in children
    )
    assert c.get("sample", parent["sample_id"])["review_status"] == "rejected"
    merged = c.reshape([{"id": s["sample_id"], "revision": 0} for s in children], None, "tester")
    assert len(merged) == 1 and merged[0]["bounding_box"] == parent["bounding_box"]


@pytest.mark.parametrize(
    "quality,defects,eligible",
    [
        ("clean", [], 1),
        ("readable-correction", ["crossed-out"], 0),
        ("unusable", ["occluded"], 0),
        ("clean", ["overwritten"], 0),
    ],
)
def test_damaged_ink_never_enters_generator_dataset(corpus, tmp_path, quality, defects, eligible):
    c, _ = corpus
    verify(c, sample(c), quality_status=quality, defects=defects)
    destination = tmp_path / "dataset"
    result = export_dataset(c, destination)
    assert result["eligible_samples"] == eligible
    assert json.loads((destination / "readiness.json").read_text())["decision"] == "NO-GO"
    validate_dataset(destination)
    with pytest.raises(FileExistsError):
        export_dataset(c, destination)


def test_candidate_zero_export_and_generated_exclusion(corpus, tmp_path):
    c, _ = corpus
    sample(c)
    assert export_dataset(c, tmp_path / "pending")["eligible_samples"] == 0
    s = c.rows("sample")[0]
    with c.db() as db:
        s["writer_provenance"] = {"domain": "model-generated", "is_generated": True}
        c._put(db, "sample", s["sample_id"], s)
    verify(c, s)
    assert export_dataset(c, tmp_path / "generated")["eligible_samples"] == 0


def test_math_preserves_written_error_and_checks_structure(corpus):
    c, _ = corpus
    s = sample(c)
    s = verify(c, s, content_type="math-expression", candidate_text="2+2=5")
    assert s["verified_latex"] == "2+2=5"  # Do not replace actual writing with correct answer.
    assert latex_structure(r"\frac{a}{b")["balanced"] is False
    assert latex_structure(r"\begin{matrix}a\end{aligned}")["balanced"] is False
    assert latex_structure(r"\sqrt{x_1^2}")["structures"] == ["radical", "superscript", "subscript"]


def test_partial_rotated_pdf_resume_and_coordinate_roundtrip(tmp_path):
    image = Image.new("RGB", (120, 200), "white")
    ImageDraw.Draw(image).rectangle((20, 30, 45, 75), fill="black")
    source = tmp_path / "source.pdf"
    image.save(source, resolution=72, save_all=True, append_images=[image])
    writer = PdfWriter()
    for page in PdfReader(source).pages:
        writer.add_page(page.rotate(90))
    rotated = tmp_path / "rotated.pdf"
    writer.write(rotated)
    c = Corpus(tmp_path / "corpus")
    d = import_file(c, rotated, course="fixture", batch="one", page_indices=[1])
    assert d["status"] == "partial" and d["rendered_page_count"] == 1
    first = c.rows("page")[0]
    a, b = np.array(first["raster_to_pdf"]), np.array(first["pdf_to_raster"])
    assert np.allclose(a @ b, np.eye(3))
    corners = a @ np.array([[0, first["width"], 0], [0, 0, first["height"]], [1, 1, 1]])
    assert np.all(corners[:2] >= -0.1)
    d = import_file(c, rotated, course="fixture", batch="one")
    assert d["status"] == "complete" and len(c.rows("page")) == 2
    assert c.get("page", first["page_id"]) == first


def test_duplicate_document_quarantine_and_session_split(corpus, tmp_path):
    c, path = corpus
    verify(c, sample(c))
    alternate = tmp_path / "other.tiff"
    Image.open(path).save(alternate)
    import_file(c, alternate, course="fixture", batch="different-session")
    result = export_dataset(c, tmp_path / "dups")
    assert result["eligible_samples"] == 0 and result["near_duplicate_pairs"] == 1
    assert split_for("a") == split_for("a")


def test_balanced_split_freezes_independent_groups_and_never_reshuffles(corpus):
    c, _ = corpus
    docs = {str(i): {"split_group": str(i)} for i in range(7)}
    plan = frozen_splits(c, docs, 902)
    assert set(plan["assignments"].values()) == {"train", "validation", "holdout"}
    docs["later"] = {"split_group": "later"}
    updated = frozen_splits(c, docs, 902)
    assert all(updated["assignments"][g] == split for g, split in plan["assignments"].items())
    with pytest.raises(ValueError):
        frozen_splits(c, docs, 903)


def test_isolated_training_worker_cannot_bypass_admission(corpus, tmp_path):
    from research.corpus.train_worker import admitted_request

    c, _ = corpus
    sample(c)
    destination = tmp_path / "unverified"
    export_dataset(c, destination)
    with pytest.raises(ValueError, match="without verified"):
        admitted_request(
            {"dataset": str(destination), "rows": [], "gate": {"decision": "GO"}, "epochs": 1}
        )


def test_studio_local_auth_persistence_and_revision_conflict(corpus):
    import http.client
    import re
    import threading

    from research.corpus.studio import server

    c, _ = corpus
    s = sample(c)
    service = server(c, port=0)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", service.server_port)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        html = response.read().decode()
        token = re.search(r'name="corpus-token" content="([^"]+)"', html)[1]
        request = {
            "sample_id": s["sample_id"],
            "revision": 0,
            "actor": "unit-fixture",
            "changes": {"candidate_text": "B", "review_status": "needs-review"},
        }
        headers = {"Content-Type": "application/json", "X-Corpus-Token": token}
        for extra, expected in [
            ({"X-Corpus-Token": "wrong"}, 403),
            ({"Origin": "https://external.example"}, 403),
            ({}, 200),
            ({}, 409),
        ]:
            connection.request("POST", "/api/review", json.dumps(request), {**headers, **extra})
            response = connection.getresponse()
            response.read()
            assert response.status == expected
        assert Corpus(c.root).get("sample", s["sample_id"])["candidate_text"] == "B"
        for body, extra, expected in [
            ({"sample_id": s["sample_id"]}, {"X-Corpus-Token": "wrong"}, 403),
            ({"sample_id": s["sample_id"], "python": "/untrusted"}, {}, 400),
            ({"sample_id": s["sample_id"], "page_id": s["page_id"]}, {}, 400),
        ]:
            connection.request("POST", "/api/recognize", json.dumps(body), {**headers, **extra})
            response = connection.getresponse()
            response.read()
            assert response.status == expected
        connection.request("GET", "/api/state", headers={"Host": "external.example"})
        response = connection.getresponse()
        response.read()
        assert response.status == 403
    finally:
        connection.close()
        service.shutdown()
        service.server_close()
        thread.join()
