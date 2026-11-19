import json

import pytest

from hwsol.profiles.io import sha256
from research.corpus.context_dataset import completeness_proof, latex_structure


def test_latex_balance_does_not_change_actual_semantics():
    report = latex_structure(r"\begin{aligned}x&=\frac{1}{2}\\y&=-x\end{aligned}")
    assert report["balanced_braces_environments"]
    assert not report["semantic_equivalence_verified"]
    assert not latex_structure(r"\frac{1}{2")["balanced_braces_environments"]
    assert not latex_structure(r"\begin{matrix}x\end{aligned}")["balanced_braces_environments"]


def test_completeness_amendment_bound_to_exact_sealed_parent(tmp_path, monkeypatch):
    from research.corpus import context_dataset

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "run.json").write_text('{"immutable":1}')
    amendment = tmp_path / "amendment"
    (amendment / "reports").mkdir(parents=True)
    (amendment / "run.json").write_text("{}")
    proof = {
        "model": "gpt-6.2-sol",
        "reasoning_effort": "xhigh",
        "human_review": False,
        "parents": [
            {
                "run": str(parent),
                "sha256": sha256(parent / "run.json"),
                "complete_samples": ["one"],
                "rejected_samples": [],
            }
        ],
    }
    (amendment / "reports/amendment.json").write_text(json.dumps(proof))
    monkeypatch.setattr(context_dataset, "verify_run", lambda _: None)
    assert completeness_proof(parent, amendment)["complete"] == {"one"}
    (parent / "run.json").write_text('{"immutable":2}')
    with pytest.raises(ValueError, match="parent changed"):
        completeness_proof(parent, amendment)
