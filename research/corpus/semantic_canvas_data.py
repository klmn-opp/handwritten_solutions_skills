"""Frozen font-teacher content data; never counted as real scan observations."""

import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def prepare(teacher: Path, words: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(teacher.parent.parent)
    verify_run(words)
    cfg = {
        "teacher": str(teacher.resolve()),
        "teacher_sha256": digest(teacher),
        "words": str(words.resolve()),
        "words_sha256": digest(words / "run.json"),
        "new_real_observations": 0,
        "font_auxiliary_only": True,
        "C_used": False,
        "protocol_sha256": digest("docs/V01020_PROTOCOL.md"),
    }
    run = create("semantic-canvas-font-data", config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01020-font-data").write_text(str(run.path))
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    __name__,
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(path):
    import numpy as np
    import torch

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.semantic_ink_model import SemanticInk

    cfg = json.loads(Path(path).read_text())
    run, words = Path(cfg["run"]), Path(cfg["words"])
    assert digest(cfg["teacher"]) == cfg["teacher_sha256"]
    assert digest(words / "run.json") == cfg["words_sha256"]
    setup(4020)
    saved = torch.load(cfg["teacher"], map_location="cuda", weights_only=False)
    assert saved["step"] == 24000 and saved["config"]["arm"] == "identity-extended"
    model = SemanticInk(context_scale=0).cuda().eval()
    model.load_state_dict(saved["model"])
    source = Path(saved["config"]["data"])
    assert digest(source / "run.json") == saved["config"]["data_sha256"]
    wi = json.loads((words / "reports/indices.json").read_text())
    rows = json.loads((words / "reports/samples.json").read_text())
    with np.load(words / "reports/arrays.npz") as archive:
        a = {k: archive[k] for k in ["tokens", "boxes"]}
    with np.load(source / "reports/arrays.npz") as archive:
        ti = json.loads((source / "reports/indices.json").read_text())["train"]
        alphabet = sorted(set(int(c) for c in archive["tokens"][ti, :, 0].ravel()) - {0})
    t = np.zeros((len(alphabet), 2, 3), np.int64)
    t[:, 0, 0] = alphabet
    b = np.zeros((len(alphabet), 2, 4), np.float32)
    b[:, 0] = [16, 16, 80, 80]
    images = []
    for i in range(len(t)):
        im = model.image(
            torch.tensor(t[i : i + 1], device="cuda"),
            torch.tensor(b[i : i + 1], device="cuda"),
            96,
            96,
        )
        images.append(im[0].cpu().numpy())
    np.savez_compressed(run / "reports/font-glyphs.npz", tokens=t, boxes=b, ink=np.stack(images))
    for name, indices in [
        ("font-words", wi["auxiliary"]),
        ("evaluation", wi["development"] + wi["novel"]),
    ]:
        targets = []
        for i in indices:
            im = model.image(
                torch.tensor(a["tokens"][i : i + 1], device="cuda"),
                torch.tensor(a["boxes"][i : i + 1], device="cuda"),
                128,
                512,
            )
            targets.append(im[0].cpu().numpy())
        np.savez_compressed(
            run / f"reports/{name}.npz",
            tokens=a["tokens"][indices],
            boxes=a["boxes"][indices],
            ink=np.stack(targets),
        )
        write(
            run / f"reports/{name}.json",
            [
                {
                    "index": i,
                    "text": rows[i]["text"],
                    "source_type": "frozen-font-teacher",
                    "real_scan": False,
                    "teacher_sha256": cfg["teacher_sha256"],
                }
                for i in indices
            ],
        )
    write(run / "reports/alphabet.json", alphabet)
    write(
        run / "reports/result.json",
        {
            "font_glyphs": len(alphabet),
            "font_train_words": len(wi["auxiliary"]),
            "evaluation_words": 24,
            "new_neural_updates": 0,
            "real_training_samples": 0,
            "C_used": False,
        },
    )


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
