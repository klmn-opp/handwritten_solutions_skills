"""Load a hash-pinned frozen model bundle and expose Torch-free inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BUNDLE = "nextgen-v1"
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "models" / BUNDLE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    manifest: dict
    model: dict
    english: dict[str, np.ndarray]
    math: dict[str, np.ndarray]
    english_glyphs: dict[str, np.ndarray]
    math_glyphs: list[dict]

    def glyph_pool(self) -> dict[str, list[int]]:
        pools: dict[str, list[int]] = {}
        for index, label in enumerate(self.english_glyphs["labels"].tolist()):
            pools.setdefault(str(label), []).append(index)
        return pools

    def math_pool(self) -> dict[tuple[str, str], list[dict]]:
        pools: dict[tuple[str, str], list[dict]] = {}
        for row in self.math_glyphs:
            pools.setdefault((row["label"], row["role"]), []).append(row)
        return pools


_CACHE: dict[Path, ModelBundle] = {}


def load_bundle(path: Path | None = None, *, verify: bool = True) -> ModelBundle:
    root = Path(path) if path else ASSETS
    root = root.resolve()
    if root in _CACHE:
        return _CACHE[root]
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "nextgen-model-bundle/1":
        raise ValueError("Unsupported next-generation model bundle")
    if verify:
        for name, digest in manifest["files"].items():
            if sha256(root / name) != digest:
                raise ValueError(f"Model bundle file changed: {name}")
    english = dict(np.load(root / "runtime/english.npz"))
    math = dict(np.load(root / "runtime/math.npz"))
    glyphs = dict(np.load(root / "glyphs/english.npz"))
    index = json.loads((root / "glyphs/math-index.json").read_text())
    model = json.loads((root / "runtime/model.json").read_text())
    bundle = ModelBundle(root, manifest, model, english, math, glyphs, index)
    _CACHE[root] = bundle
    return bundle


def load_math_gray(bundle: ModelBundle, row: dict) -> np.ndarray:
    path = bundle.root / "glyphs" / row["file"]
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))
