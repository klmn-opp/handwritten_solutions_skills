# nextgen-v1 model card

Frozen next-generation handwriting variant model for the personal CPU default.
This bundle changes glyph shape only. It does not generate pages, prose or
mathematical structure, and it is not a general text-to-handwriting system.

## What ships

| Asset | Contents |
| --- | --- |
| `runtime/english.npz` | 34 float32 arrays of the accepted English endpoint |
| `runtime/math.npz` | 38 float32 arrays of the accepted mathematical endpoint and recognizer |
| `runtime/model.json` | Class lists, supports and frozen training configuration |
| `glyphs/english.npz` | 951 reviewed English base glyphs, canonical 80x80 grayscale |
| `glyphs/math-index.json` + `glyphs/*.png` | 755 reviewed mathematical atoms with role labels |
| `manifest.json` | Per-file SHA-256 and source checkpoint identities |

No optimizer state, no raw scans, no full training corpus, no private path and no
other checkpoint is included. The `.pt` files remain local; the runtime stores the
same arrays with identical values.

## Frozen endpoints

| Branch | Source checkpoint | SHA-256 |
| --- | --- | --- |
| English | `continued-00400.pt` (`5cfb0165`) | `e2bc61dd88afeb307796455ec11f51c1c9061087773c0dec54b09ee34b8e7189` |
| Mathematics | `math-variation-01200.pt` (`757afef8`) | `4814d3ff602c6cc447071957ca4b30c889fd03dc9a3bb4e9318a8f295c29aa85` |

The English branch composes a bounded coarse warp, a conditional latent prior and
a local decoder restricted to `f/t/h/r`. The mathematical branch predicts a 9x9
displacement field conditioned on the symbol and role, then samples the real base
glyph once with a bounded bilinear transform. Both branches were accepted on the
seven complex exercises; see the public demo.

## Inference

`hwsol.neural` reimplements the trained forward graph with NumPy only. The product
runtime does not import PyTorch. A separate maintainer check compares every array
with the checkpoint and compares Torch and NumPy outputs on identical latent codes:

- English samples: 20/20 bit-identical grayscale outputs.
- Mathematical fields: maximum absolute difference `2.15e-6`.
- Single-symbol recognizer: 16/16 agreeing labels.

Latent draws use a documented CPU generator. Per-seed pixels therefore need not
match the original Torch draws; the model distribution and weights are unchanged.

## Admission and fallback

A candidate replacement is admitted only when it introduces no new dark contact,
keeps the item's connected components and holes, stays inside the page, and (where
no personal anchor exists) passes the frozen identity recognizer. If a mathematical
base is an external or inherited font, the runtime uses the personal O atom and
falls back to that fitted O atom when the warp guard rejects; it refuses to leave a
foreign math source. Text marks with a personal O sample are also replaced; text
marks without one are reported in `text_foreign_sources`. Structure elements keep
the stable personal-CPU glyph. Supported variants without a direct O atom use a
documented personal equivalent (for example vartheta -> personal theta, blackboard
capitals -> personal plain capitals, combining accents -> reviewed spacing accents,
named bars -> literal delimiters), recorded as a variant alias rather than a
foreign font. Deterministic ink and integer word-spacing corrections run on
accepted pixels and preserve page-level contour counts.

## Data and licence

Base glyphs come from the author's reviewed Session P and Session O handwriting,
under the same authorization as the bundled personal Profile. Only the inference
subset is distributed. Model weights and glyph inputs are released under the
project's MIT terms; third-party runtime dependencies keep their own licences.