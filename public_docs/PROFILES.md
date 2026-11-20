# Personal Profiles and CPU Studio

`bundled-default@v1` is the licensed Patrick Hand engineering font, not the
developer's handwriting. Use `hwsol profiles list` to inspect installed versions.
Profile resolution uses an explicitly requested version, then the user's active
default, then the bundled default. A clean installation has no private Profile.

Studio can build versioned fonts from user-provided templates, supported scans,
fonts, trajectories and calibration annotations. This deterministic font and
geometry construction is not neural-network training. Read each command's
`--help` for the supported input fields; retain the actual written label and
source, and do not treat an OCR suggestion or reference answer as ground truth.

```sh
hwsol profile template --output out/letters.pdf --chars "ABCabc123x" --sets 2
hwsol profile import --profile my-handwriting --source /path/to/scan.jpg --template out/letters.json
hwsol profile train --profile my-handwriting --append
hwsol profile evaluate --profile my-handwriting --version v1
hwsol profile publish --profile my-handwriting --version v1
hwsol profiles activate my-handwriting@v1
```

Inspect the candidate before publishing or activating it. A publish does not
implicitly authorize replacing a user's production default. New imports and
append builds create new versions; keep original samples, existing labels and
old versions. The separate `hwsol-studio` entry point exposes the same Studio.

```sh
hwsol profiles compare my-handwriting@v1 my-handwriting@v2
hwsol profiles rollback my-handwriting@v1
hwsol profiles export my-handwriting@v2 --output out/profile-v2.zip
hwsol profiles import --source /path/to/profile-v2.zip
```

Runtime exports contain the version's declared assets and integrity metadata;
they are not a backup of all raw training history. Keep a separate private data
backup. Export does not grant permission to redistribute scans, fonts or weights.
Unspecified local-only sample licenses must not be assumed public.

For a safe engineering demonstration with synthetic, nonpersonal trajectories,
run `python scripts/phase1_demo.py --output out/new-demo`. Calibration format is
illustrated by [the synthetic example](../examples/realism/calibration.json).
Do not claim inferred contours contain observed pen timing or pressure. The
system's structural math layout remains separate from personalized ink.
