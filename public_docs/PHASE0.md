# CPU and compatibility backend setup

`hwsol doctor` checks the ordinary CPU installation, its Handnote dependency,
and the bundled Profile assets. Start with `uv sync --locked`, then
`uv run hwsol doctor` when working from the source or Skill package. The default
Handnote workflow needs no Torch, CUDA, personal data, or training checkpoint.

The older `markdown-to-handwrite` compatibility backend is optional and uses a
separate environment because its historical dependency requirements differ from
the ordinary runtime. It is not installed or validated by the default wheel
smoke test. Do not add those dependencies to the main CPU environment merely to
make the compatibility probe available. `requirements-compatibility.lock`
records that historical environment; its availability is not a promise of
support for every Python version or operating system.

For normal use, follow the [CPU Job guide](../public_docs/GUIDE.md) and
[rendering options](../public_docs/RENDERING.md). This public compatibility note
replaces a private development report only inside prepared distribution files.
