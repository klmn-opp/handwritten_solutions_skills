import os
import tempfile

import pytest

# Avoid writing a font cache into the user's home during tests.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="hwsol-test-mpl-"))


@pytest.fixture(autouse=True)
def isolated_run_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HWSOL_WORKSPACE", str(tmp_path / "workspace"))
