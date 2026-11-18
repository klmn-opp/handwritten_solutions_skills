import os
import tempfile

# Avoid writing a font cache into the user's home during tests.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="hwsol-test-mpl-"))
