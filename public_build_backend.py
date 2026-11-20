"""PEP 517 boundary: public wheels/sdists require an audited staged source.

Editable installs are local development artifacts and remain available in the
private working tree. No hook changes project version or dependency requirements.
"""

from pathlib import Path


def _backend():
    from setuptools import build_meta

    return build_meta


def _public():
    from scripts.package_release import verify_public_source

    try:
        verify_public_source(Path(__file__).resolve().parent)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise RuntimeError(
            "Public wheel/sdist build refused: use `python scripts/package_release.py "
            "--stage <new-public-source>` and build that directory. The private checkout "
            "is still available for `uv sync --locked` editable development. "
            f"Reason: {exc}"
        ) from exc


def get_requires_for_build_wheel(config_settings=None):
    _public()
    return _backend().get_requires_for_build_wheel(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _public()
    return _backend().prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _public()
    return _backend().build_wheel(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_sdist(config_settings=None):
    _public()
    return _backend().get_requires_for_build_sdist(config_settings)


def build_sdist(sdist_directory, config_settings=None):
    _public()
    return _backend().build_sdist(sdist_directory, config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _backend().get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _backend().prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _backend().build_editable(wheel_directory, config_settings, metadata_directory)
