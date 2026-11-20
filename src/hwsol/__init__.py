"""Handwritten Solutions: local Profiles and CPU Studio; solving is Agent-owned."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("handwritten-solutions")
except PackageNotFoundError:
    __version__ = "0+uninstalled"
