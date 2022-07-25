"""imitation: implementations of imitation and reward learning algorithms."""

from importlib import metadata

try:
    __version__ = metadata.version("imitation")
except metadata.PackageNotFoundError:
    # package is not installed
    pass
