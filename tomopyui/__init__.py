try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Samuel Scott Welborn"
__email__ = "swelborn@seas.upenn.edu"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]
