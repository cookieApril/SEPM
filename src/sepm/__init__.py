"""Stable public entry point for SEPM.

External callers generally need only :class:`MemoryConfig` and
:class:`SEPMService`. Other modules are implementation details but remain
available to experiment code. ``__all__`` limits wildcard imports, and
``__version__`` matches the package version.
"""

from .config import MemoryConfig
from .service import SEPMService

__all__ = ["MemoryConfig", "SEPMService"]
__version__ = "0.1.0"
