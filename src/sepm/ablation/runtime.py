"""Map paper component conditions to concrete ``MemoryConfig`` switches.

This module is the sole component-condition entry point between external
benchmark adapters and the core service.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from ..config import MemoryConfig


# Names exactly match evaluation_matrix.json. Values may contain only MemoryConfig
# fields, so misspelled settings fail in dataclasses.replace instead of silently
# running the full method.
ABLATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "no-extra-components": {
        "enable_procedural_memory": False,
        "enable_divergence_detection": False,
        "share_blackboard_across_agents": False,
    },
    "blackboard-only": {
        "enable_procedural_memory": False,
        "enable_divergence_detection": False,
        "share_blackboard_across_agents": True,
    },
    "sop-only": {"enable_divergence_detection": False},
    "divergence-only": {"enable_procedural_memory": False},
    "full": {},
}


def apply_ablation(base: MemoryConfig, variant: str) -> MemoryConfig:
    """Return a new configuration with the component condition applied."""
    if variant not in ABLATION_OVERRIDES:
        choices = ", ".join(sorted(ABLATION_OVERRIDES))
        raise ValueError(f"unknown ablation {variant!r}; choose one of: {choices}")
    return replace(base, **ABLATION_OVERRIDES[variant])


def config_from_environment(base: MemoryConfig | None = None) -> MemoryConfig:
    """Read the runner-provided variant for third-party benchmark adapters."""
    return apply_ablation(base or MemoryConfig(), os.getenv("SEPM_EVAL_ABLATION", "full"))
