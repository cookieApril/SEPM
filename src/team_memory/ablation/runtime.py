"""把论文中的最小组件条件映射为真实 ``MemoryConfig`` 开关。

该模块是外部 benchmark 适配器与核心服务之间的唯一组件条件入口。
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from ..config import MemoryConfig


# 名称与 evaluation_matrix.json 完全一致。值只能包含 MemoryConfig 的字段，因此拼错配置会
# 在 dataclasses.replace 时立即失败，而不会静默运行成 Full 方法。
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
    """返回应用组件条件后的新配置，原配置保持不可变。"""
    if variant not in ABLATION_OVERRIDES:
        choices = ", ".join(sorted(ABLATION_OVERRIDES))
        raise ValueError(f"unknown ablation {variant!r}; choose one of: {choices}")
    return replace(base, **ABLATION_OVERRIDES[variant])


def config_from_environment(base: MemoryConfig | None = None) -> MemoryConfig:
    """读取统一 runner 注入的 variant，供第三方 benchmark adapter 构造服务。"""
    return apply_ablation(base or MemoryConfig(), os.getenv("TEAM_MEMORY_EVAL_ABLATION", "full"))
