"""穷举五门状态，对比固定 Prompt 规则与表格 Q-learning 基线。

固定随机种子保证结果可重复。此脚本只运行内存实验，不读取或写入 Team Memory 数据库。
"""

from __future__ import annotations

import itertools
import random

from team_memory.ablation.validator_policy import (
    PromptGatePolicy,
    SOPValidationEnv,
    TabularRLPolicy,
    ValidationScenario,
    benchmark_policy,
)


def scenarios() -> list[ValidationScenario]:
    """生成 2^5 个布尔门控组合，避免评测集遗漏边界状态。"""
    return [
        ValidationScenario(
            scenario_id=f"scenario-{index}",
            safe=values[0],
            state_verified=values[1],
            causal_supported=values[2],
            reproduced=values[3],
            positive_benefit=values[4],
        )
        for index, values in enumerate(itertools.product((False, True), repeat=5))
    ]


def main() -> None:
    """训练 500 个随机回合，再在完整场景集上无探索评测两个策略。"""
    random.seed(7)
    dataset = scenarios()
    env = SOPValidationEnv()
    rl = TabularRLPolicy(epsilon=0.25)
    # 训练抽样，最终 benchmark_policy 仍在完整 32 个场景上评估。
    for _ in range(500):
        scenario = random.choice(dataset)
        action = rl.act(scenario)
        rl.update(scenario, action, env.reward(scenario, action))
    print("prompt:", benchmark_policy(PromptGatePolicy(), dataset))
    print("rl:", benchmark_policy(rl, dataset))


if __name__ == "__main__":
    main()
