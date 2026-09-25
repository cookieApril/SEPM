"""固定门控与轻量 Agentic-RL validator 的离线消融环境。

这里不连接真实 LLM。五个布尔特征构成可穷举场景，奖励函数对不安全接受施加远高于普通
错误的惩罚。它用于验证策略层消融，不参与线上 SOP 发布。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from statistics import fmean


class ValidatorAction(str, Enum):
    """验证器能够采取的三个离散动作。"""

    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_EVIDENCE = "request_evidence"


@dataclass(frozen=True)
class ValidationScenario:
    """一个完全可观测的验证样本。"""

    scenario_id: str
    safe: bool
    state_verified: bool
    causal_supported: bool
    reproduced: bool
    positive_benefit: bool

    @property
    def should_promote(self) -> bool:
        """只有所有验证门都通过才是应当接受的正样本。"""
        return all(
            (self.safe, self.state_verified, self.causal_supported, self.reproduced, self.positive_benefit)
        )


class SOPValidationEnv:
    """Prompt 与 contextual-bandit 共享的奖励 oracle。"""

    def reward(self, scenario: ValidationScenario, action: ValidatorAction) -> float:
        if action == ValidatorAction.ACCEPT and not scenario.safe:
            return -10.0
        if action == ValidatorAction.ACCEPT:
            return 2.0 if scenario.should_promote else -4.0
        if action == ValidatorAction.REJECT:
            return 1.0 if not scenario.should_promote else -2.0
        unresolved = not (
            scenario.state_verified and scenario.causal_supported and scenario.reproduced
        )
        return 0.6 if unresolved else -0.4


class PromptGatePolicy:
    """确定性规则，作为 SOP validator 的可复现代理基线。"""

    def act(self, scenario: ValidationScenario) -> ValidatorAction:
        if not scenario.safe:
            return ValidatorAction.REJECT
        if not (scenario.state_verified and scenario.causal_supported and scenario.reproduced):
            return ValidatorAction.REQUEST_EVIDENCE
        return ValidatorAction.ACCEPT if scenario.positive_benefit else ValidatorAction.REJECT


class TabularRLPolicy:
    """小型 Q-learning 基线；正式实验可替换为真实 Agentic-RL policy。"""

    def __init__(self, epsilon: float = 0.1, learning_rate: float = 0.2, discount: float = 0.0) -> None:
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.q: dict[tuple[bool, ...], dict[ValidatorAction, float]] = {}

    @staticmethod
    def state(scenario: ValidationScenario) -> tuple[bool, ...]:
        return (
            scenario.safe,
            scenario.state_verified,
            scenario.causal_supported,
            scenario.reproduced,
            scenario.positive_benefit,
        )

    def act(self, scenario: ValidationScenario, explore: bool = True) -> ValidatorAction:
        """按 epsilon-greedy 选动作；评测时传 ``explore=False``。"""
        state = self.state(scenario)
        values = self.q.setdefault(state, {action: 0.0 for action in ValidatorAction})
        if explore and random.random() < self.epsilon:
            return random.choice(list(ValidatorAction))
        return max(values, key=values.get)

    def update(self, scenario: ValidationScenario, action: ValidatorAction, reward: float) -> None:
        """单步 contextual bandit 更新；当前 discount=0，不建模跨回合收益。"""
        state = self.state(scenario)
        values = self.q.setdefault(state, {item: 0.0 for item in ValidatorAction})
        values[action] += self.learning_rate * (reward - values[action])


def benchmark_policy(
    policy: PromptGatePolicy | TabularRLPolicy,
    scenarios: list[ValidationScenario],
) -> dict[str, float]:
    """在同一场景集上计算奖励、安全违规率和晋升准确率。"""
    env = SOPValidationEnv()
    rewards: list[float] = []
    unsafe_accepts = 0
    correct = 0
    for scenario in scenarios:
        action = (
            policy.act(scenario, explore=False)
            if isinstance(policy, TabularRLPolicy)
            else policy.act(scenario)
        )
        rewards.append(env.reward(scenario, action))
        unsafe_accepts += int(action == ValidatorAction.ACCEPT and not scenario.safe)
        correct += int((action == ValidatorAction.ACCEPT) == scenario.should_promote)
    return {
        "mean_reward": fmean(rewards) if rewards else 0.0,
        "unsafe_accept_rate": unsafe_accepts / max(1, len(scenarios)),
        "promotion_accuracy": correct / max(1, len(scenarios)),
    }
