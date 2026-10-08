"""Offline ablation environment for fixed gates and a lightweight RL validator.

No real LLM is used. Five Boolean features form an exhaustive scenario space,
and the reward penalizes unsafe acceptance much more heavily than ordinary
errors. This module evaluates policy-level ablations and never publishes SOPs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from statistics import fmean


class ValidatorAction(str, Enum):
    """The three discrete actions available to a validator."""

    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_EVIDENCE = "request_evidence"


@dataclass(frozen=True)
class ValidationScenario:
    """A fully observable validation sample."""

    scenario_id: str
    safe: bool
    state_verified: bool
    causal_supported: bool
    reproduced: bool
    positive_benefit: bool

    @property
    def should_promote(self) -> bool:
        """Return true only when every validation gate passes."""
        return all(
            (self.safe, self.state_verified, self.causal_supported, self.reproduced, self.positive_benefit)
        )


class SOPValidationEnv:
    """Reward oracle shared by prompt and contextual-bandit policies."""

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
    """Deterministic rules used as a reproducible SOP-validator proxy baseline."""

    def act(self, scenario: ValidationScenario) -> ValidatorAction:
        if not scenario.safe:
            return ValidatorAction.REJECT
        if not (scenario.state_verified and scenario.causal_supported and scenario.reproduced):
            return ValidatorAction.REQUEST_EVIDENCE
        return ValidatorAction.ACCEPT if scenario.positive_benefit else ValidatorAction.REJECT


class TabularRLPolicy:
    """Small Q-learning baseline that formal experiments may replace with an RL policy."""

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
        """Select an epsilon-greedy action; evaluation passes ``explore=False``."""
        state = self.state(scenario)
        values = self.q.setdefault(state, {action: 0.0 for action in ValidatorAction})
        if explore and random.random() < self.epsilon:
            return random.choice(list(ValidatorAction))
        return max(values, key=values.get)

    def update(self, scenario: ValidationScenario, action: ValidatorAction, reward: float) -> None:
        """Apply one contextual-bandit update; discount=0 ignores cross-episode returns."""
        state = self.state(scenario)
        values = self.q.setdefault(state, {item: 0.0 for item in ValidatorAction})
        values[action] += self.learning_rate * (reward - values[action])


def benchmark_policy(
    policy: PromptGatePolicy | TabularRLPolicy,
    scenarios: list[ValidationScenario],
) -> dict[str, float]:
    """Compute reward, unsafe-acceptance rate, and promotion accuracy."""
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
