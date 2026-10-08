"""Enumerate five-gate states and compare prompt rules with tabular Q-learning.

A fixed random seed makes results reproducible. This script runs only an
in-memory experiment and does not read or write an SEPM database.
"""

from __future__ import annotations

import itertools
import random

from sepm.ablation.validator_policy import (
    PromptGatePolicy,
    SOPValidationEnv,
    TabularRLPolicy,
    ValidationScenario,
    benchmark_policy,
)


def scenarios() -> list[ValidationScenario]:
    """Generate all 2^5 Boolean gate combinations, including boundary states."""
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
    """Train for 500 random episodes, then evaluate both policies without exploration."""
    random.seed(7)
    dataset = scenarios()
    env = SOPValidationEnv()
    rl = TabularRLPolicy(epsilon=0.25)
    # Training samples scenarios; benchmark_policy still evaluates all 32 scenarios.
    for _ in range(500):
        scenario = random.choice(dataset)
        action = rl.act(scenario)
        rl.update(scenario, action, env.reward(scenario, action))
    print("prompt:", benchmark_policy(PromptGatePolicy(), dataset))
    print("rl:", benchmark_policy(rl, dataset))


if __name__ == "__main__":
    main()
