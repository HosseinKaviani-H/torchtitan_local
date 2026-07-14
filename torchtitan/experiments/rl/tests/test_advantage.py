# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the group-relative `AdvantageEstimator`.

Pure Python, no GPU. Covers Dr.GRPO (mean baseline, the default) vs standard
GRPO (std-normalized), zero-variance groups, single-rollout groups, and that the
returned advantages stay in group order.
"""

from __future__ import annotations

import math
import statistics

import pytest

from torchtitan.experiments.rl.rollout import Rollout, RolloutGroup, RolloutStatus
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator


def _group(rewards: list[float]) -> RolloutGroup:
    rollouts = [
        Rollout(
            group_id=0,
            rollout_id=i,
            status=RolloutStatus.COMPLETED,
            turns=[],
            reward=reward,
        )
        for i, reward in enumerate(rewards)
    ]
    return RolloutGroup(group_id=0, rollouts=rollouts)


def test_drgrpo_mean_baseline_is_default():
    """Default (should_std_normalize=False): A_i = r_i - mean(r), denom = 1."""
    estimator = AdvantageEstimator.Config().build()
    assert estimator.should_std_normalize is False
    advantages = estimator(_group([0.0, 1.0, 2.0]))
    assert advantages == pytest.approx([-1.0, 0.0, 1.0])


def test_standard_grpo_divides_by_group_std():
    """should_std_normalize=True: divide the centered reward by pstdev(r) + eps."""
    estimator = AdvantageEstimator.Config(should_std_normalize=True).build()
    rewards = [0.0, 1.0, 2.0]
    denom = statistics.pstdev(rewards) + 1e-6
    expected = [(r - 1.0) / denom for r in rewards]
    assert estimator(_group(rewards)) == pytest.approx(expected)


def test_zero_variance_group_yields_all_zero_advantages():
    """A constant-reward group has no learning signal -> all-zero advantages,
    under both estimators (the numerator is 0 everywhere)."""
    drgrpo = AdvantageEstimator.Config().build()
    std_grpo = AdvantageEstimator.Config(should_std_normalize=True).build()
    assert drgrpo(_group([2.0, 2.0, 2.0])) == pytest.approx([0.0, 0.0, 0.0])
    assert std_grpo(_group([2.0, 2.0, 2.0])) == pytest.approx([0.0, 0.0, 0.0])


def test_single_rollout_group_has_zero_advantage():
    """One rollout: reward equals the group mean -> advantage 0."""
    estimator = AdvantageEstimator.Config().build()
    assert estimator(_group([5.0])) == pytest.approx([0.0])


def test_advantages_preserve_group_order():
    """The returned list is aligned with group.rollouts order."""
    estimator = AdvantageEstimator.Config().build()
    advantages = estimator(_group([3.0, 0.0, 3.0, 0.0]))
    # mean = 1.5 -> [1.5, -1.5, 1.5, -1.5], in the same order as the rewards.
    assert advantages == pytest.approx([1.5, -1.5, 1.5, -1.5])
    assert all(math.isfinite(a) for a in advantages)
