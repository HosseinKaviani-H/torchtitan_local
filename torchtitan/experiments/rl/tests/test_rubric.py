# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the base `Rubric` scoring (weighted reward-fn sum, truncation /
error short-circuit, config validation).

Task-specific reward fns are covered in `test_alphabet_sort.py` /
`test_search_r1.py`; this file covers the generic combining logic. Pure Python,
no GPU.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from torchtitan.experiments.rl.rollout import Rollout, RolloutStatus
from torchtitan.experiments.rl.rubrics import RewardFn, Rubric


class _RewardA(RewardFn):
    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        value: float = 1.0

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._value = config.value

    async def __call__(self, rollout, env_input) -> float:
        return self._value


class _RewardB(RewardFn):
    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        value: float = 0.0

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._value = config.value

    async def __call__(self, rollout, env_input) -> float:
        return self._value


class _Boom(RewardFn):
    """A reward fn that fails if it is ever called (to prove short-circuits)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout, env_input) -> float:
        raise AssertionError("reward fn should not run on a short-circuited rollout")


def _rollout(status: RolloutStatus = RolloutStatus.COMPLETED) -> Rollout:
    return Rollout(group_id=0, rollout_id=0, status=status, turns=[])


def test_weighted_sum_of_reward_fns_is_normalized():
    """reward = sum(weight_i / sum(weights) * value_i). With weights 1 and 3 and
    values 1.0 and 0.0: (1/4)*1 + (3/4)*0 = 0.25."""
    rubric = Rubric.Config(
        reward_fns=[
            _RewardA.Config(weight=1.0, value=1.0),
            _RewardB.Config(weight=3.0, value=0.0),
        ]
    ).build()
    [out] = asyncio.run(rubric.score_group([_rollout()], env_input=None))
    assert out.reward == pytest.approx(0.25)
    assert out.reward_breakdown == {"_RewardA": 1.0, "_RewardB": 0.0}


def test_single_reward_fn_returns_its_value():
    rubric = Rubric.Config(reward_fns=[_RewardA.Config(value=0.7)]).build()
    [out] = asyncio.run(rubric.score_group([_rollout()], env_input=None))
    assert out.reward == pytest.approx(0.7)


def test_truncation_reward_short_circuits_reward_fns():
    rubric = Rubric.Config(
        reward_fns=[_Boom.Config()], truncation_reward=0.0
    ).build()
    [out] = asyncio.run(
        rubric.score_group(
            [_rollout(status=RolloutStatus.TRUNCATED_LENGTH)], env_input=None
        )
    )
    assert out.reward == 0.0
    assert out.reward_breakdown == {"truncated": 0.0}


def test_error_reward_short_circuits_reward_fns():
    rubric = Rubric.Config(reward_fns=[_Boom.Config()], error_reward=-1.0).build()
    [out] = asyncio.run(
        rubric.score_group([_rollout(status=RolloutStatus.ERROR)], env_input=None)
    )
    assert out.reward == -1.0
    assert out.reward_breakdown == {"errored": -1.0}


def test_completed_rollout_runs_reward_fns_even_when_short_circuits_configured():
    # A COMPLETED rollout is neither truncated nor errored, so the reward fns run.
    rubric = Rubric.Config(
        reward_fns=[_RewardA.Config(value=1.0)],
        truncation_reward=0.0,
        error_reward=-1.0,
    ).build()
    [out] = asyncio.run(rubric.score_group([_rollout()], env_input=None))
    assert out.reward == pytest.approx(1.0)


def test_score_group_returns_one_output_per_rollout_in_order():
    rubric = Rubric.Config(reward_fns=[_RewardA.Config(value=0.5)]).build()
    rollouts = [_rollout() for _ in range(3)]
    outs = asyncio.run(rubric.score_group(rollouts, env_input=None))
    assert len(outs) == 3
    assert all(o.reward == pytest.approx(0.5) for o in outs)


def test_empty_reward_fns_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        Rubric.Config(reward_fns=[]).build()


def test_duplicate_reward_fn_names_raises():
    with pytest.raises(ValueError, match="unique"):
        Rubric.Config(
            reward_fns=[_RewardA.Config(value=1.0), _RewardA.Config(value=0.0)]
        ).build()


def test_nonpositive_weight_sum_raises():
    with pytest.raises(ValueError, match="positive"):
        Rubric.Config(reward_fns=[_RewardA.Config(weight=0.0, value=1.0)]).build()
