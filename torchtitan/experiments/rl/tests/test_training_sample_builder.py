# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `TrainingSampleBuilder.build_from_group` group-level filters.

The per-rollout packing (`rollout_to_training_samples`) is covered in
`test_rollout_utils.py`; this file covers the group filters that run before it:
failed (empty) groups, untrainable groups (a sibling with no completion), and
zero-reward-variance groups, plus the metrics each path emits.
"""

from __future__ import annotations

from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.types import RolloutTurnID

_GROUP_ID = 0


def _turn(
    *, prompt_token_ids: list[int], completion_token_ids: list[int], version: int
) -> RolloutTurn:
    return RolloutTurn(
        rollout_id=RolloutTurnID(group_id=_GROUP_ID, rollout_id=0, turn_id=0),
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        completion_logprobs=[-0.1] * len(completion_token_ids),
        min_policy_version=version,
        max_policy_version=version,
        completion_message={"role": "assistant", "content": "x"},
    )


def _rollout(
    *, rollout_id: int, reward: float, advantage: float, turns: list[RolloutTurn]
) -> Rollout:
    return Rollout(
        group_id=_GROUP_ID,
        rollout_id=rollout_id,
        status=RolloutStatus.COMPLETED,
        turns=turns,
        reward=reward,
        advantage=advantage,
    )


def _agg(group_output) -> dict[str, float]:
    return m.MetricsProcessor._aggregate_metrics(group_output.metrics)


def test_failed_group_yields_no_samples_and_passes_metrics_through():
    """An empty rollout group (failed generation) produces no training samples;
    its rollout-origin metrics (e.g. the failure metric) still ride through."""
    builder = TrainingSampleBuilder.Config().build()
    group = RolloutGroup(
        group_id=_GROUP_ID,
        rollouts=[],
        metrics=[m.Metric("rollout/group_failures", m.Sum(1.0))],
    )
    out = builder.build_from_group(rollout_group=group)
    assert out.training_samples == []
    assert _agg(out)["rollout/group_failures/sum"] == 1.0


def test_group_with_a_no_completion_sibling_is_dropped_untrainable():
    """If any sibling has no completion tokens, the whole group is dropped."""
    builder = TrainingSampleBuilder.Config().build()
    group = RolloutGroup(
        group_id=_GROUP_ID,
        rollouts=[
            _rollout(
                rollout_id=0,
                reward=1.0,
                advantage=0.5,
                turns=[_turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1)],
            ),
            _rollout(
                rollout_id=1,
                reward=0.0,
                advantage=-0.5,
                turns=[_turn(prompt_token_ids=[1, 2], completion_token_ids=[], version=1)],
            ),
        ],
    )
    out = builder.build_from_group(rollout_group=group)
    assert out.training_samples == []
    assert _agg(out)["training_sample_builder/num_groups_dropped_untrainable/sum"] == 1.0


def test_zero_std_reward_group_is_dropped_by_default():
    """Equal rewards across siblings = no learning signal -> dropped, and the
    zero-std fraction metric records 1.0."""
    builder = TrainingSampleBuilder.Config().build()
    group = RolloutGroup(
        group_id=_GROUP_ID,
        rollouts=[
            _rollout(
                rollout_id=i,
                reward=1.0,  # identical rewards -> zero variance
                advantage=0.0,
                turns=[_turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1)],
            )
            for i in range(2)
        ],
    )
    out = builder.build_from_group(rollout_group=group)
    assert out.training_samples == []
    agg = _agg(out)
    assert agg["training_sample_builder/num_groups_dropped_zero_std/sum"] == 1.0
    assert agg["rollout_reward/group_zero_std_frac/mean"] == 1.0


def test_zero_std_group_kept_when_drop_flag_disabled():
    """With drop_zero_std_reward_groups=False, a zero-std group still trains."""
    builder = TrainingSampleBuilder.Config(drop_zero_std_reward_groups=False).build()
    group = RolloutGroup(
        group_id=_GROUP_ID,
        rollouts=[
            _rollout(
                rollout_id=i,
                reward=1.0,
                advantage=0.0,
                turns=[_turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1)],
            )
            for i in range(2)
        ],
    )
    out = builder.build_from_group(rollout_group=group)
    assert len(out.training_samples) == 2
    assert _agg(out)["rollout_reward/group_zero_std_frac/mean"] == 1.0


def test_trainable_group_produces_samples_and_metrics():
    """A group with reward variance and completions on every sibling produces one
    sample per rollout and the expected group metrics."""
    builder = TrainingSampleBuilder.Config().build()
    group = RolloutGroup(
        group_id=_GROUP_ID,
        rollouts=[
            _rollout(
                rollout_id=0,
                reward=0.0,
                advantage=-0.5,
                turns=[
                    _turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=2)
                ],
            ),
            _rollout(
                rollout_id=1,
                reward=1.0,
                advantage=0.5,
                turns=[_turn(prompt_token_ids=[1, 2], completion_token_ids=[6], version=3)],
            ),
        ],
    )
    out = builder.build_from_group(rollout_group=group)
    assert len(out.training_samples) == 2
    agg = _agg(out)
    # A trainable group is not dropped, so no drop metric is emitted for it.
    assert "training_sample_builder/num_groups_dropped_zero_std/sum" not in agg
    assert "training_sample_builder/num_groups_dropped_untrainable/sum" not in agg
    assert agg["rollout_reward/group_zero_std_frac/mean"] == 0.0
    assert agg["training_sample_builder/num_training_samples/sum"] == 2.0
    # One branch (training sample) per rollout.
    assert agg["rollout/branches_per_rollout/mean"] == 1.0
    assert agg["rollout/branches_per_rollout/max"] == 1.0
    # Version span across the two samples' turns.
    assert agg["rollout/min_policy_version/min"] == 2.0
    assert agg["rollout/max_policy_version/max"] == 3.0
