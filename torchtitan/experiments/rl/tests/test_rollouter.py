# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the `Rollouter` rollout driver.

Covers the core turn loop (`_run_single_rollout`) for single-turn, multi-turn,
truncated, and error trajectories, and group orchestration
(`run_group_rollouts`): sibling fan-out, per-sample seed offset, scoring, and
advantage assignment. No GPU / generator: `generate_fn` and the envs are faked.
"""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.environment import TokenEnvOutput
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import RolloutStatus
from torchtitan.experiments.rl.rubrics import RubricOutput
from torchtitan.experiments.rl.types import Completion


class _ScriptedEnv:
    """Returns a fixed sequence of TokenEnvOutputs: [0] from init(), the rest
    from successive step() calls. A step whose scripted output is the string
    "raise" makes step() raise, to exercise the error path."""

    def __init__(self, outputs: list) -> None:
        self._outputs = outputs
        self._idx = 0
        self.closed = False

    async def init(self) -> TokenEnvOutput:
        out = self._outputs[0]
        self._idx = 1
        return out

    async def step(self, completion) -> TokenEnvOutput:
        out = self._outputs[self._idx]
        self._idx += 1
        if out == "raise":
            raise RuntimeError("env step blew up")
        return out

    async def close(self) -> None:
        self.closed = True


def _ongoing(prompt_token_ids: list[int]) -> TokenEnvOutput:
    return TokenEnvOutput(
        next_prompt_token_ids=prompt_token_ids,
        next_prompt_messages=[{"role": "user", "content": "p"}],
        status=RolloutStatus.ONGOING,
    )


def _terminal(status: RolloutStatus) -> TokenEnvOutput:
    return TokenEnvOutput(
        next_prompt_token_ids=None,
        next_prompt_messages=None,
        status=status,
        completion_message={"role": "assistant", "content": "a"},
    )


def _make_generate_fn(*, token_ids=(9,)):
    """A fake GenerateFn recording every call; returns a fixed Completion."""
    calls = []

    async def generate_fn(
        prompt_token_ids, *, request_id, routing_session_id=None, sampling_config=None
    ):
        calls.append(
            {
                "prompt_token_ids": prompt_token_ids,
                "request_id": request_id,
                "routing_session_id": routing_session_id,
                "seed": None if sampling_config is None else sampling_config.seed,
            }
        )
        return Completion(
            min_policy_version=1,
            max_policy_version=1,
            request_id=request_id,
            token_ids=list(token_ids),
            token_logprobs=[-0.1] * len(token_ids),
            finish_reason="stop",
        )

    return generate_fn, calls


def _bare_rollouter() -> Rollouter:
    # _run_single_rollout uses no instance state, so a bare instance is enough.
    return Rollouter.__new__(Rollouter)


# --- _run_single_rollout: turn loop ---


def test_single_turn_rollout_produces_one_turn():
    async def main():
        env = _ScriptedEnv([_ongoing([1, 2]), _terminal(RolloutStatus.COMPLETED)])
        generate_fn, calls = _make_generate_fn(token_ids=(9,))
        rollout = await _bare_rollouter()._run_single_rollout(
            generate_fn=generate_fn,
            env=env,
            sampling=SamplingConfig(),
            group_id=3,
            rollout_id=0,
        )
        assert rollout.status is RolloutStatus.COMPLETED
        assert len(rollout.turns) == 1
        turn = rollout.turns[0]
        assert turn.prompt_token_ids == [1, 2]
        assert turn.completion_token_ids == [9]
        assert len(calls) == 1
        assert calls[0]["request_id"] == "group=3/rollout=0/turn=0"
        # Sticky routing key drops the turn so a sample's turns reuse one generator.
        assert calls[0]["routing_session_id"] == "group=3/rollout=0"

    asyncio.run(main())


def test_multi_turn_rollout_produces_one_turn_per_step():
    async def main():
        env = _ScriptedEnv(
            [
                _ongoing([1, 2]),
                _ongoing([1, 2, 9, 3]),
                _terminal(RolloutStatus.COMPLETED),
            ]
        )
        generate_fn, calls = _make_generate_fn(token_ids=(9,))
        rollout = await _bare_rollouter()._run_single_rollout(
            generate_fn=generate_fn,
            env=env,
            sampling=SamplingConfig(),
            group_id=0,
            rollout_id=1,
        )
        assert rollout.status is RolloutStatus.COMPLETED
        assert len(rollout.turns) == 2
        assert [t.prompt_token_ids for t in rollout.turns] == [[1, 2], [1, 2, 9, 3]]
        # Turn ids increment; routing session stays constant across turns.
        assert [c["request_id"] for c in calls] == [
            "group=0/rollout=1/turn=0",
            "group=0/rollout=1/turn=1",
        ]
        assert {c["routing_session_id"] for c in calls} == {"group=0/rollout=1"}

    asyncio.run(main())


def test_error_mid_rollout_marks_error_and_keeps_prior_turns():
    async def main():
        # init ok, first step ok (turn 0 recorded), second step raises.
        env = _ScriptedEnv([_ongoing([1, 2]), _ongoing([1, 2, 9]), "raise"])
        generate_fn, _ = _make_generate_fn()
        rollout = await _bare_rollouter()._run_single_rollout(
            generate_fn=generate_fn,
            env=env,
            sampling=SamplingConfig(),
            group_id=0,
            rollout_id=0,
        )
        assert rollout.status is RolloutStatus.ERROR
        assert len(rollout.turns) == 1  # only the turn that completed before the failure

    asyncio.run(main())


def test_truncated_status_propagates():
    async def main():
        env = _ScriptedEnv([_ongoing([1, 2]), _terminal(RolloutStatus.TRUNCATED_LENGTH)])
        generate_fn, _ = _make_generate_fn()
        rollout = await _bare_rollouter()._run_single_rollout(
            generate_fn=generate_fn,
            env=env,
            sampling=SamplingConfig(),
            group_id=0,
            rollout_id=0,
        )
        assert rollout.status is RolloutStatus.TRUNCATED_LENGTH
        assert len(rollout.turns) == 1

    asyncio.run(main())


# --- run_group_rollouts: orchestration ---


class _GroupRollouter(Rollouter):
    """Bare Rollouter with make_env_group / score_group overridden to fakes."""

    def __init__(self, *, envs, rewards) -> None:
        self._envs = envs
        self._rewards = rewards
        self.advantage_estimator = AdvantageEstimator.Config().build()

    def make_env_group(self, *, sample, group_size, renderer):
        return self._envs[:group_size]

    async def score_group(self, rollouts, env_input):
        return [RubricOutput(reward=r) for r in self._rewards]


def test_run_group_rollouts_scores_and_assigns_advantages():
    async def main():
        envs = [
            _ScriptedEnv([_ongoing([1, 2]), _terminal(RolloutStatus.COMPLETED)])
            for _ in range(2)
        ]
        rollouter = _GroupRollouter(envs=envs, rewards=[0.0, 1.0])
        generate_fn, _ = _make_generate_fn()
        group = await rollouter.run_group_rollouts(
            generate_fn=generate_fn,
            sample=object(),
            group_id=5,
            group_size=2,
            sampling=SamplingConfig(seed=None),
            renderer=None,
        )
        assert group.group_id == 5
        assert [r.reward for r in group.rollouts] == [0.0, 1.0]
        # Dr.GRPO advantage = reward - mean; mean = 0.5 -> [-0.5, 0.5].
        assert [r.advantage for r in group.rollouts] == pytest.approx([-0.5, 0.5])
        # Envs are closed after the group runs.
        assert all(env.closed for env in envs)

    asyncio.run(main())


def test_run_group_rollouts_offsets_seed_per_sample():
    async def main():
        envs = [
            _ScriptedEnv([_ongoing([1, 2]), _terminal(RolloutStatus.COMPLETED)])
            for _ in range(3)
        ]
        rollouter = _GroupRollouter(envs=envs, rewards=[0.0, 0.0, 0.0])
        generate_fn, calls = _make_generate_fn()
        await rollouter.run_group_rollouts(
            generate_fn=generate_fn,
            sample=object(),
            group_id=0,
            group_size=3,
            sampling=SamplingConfig(seed=100),
            renderer=None,
        )
        # Base seed 100 offset by sample index -> {100, 101, 102}, one per sibling.
        assert {c["seed"] for c in calls} == {100, 101, 102}

    asyncio.run(main())
