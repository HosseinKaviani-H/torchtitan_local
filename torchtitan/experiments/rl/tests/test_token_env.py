# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the `TokenEnv` wrapper contract.

`TokenEnv` is the shared driver for BOTH single-turn and multi-turn rollouts: it
converts messages <-> tokens via a renderer and enforces every terminal status
(prompt too long, length/abort finish reasons, parse errors, step timeouts, max
turns, env done). No GPU, no real renderer, no real MessageEnv: both are faked
(the renderer is duck-typed, matching the methods `TokenEnv` calls).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from torchtitan.experiments.rl.environment import (
    MessageEnvInitOutput,
    MessageEnvStepOutput,
    TokenEnv,
)
from torchtitan.experiments.rl.rollout.types import RolloutStatus
from torchtitan.experiments.rl.types import Completion


class _FakeRenderer:
    """Duck-typed stand-in for renderers.Renderer (only the methods TokenEnv calls)."""

    def __init__(
        self,
        *,
        render_tokens=(1, 2, 3),
        bridge_tokens=None,
        parse_raises=False,
        parsed_content="answer",
        tool_calls=None,
    ) -> None:
        self._render_tokens = list(render_tokens)
        self._bridge_tokens = bridge_tokens  # None -> bridge returns None
        self._parse_raises = parse_raises
        self._parsed_content = parsed_content
        self._tool_calls = tool_calls
        self.render_calls = 0
        self.bridge_calls = 0

    def render_ids(self, *, messages, tools, add_generation_prompt):
        self.render_calls += 1
        return list(self._render_tokens)

    def parse_response(self, *, token_ids):
        if self._parse_raises:
            raise ValueError("unparseable response")
        return SimpleNamespace(
            content=self._parsed_content,
            reasoning_content=None,
            tool_calls=self._tool_calls,
        )

    def bridge_to_next_turn(
        self, *, previous_prompt_ids, previous_completion_ids, new_messages, tools
    ):
        self.bridge_calls += 1
        if self._bridge_tokens is None:
            return None
        return SimpleNamespace(token_ids=list(self._bridge_tokens))


class _FakeMessageEnv:
    """Duck-typed MessageEnv: fixed init output and one canned step output."""

    def __init__(
        self,
        *,
        init_messages=None,
        tools=None,
        step_output=None,
        step_sleep_s=0.0,
    ) -> None:
        self._init_messages = init_messages or [{"role": "user", "content": "hi"}]
        self._tools = tools or []
        self._step_output = step_output or MessageEnvStepOutput(done=True)
        self._step_sleep_s = step_sleep_s
        self.step_calls = 0

    async def init(self) -> MessageEnvInitOutput:
        return MessageEnvInitOutput(
            init_prompt_messages=self._init_messages, tools=self._tools
        )

    async def step(self, completion_message) -> MessageEnvStepOutput:
        self.step_calls += 1
        if self._step_sleep_s:
            await asyncio.sleep(self._step_sleep_s)
        return self._step_output

    async def close(self) -> None:
        pass


def _completion(*, token_ids=(4, 5), finish_reason="stop") -> Completion:
    token_ids = list(token_ids)
    return Completion(
        min_policy_version=1,
        max_policy_version=1,
        request_id="r",
        token_ids=token_ids,
        token_logprobs=[-0.1] * len(token_ids),
        finish_reason=finish_reason,
    )


def _env(*, renderer=None, message_env=None, **config_kwargs) -> TokenEnv:
    return TokenEnv.Config(**config_kwargs).build(
        message_env=message_env or _FakeMessageEnv(),
        renderer=renderer or _FakeRenderer(),
    )


# --- init ---


def test_init_renders_prompt_and_is_ongoing():
    async def main():
        renderer = _FakeRenderer(render_tokens=[1, 2, 3])
        env = _env(renderer=renderer)
        out = await env.init()
        assert out.status is RolloutStatus.ONGOING
        assert out.next_prompt_token_ids == [1, 2, 3]
        assert out.next_prompt_messages == [{"role": "user", "content": "hi"}]

    asyncio.run(main())


def test_init_prompt_too_long_is_truncated_but_still_carried():
    async def main():
        renderer = _FakeRenderer(render_tokens=[1, 2, 3])
        env = _env(renderer=renderer, max_rollout_tokens=2)  # 3 >= 2
        out = await env.init()
        assert out.status is RolloutStatus.TRUNCATED_PROMPT_TOO_LONG
        # The over-budget prompt is kept so it stays debuggable.
        assert out.next_prompt_token_ids == [1, 2, 3]

    asyncio.run(main())


# --- step: finish_reason terminals (env is not stepped) ---


def test_step_length_finish_truncates_without_stepping_env():
    async def main():
        message_env = _FakeMessageEnv()
        env = _env(message_env=message_env)
        await env.init()
        out = await env.step(_completion(finish_reason="length"))
        assert out.status is RolloutStatus.TRUNCATED_LENGTH
        assert out.next_prompt_token_ids is None
        assert out.completion_message == {"role": "assistant", "content": "answer"}
        assert message_env.step_calls == 0  # env not advanced on a partial response

    asyncio.run(main())


def test_step_abort_finish_is_error_abort():
    async def main():
        message_env = _FakeMessageEnv()
        env = _env(message_env=message_env)
        await env.init()
        out = await env.step(_completion(finish_reason="abort"))
        assert out.status is RolloutStatus.ERROR_ABORT
        assert message_env.step_calls == 0

    asyncio.run(main())


def test_step_parse_failure_is_error_parse():
    async def main():
        renderer = _FakeRenderer(parse_raises=True)
        message_env = _FakeMessageEnv()
        env = _env(renderer=renderer, message_env=message_env)
        await env.init()
        out = await env.step(_completion())
        assert out.status is RolloutStatus.ERROR_PARSE
        assert out.next_prompt_token_ids is None
        assert message_env.step_calls == 0

    asyncio.run(main())


def test_step_timeout_is_error_timeout():
    async def main():
        # env.step sleeps longer than the wrapper's step timeout.
        message_env = _FakeMessageEnv(step_sleep_s=0.5)
        env = _env(message_env=message_env, step_timeout_s=0.01)
        await env.init()
        out = await env.step(_completion())
        assert out.status is RolloutStatus.ERROR_TIMEOUT

    asyncio.run(main())


# --- step: env-driven terminals ---


def test_step_done_completes_and_propagates_rewards():
    async def main():
        message_env = _FakeMessageEnv(
            step_output=MessageEnvStepOutput(
                done=True, env_rewards={"correct": 1.0}
            )
        )
        env = _env(message_env=message_env)
        await env.init()
        out = await env.step(_completion())
        assert out.status is RolloutStatus.COMPLETED
        assert out.env_rewards == {"correct": 1.0}
        assert out.completion_message == {"role": "assistant", "content": "answer"}

    asyncio.run(main())


def test_single_turn_env_completes_on_first_step():
    async def main():
        # An env that is done after one step is the single-turn case.
        env = _env(message_env=_FakeMessageEnv(step_output=MessageEnvStepOutput(done=True)))
        first = await env.init()
        assert first.status is RolloutStatus.ONGOING
        second = await env.step(_completion())
        assert second.status is RolloutStatus.COMPLETED

    asyncio.run(main())


def test_step_over_max_turns_is_truncated_max_turns():
    async def main():
        # Env would continue (not done), but max_num_turns=1 stops after one turn.
        message_env = _FakeMessageEnv(
            step_output=MessageEnvStepOutput(
                env_messages=[{"role": "user", "content": "next"}], done=False
            )
        )
        env = _env(message_env=message_env, max_num_turns=1)
        await env.init()
        out = await env.step(_completion())
        assert out.status is RolloutStatus.TRUNCATED_MAX_TURNS

    asyncio.run(main())


def test_next_prompt_over_budget_is_truncated_prompt_too_long():
    async def main():
        renderer = _FakeRenderer(render_tokens=[1, 2], bridge_tokens=[1, 2, 4, 7, 8])
        message_env = _FakeMessageEnv(
            step_output=MessageEnvStepOutput(
                env_messages=[{"role": "user", "content": "next"}], done=False
            )
        )
        env = _env(renderer=renderer, message_env=message_env, max_rollout_tokens=3)
        first = await env.init()
        assert first.status is RolloutStatus.ONGOING  # 2 < 3
        out = await env.step(_completion())  # bridged next prompt has 5 >= 3 tokens
        assert out.status is RolloutStatus.TRUNCATED_PROMPT_TOO_LONG

    asyncio.run(main())


# --- step: continuing turns (bridge vs full re-render) ---


def test_bridge_tokens_used_directly_for_next_prompt():
    async def main():
        renderer = _FakeRenderer(render_tokens=[1, 2], bridge_tokens=[1, 2, 4, 9])
        message_env = _FakeMessageEnv(
            step_output=MessageEnvStepOutput(
                env_messages=[{"role": "user", "content": "next"}], done=False
            )
        )
        env = _env(renderer=renderer, message_env=message_env)
        await env.init()
        out = await env.step(_completion())
        assert out.status is RolloutStatus.ONGOING
        assert out.next_prompt_token_ids == [1, 2, 4, 9]
        # Bridge succeeded, so no second full render.
        assert renderer.bridge_calls == 1
        assert renderer.render_calls == 1  # only the init render

    asyncio.run(main())


def test_bridge_none_falls_back_to_full_render():
    async def main():
        renderer = _FakeRenderer(render_tokens=[1, 2], bridge_tokens=None)
        message_env = _FakeMessageEnv(
            step_output=MessageEnvStepOutput(
                env_messages=[{"role": "user", "content": "next"}], done=False
            )
        )
        env = _env(renderer=renderer, message_env=message_env)
        await env.init()
        out = await env.step(_completion())
        assert out.status is RolloutStatus.ONGOING
        assert out.next_prompt_token_ids == [1, 2]  # from the fallback render
        assert renderer.bridge_calls == 1
        assert renderer.render_calls == 2  # init render + fallback render

    asyncio.run(main())
