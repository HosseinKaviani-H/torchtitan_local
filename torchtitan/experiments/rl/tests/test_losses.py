# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the GRPO / DAPO per-token clipped surrogate loss.

Pure-tensor, tiny shapes, no GPU. The trainer logprobs are made analytic by
feeding all-zero logits: softmax is uniform over a size-V vocab, so every
position's logprob is ``-ln(V)`` and its entropy is ``ln(V)``. That lets each
case set ``generator_logprobs`` to a known offset from ``-ln(V)`` and assert the
resulting importance ratio, clip behavior, and metrics against hand-computed
values.

The loss math these tests pin (from ``losses/dapo.py``):
    ratio         = exp(clamp(trainer_logprob - generator_logprob, +/-10))
    clipped_ratio = clamp(ratio, 1 - clip_low, 1 + clip_high)
    token_loss    = -min(ratio * adv, clipped_ratio * adv)
    loss          = (token_loss * loss_mask).sum() / max(global_valid_tokens, 1)
"""

from __future__ import annotations

import math

import pytest
import torch

from torchtitan.experiments.rl.losses import DAPOLoss, GRPOLoss

_LN2 = math.log(2.0)


def _uniform_logits(batch: int, seq_len: int, vocab: int = 2) -> torch.Tensor:
    """All-zero logits -> uniform softmax -> per-token logprob == -ln(vocab)."""
    return torch.zeros(batch, seq_len, vocab, dtype=torch.float32)


def _labels(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def _row(values: list[float], *, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([values], dtype=dtype)


def _mask(values: list[bool]) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.bool)


# --- on-policy identity (ratio == 1) ---


def test_on_policy_identity_loss_equals_negative_masked_advantage():
    """generator_logprobs == trainer_logprobs -> ratio 1, no clipping.

    loss = sum(-adv over response tokens) / denom. With adv=[0, 2, -1],
    mask=[F, T, T], denom=2: loss = (-(2) + -(-1)) / 2 = -0.5.
    """
    loss_fn = GRPOLoss.Config(clip_eps=0.2).build()
    logits = _uniform_logits(1, 3)
    loss, metrics = loss_fn(
        logits,
        _labels([[0, 1, 0]]),
        2,
        generator_logprobs=_row([-_LN2, -_LN2, -_LN2]),
        advantages=_row([0.0, 2.0, -1.0]),
        loss_mask=_mask([False, True, True]),
    )
    assert loss.item() == pytest.approx(-0.5)
    # ratio == 1 on both response tokens: masked_ratio.sum()/denom = 2/2.
    assert metrics["loss/ratio_mean"].item() == pytest.approx(1.0)
    assert metrics["loss/ratio_clipped_frac"].item() == pytest.approx(0.0)
    # entropy of a uniform size-2 vocab is ln(2), averaged over 2 response tokens.
    assert metrics["trainer/entropy/mean"].item() == pytest.approx(_LN2, abs=1e-5)
    # no non-finite generator logprobs.
    assert metrics["loss/generator_logprob_nan_frac"].item() == pytest.approx(0.0)


def test_prompt_and_padding_tokens_do_not_contribute():
    """loss_mask=False tokens (prompt/pad, advantage 0) add nothing to the loss."""
    loss_fn = GRPOLoss.Config(clip_eps=0.2).build()
    logits = _uniform_logits(1, 3)
    # Give the masked-out position a wild generator logprob + advantage; it must
    # still be ignored because loss_mask is False there.
    loss, _ = loss_fn(
        logits,
        _labels([[0, 1, 0]]),
        2,
        generator_logprobs=_row([5.0, -_LN2, -_LN2]),
        advantages=_row([99.0, 2.0, -1.0]),
        loss_mask=_mask([False, True, True]),
    )
    assert loss.item() == pytest.approx(-0.5)


# --- clipping ---


def test_clip_bounds_the_ratio_and_counts_clipped_fraction():
    """generator_logprob = -ln2 - 0.5 -> raw_log_ratio 0.5 -> ratio e^0.5 ~ 1.6487,
    clamped to 1 + clip_eps = 1.2 on both response tokens.

    For adv=+2: min(1.6487*2, 1.2*2) = 2.4 -> contributes -2.4.
    For adv=-1: min(1.6487*-1, 1.2*-1) = -1.6487 -> contributes +1.6487.
    loss = (-2.4 + 1.6487) / 2.
    """
    loss_fn = GRPOLoss.Config(clip_eps=0.2).build()
    logits = _uniform_logits(1, 3)
    ratio = math.exp(0.5)
    loss, metrics = loss_fn(
        logits,
        _labels([[0, 1, 0]]),
        2,
        generator_logprobs=_row([-_LN2 - 0.5, -_LN2 - 0.5, -_LN2 - 0.5]),
        advantages=_row([0.0, 2.0, -1.0]),
        loss_mask=_mask([False, True, True]),
    )
    expected = (-2.4 + ratio * 1.0) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    # Both response tokens clip (ratio != clipped_ratio regardless of adv sign).
    assert metrics["loss/ratio_clipped_frac"].item() == pytest.approx(1.0)


def test_grpo_is_symmetric_dapo():
    """GRPOLoss(clip_eps=e) == DAPOLoss(ratio_clip_low=e, ratio_clip_high=e)."""
    args = dict(
        logits=_uniform_logits(1, 3),
        labels=_labels([[0, 1, 0]]),
        global_valid_tokens=2,
        generator_logprobs=_row([-_LN2 - 0.5, -_LN2 - 0.5, -_LN2 - 0.5]),
        advantages=_row([0.0, 2.0, -1.0]),
        loss_mask=_mask([False, True, True]),
    )
    grpo = GRPOLoss.Config(clip_eps=0.2).build()
    dapo = DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.2).build()

    grpo_loss, _ = grpo(
        args["logits"],
        args["labels"],
        args["global_valid_tokens"],
        generator_logprobs=args["generator_logprobs"],
        advantages=args["advantages"],
        loss_mask=args["loss_mask"],
    )
    dapo_loss, _ = dapo(
        args["logits"],
        args["labels"],
        args["global_valid_tokens"],
        generator_logprobs=args["generator_logprobs"],
        advantages=args["advantages"],
        loss_mask=args["loss_mask"],
    )
    assert grpo_loss.item() == pytest.approx(dapo_loss.item())


def test_clip_higher_upper_bound_keeps_more_mass_on_upweighted_token():
    """DAPO clip-higher (ratio_clip_high > ratio_clip_low): for a positive-advantage
    token whose ratio exceeds 1, a larger upper bound allows a larger surrogate,
    i.e. a more negative (larger-magnitude) loss than the symmetric clip.
    """
    logits = _uniform_logits(1, 1)
    labels = _labels([[1]])
    # raw_log_ratio 0.5 -> ratio ~ 1.6487, above both 1.2 and 1.28 upper bounds.
    gen = _row([-_LN2 - 0.5])
    adv = _row([2.0])
    mask = _mask([True])

    symmetric = DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.2).build()
    clip_higher = DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28).build()

    sym_loss, _ = symmetric(
        logits, labels, 1, generator_logprobs=gen, advantages=adv, loss_mask=mask
    )
    hi_loss, _ = clip_higher(
        logits, labels, 1, generator_logprobs=gen, advantages=adv, loss_mask=mask
    )
    # clip-higher clamps to 1.28 (vs 1.2): loss = -(1.28*2) vs -(1.2*2).
    assert hi_loss.item() == pytest.approx(-(1.28 * 2.0))
    assert sym_loss.item() == pytest.approx(-(1.2 * 2.0))
    assert hi_loss.item() < sym_loss.item()


# --- non-finite generator logprobs (#3593) ---


def test_nonfinite_generator_logprob_is_dropped_from_loss_and_denominator():
    """A response token with a non-finite generator logprob is removed from
    loss_mask AND the denominator; generator_logprob_nan_frac counts it against
    the original response mask.

    mask=[F, T, T], gen[pos1]=+inf -> only pos2 survives (adv=-1, ratio=1):
    loss = -min(-1, -1) / denom(2) = 0.5. nan_frac = 1 / 2.
    """
    loss_fn = GRPOLoss.Config(clip_eps=0.2).build()
    logits = _uniform_logits(1, 3)
    loss, metrics = loss_fn(
        logits,
        _labels([[0, 1, 0]]),
        2,
        generator_logprobs=_row([-_LN2, float("inf"), -_LN2]),
        advantages=_row([0.0, 2.0, -1.0]),
        loss_mask=_mask([False, True, True]),
    )
    assert loss.item() == pytest.approx(0.5)
    assert metrics["loss/generator_logprob_nan_frac"].item() == pytest.approx(0.5)


def test_huge_log_ratio_is_clamped_so_loss_stays_finite():
    """An enormous generator/trainer mismatch clamps |log_ratio| at 10.0 before
    exp(), so the loss never overflows to inf/NaN."""
    loss_fn = GRPOLoss.Config(clip_eps=0.2).build()
    logits = _uniform_logits(1, 3)
    loss, _ = loss_fn(
        logits,
        _labels([[0, 1, 0]]),
        2,
        # raw_log_ratio = -ln2 - (-ln2 - 100) = +100 -> clamped to 10.
        generator_logprobs=_row([-_LN2 - 100.0, -_LN2 - 100.0, -_LN2 - 100.0]),
        advantages=_row([0.0, 2.0, -1.0]),
        loss_mask=_mask([False, True, True]),
    )
    assert math.isfinite(loss.item())


# --- gradient-accumulation equivalence ---


def test_grad_accumulation_matches_single_large_batch():
    """Two microbatches summed, each normalized by the SAME global token count,
    equal one big batch. This underpins the controller normalizing the loss by
    num_global_valid_tokens across all microbatches before any backward.
    """
    loss_fn = GRPOLoss.Config(clip_eps=0.2).build()
    # Each microbatch: 2 response tokens; global valid tokens across both = 4.
    logits = _uniform_logits(1, 2)
    labels = _labels([[1, 0]])
    gen = _row([-_LN2, -_LN2])
    adv = _row([3.0, 4.0])
    mask = _mask([True, True])

    mb_a, _ = loss_fn(logits, labels, 4, generator_logprobs=gen, advantages=adv, loss_mask=mask)
    mb_b, _ = loss_fn(logits, labels, 4, generator_logprobs=gen, advantages=adv, loss_mask=mask)

    big, _ = loss_fn(
        _uniform_logits(1, 4),
        _labels([[1, 0, 1, 0]]),
        4,
        generator_logprobs=_row([-_LN2] * 4),
        advantages=_row([3.0, 4.0, 3.0, 4.0]),
        loss_mask=_mask([True] * 4),
    )
    assert (mb_a + mb_b).item() == pytest.approx(big.item(), rel=1e-6)


def test_denominator_defaults_to_one_when_global_valid_tokens_none():
    """global_valid_tokens=None -> denominator 1 (raw summed token loss)."""
    loss_fn = GRPOLoss.Config(clip_eps=0.2).build()
    logits = _uniform_logits(1, 2)
    loss, _ = loss_fn(
        logits,
        _labels([[1, 0]]),
        None,
        generator_logprobs=_row([-_LN2, -_LN2]),
        advantages=_row([2.0, 3.0]),
        loss_mask=_mask([True, True]),
    )
    # ratio 1 on both -> loss = -(2 + 3) / 1.
    assert loss.item() == pytest.approx(-5.0)


# --- bit_wise diagnostic metrics (trainer vs generator logprob divergence) ---


def test_bitwise_logprob_diff_metrics_over_response_tokens():
    """`bit_wise/*` track the trainer-vs-generator logprob gap on response tokens.

    trainer_logprob = -ln2 everywhere. gen = [-ln2, -ln2-0.5, -ln2+0.3] ->
    diff = [0, +0.5, -0.3], masked to response tokens [F, T, T], denom=2:
      logprob_diff/mean = (0.5 + -0.3) / 2 = 0.1
      ratio_tokens_different/mean = 2 / 2 = 1.0  (both response diffs exceed 1e-6)
      logprob_diff/max = max(|0|, |0.5|, |-0.3|) = 0.5
    """
    loss_fn = GRPOLoss.Config(clip_eps=0.2).build()
    logits = _uniform_logits(1, 3)
    _, metrics = loss_fn(
        logits,
        _labels([[0, 1, 0]]),
        2,
        generator_logprobs=_row([-_LN2, -_LN2 - 0.5, -_LN2 + 0.3]),
        advantages=_row([0.0, 1.0, 1.0]),
        loss_mask=_mask([False, True, True]),
    )
    assert metrics["bit_wise/logprob_diff/mean"].item() == pytest.approx(0.1, abs=1e-5)
    assert metrics["bit_wise/ratio_tokens_different/mean"].item() == pytest.approx(1.0)
    assert metrics["bit_wise/logprob_diff/max"].item() == pytest.approx(0.5, abs=1e-5)
