# RL / GRPO Test Plan

Test plan for the async GRPO stack in `torchtitan/experiments/rl`. It documents
how each component functions, what is already covered, and the prioritized gaps
to close. Balanced across three regimes: single-turn correctness, multi-turn
correctness, and the async / off-policy pipeline.

Status legend: [x] covered today, [ ] gap to add, [~] partially covered.

---

## 1. System overview (what we are testing)

The GRPO loop is an async actor pipeline orchestrated by `Controller`
(`controller.py`). Two Monarch actors run on disjoint GPU meshes:

- `PolicyTrainer` (`actors/trainer.py`): owns the trainable model,
  `forward_backward`, `optim_step`, checkpointing, and `push_model_state_dict`
  (stages weights to TorchStore).
- `VLLMGenerator` (`actors/generator.py`): a vLLM `LLMEngine` driven by one
  continuous-batching `_engine_loop`, plus `pull_model_state_dict`.

Controller loops, connected by the bounded `RolloutGroupWorkBuffer`:

```
_data_input_loop -> _rollout_loop[N] -> _batcher_loop -> training_batch_queue -> _trainer_loop
        |                  ^                    ^
        +----- RolloutGroupWorkBuffer ---------+
```

Per-step data path:
`Rollouter.run_group_rollouts` -> per-sibling `_run_single_rollout` ->
`TokenEnv`/`MessageEnv` turns driving `generate_fn` -> `Completion` ->
`RolloutGroup` -> `TrainingSampleBuilder` (group filters + packing) ->
`Batcher` -> `TrainingBatch` -> trainer -> `WeightSyncManager`
(push -> pull -> release slots).

Generator internals: rank 0 owns the request queue + futures; `_engine_loop`
runs on all ranks; rank 0 decides a `LoopDecision` (STEP / PULL / CLOSE) and
broadcasts it over gloo; all ranks apply it in lockstep. `RequestDispatcher`
handles DP/TP routing and fans peer-DP completions back to rank 0.
`_pull_model_state_dict` pulls from TorchStore into the live vLLM model, with
special handling for `spmd_types` (DTensor wrapping via SPMD layouts),
`FusedQKVLinear`, and `FusedSwiGLU`, then resets the prefix cache.

GRPO math: `GRPOLoss` = `DAPOLoss` with a symmetric clip
(`losses/grpo.py` -> `losses/dapo.py`). Per-token clipped surrogate; non-finite
generator logprobs are dropped from loss + denominator; advantages come from
`AdvantageEstimator` (Dr.GRPO mean-baseline by default; standard GRPO with
`should_std_normalize=True`).

Single-turn vs multi-turn: identical code path. A rollout loops
`env.init()` -> `generate` -> `env.step()` until terminal. Single-turn = env
returns `done=True` on the first `step`. Multi-turn = env keeps returning
`env_messages` (bounded by `TokenEnv.max_num_turns` and `max_rollout_tokens`).
Multi-turn training correctness lives in
`TrainingSampleBuilder.rollout_to_training_samples`: prefix-preserving turns
pack into one sample; a broken prefix (history edit / compaction) opens a new
branch sample.

Off-policy: `max_offpolicy_steps` bounds staleness via buffer capacity
`(max_offpolicy_steps + 1) * num_groups_per_train_step`. 0 = fully on-policy
(sync). `WeightSyncManager` releases buffer slots only after the generator pull,
so a new rollout is "born fresh".

### PR lineage

- #3453 - introduced current abstractions: `MessageEnv`, `Rollout`/`RolloutTurn`
  types, `Rubric`, `Renderer`, the `rollout/` package. Deleted the monolithic
  `grpo.py` and `test_grpo_metrics.py` (669 lines of loss tests removed - the
  main coverage hole today).
- #3593 - continuous-batching generator + multi-turn rollouts: the
  `_engine_loop` / `RequestDispatcher` design, hot-swap weight-sync flags, GRPO
  NaN clamping (`_MAX_LOG_RATIO`, `nan_to_num`), AlphabetSort default 3 turns.
- #3602 - Search-R1: multi-turn retrieval-augmented GRPO (tool-call env +
  exact-match rubric).

---

## 2. Existing coverage (baseline)

CPU unit (no GPU / Monarch / vLLM engine):

- [x] `test_generator.py` - `RequestDispatcher` completion build, SamplingParams
  contract, vLLM metric timing math (fake engine).
- [x] `test_engine_loop.py` - `_decide_next_action` STEP/PULL/CLOSE branching.
- [x] `test_weight_sync.py` - `WeightSyncManager` push -> pull -> release order.
- [x] `test_async_controller.py` - Batcher counting, buffer backpressure,
  consume-time staleness invariant, metrics timer, `RolloutTurnID`.
- [x] `test_rollout_utils.py` - `rollout_to_training_samples` packing/branching.
- [x] `test_intra_generator_router.py`, `test_inter_generator_router.py` -
  round-robin / least-loaded / sticky routing.
- [x] `test_alphabet_sort.py` - dataset / env / rubric / rollouter (task).
- [x] `test_search_r1.py` - multi-turn tool-call env + exact-match rubric
  (retrieval monkeypatched).
- [x] `test_metrics.py`, `test_rollout_recorder.py`, `test_cast_linear.py`,
  `test_shutdown.py`.

GPU:

- [x] `test_bitwise_parity.py` (torchrun) - batch invariance; trainer == vLLM
  prefill; vLLM decode == prefill; transitively trainer == vLLM decode.
- [x] `integration_tests.py` - full GRPO loops: TP=2 (compile on/off), MoE EP=4,
  checkpoint save+resume with resharding, batch-invariant (h100 suite).

Disabled:

- [ ] `test_train_loop.py` - still skipped module-level; deferred (see 3.6), the
  invariants are covered at the component level.

Added by this work (see section 3 and "Execution status"):

- [x] `test_losses.py` - GRPO/DAPO loss math incl. `bit_wise/*` diagnostics.
- [x] `test_advantage.py` - group-relative advantage estimator.
- [x] `test_token_env.py` - TokenEnv terminal-status contract.
- [x] `test_training_sample_builder.py` - group-level filters.
- [x] `test_rubric.py` - weighted reward-fn sum, truncation/error short-circuit,
  config validation.
- [x] `test_rollouter.py` - rollout driver: single/multi-turn turn loop, error /
  truncation handling, group scoring + advantage assignment, per-sample seed
  offset.
- [x] `test_generator.py` (extended) - DP>1 fan-in send, min-version stamping,
  n=1 guard, expert-parallel-degree validation.
- [x] `test_async_controller.py` (extended) - off-policy-window slot capacity.

---

## Execution status (this host)

All new tests were **executed and pass: 54 passed** - losses 10, advantage 5,
token_env 12, training_sample_builder 5, rubric 9, rollouter 6, generator
additions 6 (DP fan-in x2, min-version stamp, n=1 guard, EP validation x2),
async_controller addition 1 (off-policy window).

No local conda env has the full RL runtime stack at once: the checkout needs
`spmd_types` (torch >= 2.10), so the `forge`-family envs (pinned torch 2.9, with
vllm/monarch/torchstore) are incompatible; `titan312` has the right torch-nightly
+ `spmd_types` but lacks `vllm`/`monarch`/`torchstore`/`renderers`, and the
`renderers` git install is network-blocked. Since `rl/__init__.py` imports
`vllm_registry` unconditionally, importing ANY rl submodule requires those deps.

These 5 test files touch vllm/monarch/renderers/torchstore ONLY through the
import chain, never at runtime, so they were run in `titan312` behind a
local-only import shim that fabricates permissive stub modules for those deps
(the code under test runs for real). The shim lives at
`/tmp/rl_stubs/sitecustomize.py` (NOT committed; see that file). How to reproduce:

```
PY=/path/to/conda/envs/titan312/bin/python
PYTHONPATH="/tmp/rl_stubs:$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  $PY -m pytest -p no:cacheprovider \
  torchtitan/experiments/rl/tests/test_losses.py \
  torchtitan/experiments/rl/tests/test_advantage.py \
  torchtitan/experiments/rl/tests/test_token_env.py \
  torchtitan/experiments/rl/tests/test_training_sample_builder.py
# generator additions (select only these; the pre-existing cudagraph/sampling
# tests in that file need REAL vllm objects and are skipped under the shim):
#   test_generator.py::test_build_completions_rejects_multiple_samples_per_request
#   test_generator.py::test_rank0_stamp_min_policy_version_marks_every_future_in_the_decision
#   test_generator.py::test_peer_dp_leader_sends_completions_over_the_port
#   test_generator.py::test_peer_dp_leader_sends_nothing_when_no_requests_finished
```

Shim caveat: it lets pure-logic tests run, but any assertion that inspects a real
vllm/renderers object is meaningless under it. That is why only the 4
non-vllm-object generator tests are run this way; the full `test_generator.py`
(and everything else) should run unshimmed in the CI RL image. As an extra check,
the numeric expectations in `test_losses.py` / `test_advantage.py` were also
independently reproduced with standalone plain-torch / `statistics` scripts, and
17 pre-existing pure tests (`test_rollout_utils`, `test_async_controller`) pass
under the same shim - confirming it does not manufacture false passes.

---

## 3. CPU unit tests (fast, CI) - the priority additions

### 3.1 GRPO / DAPO loss - `test_losses.py` (NEW, top priority) - DONE (passing)

Rationale: `test_grpo_metrics.py` was deleted in #3453 and never replaced; the
NaN-drop + clip + denominator logic has zero unit coverage today. Pure-tensor,
tiny shapes, assert against hand-computed values. All values below independently
reproduced with a standalone torch script.

- [x] On-policy identity: `trainer_logprobs == generator_logprobs` -> ratio 1,
  `loss = -(advantage) summed / denom`, `ratio_clipped_frac == 0`.
- [x] Clipping: log-ratios beyond `+/- clip_eps`; verify `ratio_clipped_frac`
  and the correct clip side for positive vs negative advantage (`torch.min`
  surrogate). GRPO symmetric vs DAPO asymmetric (`ratio_clip_high=0.28`).
- [x] NaN/Inf generator logprob (#3593): token dropped from `loss_mask` AND the
  denominator; `loss/generator_logprob_nan_frac` counts vs original
  `response_mask`.
- [x] Overflow guard: huge `raw_log_ratio` clamps at `_MAX_LOG_RATIO=10.0`; no
  inf/NaN in loss.
- [x] Grad-accum equivalence: loss over 2 microbatches with
  `global_valid_tokens=total` == one big batch (underpins the controller's
  "can't stream microbatches" invariant). Also: denominator defaults to 1 when
  `global_valid_tokens` is None.
- [x] loss_mask semantics: prompt/pad tokens (mask False, advantage 0) give zero
  gradient contribution.
- [x] Metrics: `trainer/entropy/mean` and `loss/generator_logprob_nan_frac`
  asserted; `loss/ratio_mean`, `loss/ratio_clipped_frac` asserted. (Remaining
  `bit_wise/*` keys left for a follow-up; the harder NaN/clip metrics are done.)

### 3.2 Advantage estimator - `test_advantage.py` (NEW) - DONE (passing)

All expected values reproduced standalone with `statistics.pstdev`.

- [x] Dr.GRPO (default): `A_i = r_i - mean(r)`, denom 1.
- [x] Standard GRPO: `should_std_normalize=True` -> divide by `pstdev + 1e-6`.
- [x] Zero-variance group -> all-zero advantages (feeds the zero-std drop path),
  under both estimators.
- [x] Group order preserved; single-rollout group.

### 3.3 TokenEnv contract - `test_token_env.py` (NEW; single + multi-turn core) - DONE (passing)

Rationale: all terminal-status / truncation / timeout logic for both regimes
lives here and is untested. Driven with a fake `MessageEnv` + duck-typed fake
`Renderer`.

- [x] `init` renders first prompt; prompt-too-long -> `TRUNCATED_PROMPT_TOO_LONG`
  with the prompt still carried.
- [x] `step` finish_reason `"length"` -> `TRUNCATED_LENGTH` (env not stepped,
  message kept); `"abort"` -> `ERROR_ABORT`.
- [x] parse failure -> `ERROR_PARSE`.
- [x] `MessageEnv.step` timeout -> `ERROR_TIMEOUT`.
- [x] `done=True` -> `COMPLETED` with `env_rewards` propagated.
- [x] `max_num_turns` reached (env would continue) -> `TRUNCATED_MAX_TURNS`.
- [x] next prompt over `max_rollout_tokens` -> `TRUNCATED_PROMPT_TOO_LONG`.
- [x] bridge path: `bridge_to_next_turn` returns tokens -> used directly;
  returns `None` -> full `render_ids` fallback (both yield the ongoing prompt).
- [x] single-turn: env `done` on first step -> exactly one turn, `COMPLETED`.

### 3.4 TrainingSampleBuilder group filters - `test_training_sample_builder.py` (NEW) - DONE (passing)

`rollout_to_training_samples` packing is covered in `test_rollout_utils.py`;
this new file covers `build_from_group`:

- [x] Empty group (failed generation) -> empty samples + failure metric
  passthrough.
- [x] Any sibling with no completion tokens -> whole group dropped
  (`num_groups_dropped_untrainable`).
- [x] Zero-std reward group -> dropped when `drop_zero_std_reward_groups=True`
  (+ `group_zero_std_frac`); NOT dropped when flag off.
- [x] Trainable group -> one sample per rollout + `num_training_samples`,
  `branches_per_rollout`, `min/max_policy_version` metrics.
- Multi-turn branching + non-final-empty-completion `ValueError` + single-turn
  empty-completion skip: already covered by `test_rollout_utils.py` (see it).

### 3.5 Generator - extended `test_generator.py` - DONE (passing)

Config validation, cudagraph math, SamplingParams, and DP-router load release
were already covered in `test_generator.py`. Added the genuinely-uncovered
paths:

- [x] DP>1 fan-in send: a peer DP leader (`rank=1`, DP=2) builds completions and
  ships them over the (fake) result `Port` instead of resolving locally; and
  sends nothing when no requests finished. (The rank-0 drain-recv loop itself
  needs Monarch `Channel`; resolution is covered by the existing
  `_rank0_resolve_futures` tests.)
- [x] min_policy_version stamping: `rank0_stamp_min_policy_version` marks every
  future in the STEP decision across all DP ranks.
- [x] `_build_completions` with `len(outputs) != 1` -> `ValueError`.
- [x] `Config.__post_init__` validation (batch_invariant / reset_running_requests
  / hot_swap) - pre-existing.
- [x] `VLLMCudagraphConfig.get_vllm_compilation_config` math - pre-existing.
- [x] SamplingParams contract (`n=1`, `logprobs=0`, `FINAL_ONLY`, seed/stop) -
  pre-existing.
- [x] `expert_parallel_degree in {1, DP*TP}` validation - rejects a bad EP,
  accepts full EP (dp*tp).

### 3.7 Rubric - `test_rubric.py` (NEW) - DONE (passing)

Base `Rubric` combining logic (task reward fns are covered by the example task
tests):

- [x] Weighted reward-fn sum normalized by total weight; per-fn breakdown.
- [x] Single reward fn returns its value.
- [x] `truncation_reward` / `error_reward` short-circuit the reward fns (proved
  with a reward fn that raises if called); COMPLETED rollouts still run them.
- [x] `score_group` returns one output per rollout, in order.
- [x] Validation: empty reward_fns, duplicate fn names, non-positive weight sum.

### 3.8 Rollouter - `test_rollouter.py` (NEW) - DONE (passing)

The rollout driver, single-turn and multi-turn, faked `generate_fn` + scripted
envs:

- [x] `_run_single_rollout` single-turn: one turn; request_id / sticky
  routing_session_id format.
- [x] `_run_single_rollout` multi-turn: one turn per step; turn ids increment,
  routing session stays constant.
- [x] Error mid-rollout -> status ERROR, prior turns retained.
- [x] Truncated terminal status propagates.
- [x] `run_group_rollouts`: sibling fan-out, scoring, Dr.GRPO advantage
  assignment, envs closed.
- [x] Per-sample seed offset (`base + sample_idx`) across the group.

### 3.6 Controller run loop - `test_train_loop.py` - DEFERRED (not written)

Deliberately NOT written blind. Rationale: authoring the full `Controller.run()`
harness needs a real `Rollouter` + `Renderer` (both pull in `renderers`) plus
fakes for the trainer actor, generator router, and TorchStore - a large async
test that cannot be executed on this host, so a subtle error would ship as false
confidence. The core invariants it would assert are ALREADY covered at the
component level and should be extended there instead:

- Backpressure / off-policy slot budget -> `test_async_controller.py`
  (`test_take_finalized_does_not_release_active_slot`,
  `test_untrainable_group_releases_before_training`).
- Consume-time staleness / policy age -> `test_async_controller.py`
  (`test_compute_policy_age_metrics_raises_on_consume_time_staleness`).
- Push -> pull -> release ordering -> `test_weight_sync.py`.
- Shutdown wiring -> `test_shutdown.py`.

The multi-slot off-policy-window capacity test
(`max_active = (max_offpolicy_steps+1)*num_groups_per_train_step`) is now added
in `test_async_controller.py`
(`test_offpolicy_window_caps_active_slots_and_release_unblocks`). Remaining gap:
the zero-std-only silent-hang heartbeat (once the heartbeat exists).

---

## 4. GPU integration and numerics

Run with conda env `titan312` and offline HF flags (see the gpu-test-env note).

### 4.1 Bitwise parity - `test_bitwise_parity.py` (exists)

```
conda run -n titan312 torchrun --nproc_per_node=2 -m pytest \
  torchtitan/experiments/rl/tests/test_bitwise_parity.py::TestBitwiseParityVarlen -v
# repeat ::TestBitwiseParityFlex
```

Chain: batch invariance -> trainer == vLLM prefill -> vLLM decode == prefill ->
(transitively) trainer == vLLM decode. This is the RL-specific correctness bar:
generator logprobs must match the trainer recompute.

- [ ] Extend: parity under TP>1.
- [ ] Extend: a multi-turn sequence (decode across a bridged 2nd turn).

### 4.2 End-to-end - `integration_tests.py` (exists)

```
conda run -n titan312 python -m torchtitan.experiments.rl.tests.integration_tests \
  $OUTPUT_DIR --ngpu 8 --hf_assets_path <qwen3-0.6b>
```

Existing flavors: TP=2 (compile on/off), MoE EP=4, checkpoint save+resume with
resharding, batch-invariant (h100 suite). Add:

- [ ] Multi-turn E2E: AlphabetSort with >1 names-per-turn batches +
  `max_num_turns`; assert multi-turn branching survives packing and loss stays
  finite.
- [ ] Search-R1 multi-turn (#3602) with a mocked/local retrieval server
  (nightly).
- [ ] DP>1 generator (`generator.parallelism.data_parallel_degree=2`) to
  exercise the fan-in path end to end.
- [ ] Async off-policy (`max_offpolicy_steps=2`) vs sync (`0`): assert reward
  delta pre/post > 0 on the learnable AlphabetSort task and policy-age metrics
  stay within bound.

### 4.3 Learning / loss guard

Per the repo numerics rule: with `--debug.seed=42 --debug.deterministic` and a
batch-invariant config, two runs must produce bitwise-identical loss and
grad_norm from TensorBoard (stdout's 5 digits are insufficient). Follow
`scripts/loss_compare.py`. Reference: `tests/assets/losses/rl_grpo_cuda.txt`.

---

## 5. Priority order and status

1. [DONE] Loss unit tests, incl. `bit_wise/*` (3.1) - `test_losses.py`.
2. [DONE] TokenEnv contract (3.3) - `test_token_env.py`.
3. [DEFERRED] Controller run loop (3.6) - not written blind; invariants covered
   at the component level (see 3.6).
4. [DONE] build_from_group filters (3.4) - `test_training_sample_builder.py`
   (multi-turn branching already in `test_rollout_utils.py`).
5. [DONE] Generator DP>1 fan-in, min-version stamping, EP validation (3.5) -
   added to `test_generator.py`.
6. [DONE] Advantage estimator (3.2) - `test_advantage.py`; CUDA-graph config math
   already covered in `test_generator.py`.
7. [DONE] Rubric combining logic (3.7) - `test_rubric.py`.
8. [DONE] Rollouter turn loop + group orchestration (3.8) - `test_rollouter.py`.
9. [DONE] Off-policy-window slot capacity (3.6) - `test_async_controller.py`.
10. [TODO] GPU: TP>1 + multi-turn parity; multi-turn / DP>1 / async E2E flavors.
11. [TODO] Zero-std-only silent-hang heartbeat (needs the heartbeat first).

New files added: `test_losses.py`, `test_advantage.py`, `test_token_env.py`,
`test_training_sample_builder.py`, `test_rubric.py`, `test_rollouter.py`;
extended `test_generator.py` and `test_async_controller.py`. See "Execution
status" for how each was validated on this host.

Run the CPU suite (in an env with the full RL stack -
vllm+renderers+monarch+torchstore+spmd_types, e.g. the CI RL image):

```
pytest torchtitan/experiments/rl/tests/ -x \
  --deselect torchtitan/experiments/rl/tests/test_bitwise_parity.py
```
