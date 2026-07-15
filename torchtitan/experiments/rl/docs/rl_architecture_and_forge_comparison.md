# TorchTitan RL (GRPO): Architecture, API Reference, and Comparison with Forge

A single reference for understanding the GRPO reinforcement-learning paradigm in
`torchtitan/experiments/rl` -- what every trainer and generator API does, how
weight sync works, how the message queue / rollout buffer works, the known
hurdles, and how the tests map to the components. The second half compares
TorchTitan-RL with **Forge** (`/home/hosseinkh/pytorch/torchforge`) and explains
how the in-tree design evolved from it.

Everything here is grounded in code; file:symbol references are given so you can
jump straight to the source.

---

## Table of contents

1. The GRPO paradigm in one page
2. TorchTitan-RL architecture (the big picture)
3. Generator API reference (`VLLMGenerator`)
4. Trainer API reference (`PolicyTrainer`)
5. The GRPO loss
6. Weight sync (TorchStore + `WeightSyncManager`)
7. The message queue and rollout buffer
8. Single-turn vs multi-turn
9. Known hurdles / TODOs / limitations
10. What the tests cover (and how they were run)
11. Forge: the system this evolved from
12. Side-by-side comparison
13. How TorchTitan-RL evolved from Forge
14. Glossary

---

## 1. The GRPO paradigm in one page

**GRPO (Group Relative Policy Optimization)** is a critic-free policy-gradient
method. The loop:

1. **Generate**: for a prompt, sample a *group* of G answers (siblings).
2. **Score**: grade each answer with a reward (rubric / verifier).
3. **Advantage**: center rewards within the group -- `A_i = r_i - mean(r)`
   (optionally `/ std`). No value network; the group *is* the baseline.
4. **Learn**: increase the probability of above-average answers, decrease
   below-average, with a PPO-style clipped importance ratio for stability.

The importance ratio is the heart of it:

```
ratio_t = exp( logprob_trainer(token_t) - logprob_generator(token_t) )
loss_t  = -min( ratio_t * A , clip(ratio_t, 1-eps, 1+eps) * A )
```

- `logprob_generator` = how likely the *sampling* policy thought the token was
  (recorded at generation time).
- `logprob_trainer`  = how likely the *current* policy thinks it is now.

Because generation and training are different workloads, they run as **two
separate model copies on two separate GPU pools**: a fast sampler (vLLM) and a
learner (TorchTitan). Keeping those two copies in sync is "weight sync", and
bounding how far the sampler lags the learner is the "off-policy window".

---

## 2. TorchTitan-RL architecture (the big picture)

One **`Controller`** (`controller.py`) owns everything and runs four concurrent
asyncio loops connected by one bounded buffer:

```
_data_input_loop -> _rollout_loop[N] -> _batcher_loop -> training_batch_queue -> _trainer_loop
      (get prompt)   (generate+score)    (pack tokens)      (size-1 mailbox)       (learn+sync)
        |                   ^                    ^
        +------- RolloutGroupWorkBuffer --------+
```

Two Monarch actors live on disjoint GPU meshes, spawned in
`Controller.setup_async`:

- **`PolicyTrainer`** (`actors/trainer.py`) -- the learner.
- **`VLLMGenerator`** (`actors/generator.py`) -- the sampler (one or more
  replicas behind an `InterGeneratorRouter`).

Supporting non-actor components (plain `Configurable` objects the controller
builds): `Rollouter` (drives rollouts), `Rubric` (scores), `AdvantageEstimator`,
`TrainingSampleBuilder`, `Batcher`, `WeightSyncManager`, `RolloutGroupWorkBuffer`.

The controller is a `Configurable`: its entire configuration is one typed
dataclass tree (`Controller.Config`) built via `.build()`. Swapping a loss, env,
reward, dataset, or advantage estimator is a config change, not a code change.

---

## 3. Generator API reference (`VLLMGenerator`)

`actors/generator.py`. A Monarch `Actor` wrapping a vLLM `LLMEngine`. Rank 0 owns
the request queue and the awaited futures; all ranks run one background engine
loop in lockstep.

### Endpoints (the RPC surface the controller calls)

| Endpoint | Signature | What it does |
|---|---|---|
| `start_engine_loop()` | `()` | Starts the background `_engine_loop` on every rank (idempotent). Must run before any generate/pull. |
| `generate(...)` | `(prompt_token_ids, *, request_id, routing_session_id, sampling_config=None, metrics_prefix="generator") -> Completion` | Queue one prompt, await its `Completion`. Rank-0 only (via `call_one`). |
| `pull_model_state_dict(version)` | `(version: int)` | Queue a weight pull; block until the engine loop applies it. |
| `sync_log_step(step, relative_step=None)` | | Sync the structured-logger step counter. |
| `close()` | `()` | Stop the engine loop, fail outstanding futures, release renderer resources. |

### How generation runs (the engine loop)

`_engine_loop` (`generator.py`) is the heartbeat. Each iteration:

1. **Rank 0 decides** a `LoopDecision` in `_decide_next_action`: `STEP`,
   `PULL_MODEL_STATE_DICT`, or `CLOSE`. It sleeps on a condition variable until
   there is work.
2. The decision is **broadcast over gloo (CPU)** to all ranks so they act
   identically -- `engine.step()` is a TP collective and every rank must call it.
3. On `STEP`: admit any newly-queued requests (`engine.add_request`), then run a
   **burst** of `engine.step()` up to `max_engine_steps_between_decisions` (16)
   before re-deciding. The burst lets new requests batch together instead of
   forcing a prefill between every decode step -- this is **continuous batching**:
   a new request can join the moving batch mid-flight.
4. Finished `RequestOutput`s are turned into `Completion`s and their futures
   resolved.

### `RequestDispatcher` (DP/TP fan-in)

When the generator spans multiple GPUs, `RequestDispatcher` hides the rank
layout (`global_rank = dp_rank * tp_degree + tp_rank`):

- Rank 0 routes each queued request to a DP replica (`IntraGeneratorRouter`),
  stamps the admitted (min) policy version on each future, and resolves its own
  replica's completions locally.
- A peer DP leader (tp_rank 0 of another replica) builds completions and **sends
  them back to rank 0 over a Monarch `Port`** ("fan-in"), because only rank 0
  holds the futures the caller awaits.

### Sampling contract

`_build_sampling_params` always sets **`n=1`** (one sample per request),
`logprobs=0` (return the sampled token's logprob, needed for the GRPO ratio),
and `output_kind=FINAL_ONLY`. A group of G siblings is G separate `n=1` requests,
each with a per-sample **seed offset** (`base_seed + sample_idx`) so siblings are
diverse yet reproducible.

Config highlights (`VLLMGenerator.Config`): `parallelism` (DP/TP/EP),
`sampling`, `model_dtype`, `gpu_memory_limit`, `cudagraph`,
`max_engine_steps_between_decisions`, and the weight-sync reset flags
(`reset_prefix_cache_on_weight_sync`, `reset_running_requests_on_weight_sync`).

---

## 4. Trainer API reference (`PolicyTrainer`)

`actors/trainer.py`. A Monarch `Actor` that builds the model directly from a
`ModelSpec` (via `model_spec.parallelize_fn`), a real optimizer, and a
`CheckpointManager`. Unlike a normal TorchTitan `Trainer`, it exposes forward and
optimizer as **separate endpoints** so the controller can overlap weight sync
between them.

### Endpoints

| Endpoint | Signature | What it does |
|---|---|---|
| `forward_backward(...)` | `(training_data: list[TrainingMicrobatch], num_global_valid_tokens: int) -> dict[str,float]` | Pick this DP rank's microbatch, forward, GRPO loss, `backward()`. Returns globally-reduced metrics. Does NOT step the optimizer. |
| `optim_step()` | `() -> OptimStepOutput` | Clip grads, `optimizer.step()`, LR step, zero grads, bump `policy_version`. |
| `push_model_state_dict()` | `()` | Stage weights to TorchStore (CPU) for generators to pull. |
| `get_policy_version()` | `() -> int` | Current version (for resume / initial sync). |
| `save_checkpoint(step, last_step=False)` | `-> bool` | DCP checkpoint via `CheckpointManager`. |
| `sync_log_step(step, ...)` | | Sync logger step. |
| `close()` | `()` | Release actor-local resources (PG teardown is owned by `ProcMesh.stop`). |

### The training step (in the controller's `_trainer_loop`)

```
packed = await training_batch_queue.get()          # a TrainingBatch (or None -> stop)
for microbatch in packed.microbatches:
    await trainer.forward_backward(microbatch, packed.num_global_valid_tokens)
await weight_sync.wait_prev_push()                 # finish prior push before mutating weights
await trainer.optim_step()                          # weights change here
await weight_sync.wait_prev_pull()                 # finish prior pull before next push
weight_sync.start_async_push_pull(version)          # overlap this step's sync with next fwd/bwd
```

Notes: RL does not support pipeline parallelism yet (`forward_backward` asserts
exactly one model part). The loss is normalized by `num_global_valid_tokens`
(summed across all microbatches and DP ranks) so gradient accumulation equals one
big batch -- which is why microbatches cannot be streamed.

---

## 5. The GRPO loss

`losses/grpo.py` -> `losses/dapo.py`. `GRPOLoss` is `DAPOLoss` with a symmetric
clip. Per token:

```
raw_log_ratio = trainer_logprob - generator_logprob
# A non-finite generator logprob (vLLM under cudagraph) is DROPPED from loss + denom.
log_ratio = clamp(nan_to_num(raw_log_ratio), -10, +10)   # overflow guard
ratio     = exp(log_ratio)
clipped   = clamp(ratio, 1 - clip_low, 1 + clip_high)     # GRPO: low == high
token_loss = -min(ratio * adv, clipped * adv)
loss = (token_loss * loss_mask).sum() / max(global_valid_tokens, 1)
```

`DAPOLoss` allows asymmetric bounds (clip-higher, e.g. `ratio_clip_high=0.28`).
There is **no KL term and no reference model** -- stability comes entirely from
the clip. Advantages come from `AdvantageEstimator` (`rollout/advantage.py`):
Dr.GRPO (mean baseline, `denom=1`) by default, standard GRPO
(`should_std_normalize=True`, `denom=std+1e-6`) optionally.

Diagnostic metrics the loss emits: `loss/ratio_clipped_frac`,
`loss/generator_logprob_nan_frac`, `trainer/entropy/mean`, and
`bit_wise/logprob_diff/{mean,max}` (trainer-vs-generator logprob gap -- the
signal for on-policy parity).

---

## 6. Weight sync (TorchStore + `WeightSyncManager`)

After `optim_step`, the trainer's weights are newer than the generator's copy.

**Producer (`PolicyTrainer.push_model_state_dict`)**: `ts.put_state_dict(
state_dict, "model_state_dict", direct_rdma=False)`. `direct_rdma=False` stages
GPU->CPU, so the trainer's GPU is free immediately and any number of generators
can read the staged copy. If the generator dtype differs, params are cast here
(buffers excluded).

**Consumer (`VLLMGenerator._pull_model_state_dict`)**: `ts.get_state_dict(
"model_state_dict", ...)` into the live vLLM model, then `reset_prefix_cache` so
no new request reuses KV computed under old weights. For the `spmd_types`
backend, each local tensor is wrapped as a `DTensor` via its declared SPMD layout
so TorchStore fills the right shard; fused params (`FusedQKVLinear`,
`FusedSwiGLU`) get special-cased FQN handling.

**Key detail:** a **single key** `"model_state_dict"` is overwritten every step.

**Overlap (`components/weight_sync.py`):** `WeightSyncManager.start_async_push_pull`
fires push -> pull -> (release buffer slots) in the background so the network
transfer hides behind the next step's forward/backward. The loop only *waits* for
the previous sync right before it would clobber the weights again
(`wait_prev_push` before optim, `wait_prev_pull` before the next push). Buffer
slots are released only *after* the pull completes -- the "born-fresh" guarantee
(see next section).

---

## 7. The message queue and rollout buffer

The four loops are decoupled by `RolloutGroupWorkBuffer`
(`components/work_buffer.py`) and one size-1 `asyncio.Queue`.

### `RolloutGroupWorkBuffer` -- a FIFO work buffer that is really an off-policy valve

A work item moves WAITING -> INFLIGHT -> FINALIZED. Public API:
`wait_for_slot()`, `add_work()`, `claim_next()`, `finalize_work()`,
`take_finalized()`, `release_active_groups(count, reason)`, `close()`.

The capacity **is** the off-policy window:

```
max_active = (max_offpolicy_steps + 1) * num_groups_per_train_step
```

- `max_offpolicy_steps = 0` -> fully synchronous (generator and trainer alternate).
- `max_offpolicy_steps = k` -> generation may run k train-steps ahead, then
  blocks (no free slots) until a train step releases its groups.

Slots are released by `release_active_groups(..., "trained")` only *after* the
weight pull completes, which enforces the **born-fresh invariant**: a new rollout
always starts on weights no more than `max_offpolicy_steps` behind. Consumed
staleness is re-checked at train time by `compute_policy_age_metrics`, which
raises if a sample is older than allowed.

### The four loops and their backpressure

| Loop | Consumes | Produces | Waits on |
|---|---|---|---|
| `_data_input_loop` | dataset | `RolloutGroupWork` | a free slot (`wait_for_slot`) |
| `_rollout_loop[N]` | a WAITING item | a scored `RolloutGroup` | a claimable item |
| `_batcher_loop` | oldest FINALIZED group | `TrainingBatch` | a free queue slot (maxsize=1) |
| `_trainer_loop` | a `TrainingBatch` | trained weights | a batch in the queue |

`TrainingSampleBuilder` turns rollout groups into trainable token sequences and
drops failed / untrainable / zero-std-reward groups; `Batcher` packs surviving
samples into microbatches at `num_groups_per_train_step`.

---

## 8. Single-turn vs multi-turn

Both use the same driver (`rollout/rollouter.py::_run_single_rollout`):

```
env_step = env.init()
while not env_step.status.is_terminal():
    completion = generate_fn(env_step.next_prompt_token_ids, ...)
    env_step   = env.step(completion)
```

- **Single-turn** = the env returns `done=True` on the first `step`.
- **Multi-turn** = the env keeps replying (tool result, follow-up), bounded by
  `TokenEnv.max_num_turns` / `max_rollout_tokens`.

`TokenEnv` (`environment/token.py`) translates token-space <-> message-space via a
renderer and enforces every terminal status (`COMPLETED`, `TRUNCATED_LENGTH`,
`TRUNCATED_MAX_TURNS`, `ERROR_PARSE`, `ERROR_TIMEOUT`, ...). Multi-turn training
correctness lives in `TrainingSampleBuilder.rollout_to_training_samples`:
prefix-preserving turns pack into one sample; an edited history "branches" into a
new sample. Multi-turn is a **first-class, tested feature** here.

---

## 9. Known hurdles / TODOs / limitations

- **No pipeline parallelism** in the trainer (`forward_backward` asserts one
  model part); no context parallelism validated with batch-invariant mode.
- **Generator is vLLM-only** and requires varlen or flex attention; the model
  must be registerable to vLLM. `generator.py` carries a
  `TODO: Split a backend-agnostic BaseGenerator`.
- **The generate backend is not yet config-pluggable**: the `GenerateFn` protocol
  abstracts it, but the controller wires `VLLMGenerator` + router directly
  (`_make_generate_fn` has a `TODO: make this a pluggable config`).
- **Single-key weight sync** means exactly one staged copy; no double-buffering.
- **Zero-std-only stream can silently hang** the trainer (no batch ever packs) --
  a known TODO to add a heartbeat.
- **Batch-invariant / bitwise parity** only holds under matched parallelism and
  bf16 forward (FSDP mixed precision); SP/CP not supported in that mode.
- **Resume** restores model/optimizer/policy_version but not the in-flight buffer
  or dataset stream position.

---

## 10. What the tests cover (and how they were run)

`tests/` -- 54 new unit tests were added on top of the existing suite. They are
CPU-only in the sense that they exercise pure logic; the numeric expectations were
independently reproduced with standalone torch/statistics scripts.

| Component | Test file | Focus |
|---|---|---|
| GRPO/DAPO loss | `test_losses.py` | ratio/clip math, NaN-drop, overflow clamp, grad-accum equivalence, `bit_wise` metrics |
| Advantage | `test_advantage.py` | Dr.GRPO vs std-normalized, zero-variance |
| TokenEnv | `test_token_env.py` | every terminal status, bridge vs re-render (single + multi-turn) |
| Sample packing | `test_training_sample_builder.py` | failed / untrainable / zero-std group filters |
| Rubric | `test_rubric.py` | weighted reward sum, truncation/error short-circuit, validation |
| Rollouter | `test_rollouter.py` | single/multi-turn loop, error handling, group scoring + advantage, seed offset |
| Generator | `test_generator.py` | DP fan-in send, min-version stamping, n=1 guard, EP validation, cudagraph config |
| Buffer | `test_async_controller.py` | off-policy window capacity, backpressure, staleness invariant |
| Weight sync | `test_weight_sync.py` | push -> pull -> release ordering/overlap |
| Numerics (GPU) | `test_bitwise_parity.py` | trainer == vLLM prefill == vLLM decode |
| End-to-end (GPU) | `integration_tests.py` | full GRPO loops: TP, MoE EP, checkpoint resume |

**Deferred (honestly):** the full `Controller.run()` loop test -- it needs the
real Rollouter/Renderer plus heavy fakes and could not be executed on this host,
so its invariants are covered piece-by-piece instead. The environment on this host
had no single env with the full RL stack (vllm+renderers+monarch+torchstore+
spmd_types); the pure tests were run behind a local import shim in the env with
the correct torch + spmd_types. See `docs/test_plan.md` for the full status.

---

## 11. Forge: the system this evolved from

Forge (`/home/hosseinkh/pytorch/torchforge`, "a PyTorch-native agentic RL
library") is the predecessor. Its README now states: *"Development in Forge has
paused. LLM training at PyTorch is being consolidating in torchtitan."* Forge is
built **on top of** torchtitan (it imports `torchtitan.experiments.forge.engine`).

### Shape: services + actors, not a monolithic controller

The GRPO app (`apps/grpo/main.py`) is an **imperative script** with two async
loops -- `continuous_rollouts()` and `continuous_training()` -- calling a fleet of
independent Monarch actors and services:

```python
generator = await Generator.options(**cfg.services.generator).as_service(**cfg.generator)
trainer   = await TitanTrainer.options(**cfg.actors.trainer).as_actor(**cfg.trainer, loss=loss_fn)
replay_buffer = await ReplayBuffer.options(...).as_actor(...)
ref_model = await ReferenceModel.options(...).as_service(...)   # optional (KL)
reward_actor  = await RewardActor.options(...).as_service(...)
compute_advantages = await ComputeAdvantages.options(...).as_actor()
```

Two placement primitives on `ForgeActor` (`controller/actor.py`):

- **`.as_actor()`** -- one addressable instance (one ProcMesh).
- **`.as_service()`** -- a pool of fault-tolerant, load-balanced **replicas**.

And Monarch "adverbs" to invoke them:

- **`.route()`** -- load-balanced single call to one replica (services).
- **`.fanout()`** -- broadcast to all replicas (services).
- **`.call()` / `.call_one()`** -- whole-mesh / singleton call (actors).

A global **`Provisioner`** (`controller/provisioner.py`) allocates hosts, GPUs
(`GpuManager`), and ProcMeshes; a **`Service`** wraps replicas with a router
(round-robin / least-loaded / sticky-session), a health loop, request migration,
and recovery. This "controller" role -- placement, health, retry, load balancing
-- is decomposed into infrastructure, not a single loop.

### Forge generator

`actors/vllm/v1/generator.py`. Wraps vLLM **`AsyncLLM`** with a custom
`ForgeMonarchExecutor` so vLLM's workers are Monarch actors. `generate(prompt)`
is **async and per-prompt**; a group is one request with **`n=group_size`** in the
sampling params (contrast: TorchTitan issues G separate `n=1` requests).
Continuous batching is entirely inside vLLM.

### Forge weight sync: ping-pong + shared-memory prefetch

`_torchstore_utils.py`: keys are `policy_ver_{0|1}.{param}` where the slot is
`step % 2` -- **ping-pong** between two storage slots, reusing allocations instead
of incrementing/deleting versions (commit "ping-pong weight sync #763").

- Trainer `push_weights(version)` converts to **HF format** (`sd_adapter.to_hf`)
  and `ts.put_batch(entries)` (per-param keys).
- Generator `update_weights(version)`: optionally **prefetch weights into POSIX
  shared memory** (`_WeightFetcher` actors) overlapping with
  `pause_generation(wait_for_inflight_requests=True, clear_cache=True)`, then
  workers `model.load_weights([(name, param.cuda())])`, then `resume_generation()`.

### Forge trainer

`actors/trainer/titan.py`. `TitanTrainer` wraps torchtitan's **`ForgeEngine`**
(not a native model build) and exposes a **fused `train_step(list[TrainBatch])`**
(one `forward_backward` + `all_reduce(loss)` + optim + checkpoint), plus
`push_weights`. A formal `Trainer` Protocol exists in `api/trainer.py` but
`TitanTrainer` only partially conforms.

### Forge replay buffer

`actors/replay_buffer.py`. A genuine **replay buffer**: `add(episode)` and
`sample(curr_policy_version)` with **random sampling**, **age-based eviction**
(`max_policy_age` = the config's `off_by_n`), and optional **resampling**
(`max_resample_count`). It reshapes into `(dp_size, batch_size, ...)` and collates.

### Forge losses and reference model

Five losses (`rl/loss/`): **GRPO** (PPO clip + optional **KL via `beta` and a
`ReferenceModel`**), **DAPO** (dual-clip), **GSPO** (sequence-level ratio),
**CISPO** (REINFORCE-style detached ratio), **SAPO** (soft sigmoid gate). All
built from shared `ops.py` primitives (`compute_ratio`, `compute_kl`,
`pg_ppo_clip`, `aggregate`). Advantage (`rl/advantage.py`) is GRPO group-relative
with **z-score normalization** (`(r-mean)/(std+1e-4)`).

The **`ReferenceModel`** is a separate forward-only actor built on torchtitan
`ForgeEngine` that returns shifted-target logprobs for the KL term -- used only
when the loss needs it.

---

## 12. Side-by-side comparison

| Axis | TorchTitan-RL (`experiments/rl`) | Forge (`torchforge`) |
|---|---|---|
| Position | In-tree successor; native to torchtitan | Standalone library on top of torchtitan (paused) |
| Orchestration | One `Controller` + 4 structured asyncio loops | Imperative app loop + services/actors + `Provisioner` |
| Actor framework | Monarch | Monarch |
| Placement | Controller spawns 2 actors on explicit meshes; `InterGeneratorRouter` for N generators | `.as_service()` (replica pool) / `.as_actor()`; `Provisioner` + `Service`/`Replica` + routers |
| Invocation | direct endpoint calls + a generator router | Monarch adverbs `.route`/`.fanout`/`.call`/`.call_one` |
| Trainer | `PolicyTrainer`: native `ModelSpec` build; **separate** `forward_backward` + `optim_step` | `TitanTrainer`: wraps `ForgeEngine`; **fused** `train_step` |
| Generator | `VLLMGenerator`: sync `LLMEngine` + custom continuous-batching loop + `RequestDispatcher`; **n=1** per request | `Generator`: vLLM `AsyncLLM` + `ForgeMonarchExecutor`; **n=group_size** |
| Sampling diversity | per-sample seed offset over `n=1` requests | vLLM `n>1` sampling |
| Weight transport | single key `"model_state_dict"`, overwritten; `WeightSyncManager` overlap | ping-pong 2-slot keys, HF format, `put_batch`; shm prefetch + pause/resume |
| Off-policy control | **FIFO work buffer**, strict born-fresh window `(k+1)*groups` | **replay buffer**: random sample, age eviction, resample; `off_by_n` |
| Advantage | Dr.GRPO mean baseline (std optional), a `Configurable` | GRPO z-score (mean/std), a small actor |
| Losses | GRPO, DAPO (clip only, no KL) | GRPO(+KL), DAPO, GSPO, CISPO, SAPO |
| Reference model / KL | none in-loop | optional `ReferenceModel` service + `beta` |
| Multi-turn | first-class (`MessageEnv`/`TokenEnv`/renderer, branching packing), tested | GRPO app is single-turn (GSM8K); agentic via external envs |
| Bitwise parity | explicit goal (unified model, batch-invariant mode, parity tests) | not a central theme |
| Config | typed `Configurable` dataclass tree, `.build()` | OmegaConf YAML + `services`/`actors` resource blocks + `.options()` |
| Data model | `Completion` (lists; `min`/`max_policy_version`) | `Completion` (tensors; single `generator_version`) + `Episode` |
| Reward | `Rubric` + `RewardFn` (weighted, truncation/error short-circuit) | `RewardActor` service + reward functions |
| PP / CP | not supported | not supported (PP explicitly `NotImplementedError`) |

---

## 13. How TorchTitan-RL evolved from Forge

The lineage is explicit: Forge proved the pattern (Monarch actors + vLLM sampler +
torchtitan learner + TorchStore weight sync), then LLM RL was **consolidated into
torchtitan** and re-architected in-tree. The key shifts:

1. **From an infra library to a single controller.** Forge's power is its
   generality: a `Provisioner`, load-balanced `Service` replicas with health /
   recovery / sticky sessions, and adverb-addressable actors -- the "controller"
   is decomposed into infrastructure you compose in an app script. TorchTitan-RL
   trades that generality for a single, legible `Controller` with an explicit
   4-loop pipeline. Less infrastructure to reason about; the orchestration is one
   file, not a service mesh. (The service/replica machinery, autoscaling, and V2
   interfaces in Forge were themselves partly aspirational -- e.g. "nested actors
   disabled ... temporary", autoscaling metrics-only.)

2. **From a replay buffer to an off-policy window.** Forge samples randomly from a
   replay buffer with age-based eviction and resampling -- flexible, but staleness
   is a distribution, not a bound. TorchTitan-RL uses a FIFO work buffer whose
   capacity *is* the off-policy bound and releases slots only after the pull, giving
   a hard **born-fresh** guarantee and a consume-time staleness assertion.

3. **From a wrapped engine to a native trainer.** Forge's `TitanTrainer` wraps
   `ForgeEngine` and fuses the step. TorchTitan-RL's `PolicyTrainer` builds the
   model natively from a `ModelSpec` and **splits** `forward_backward` from
   `optim_step` -- precisely so weight sync can overlap between them.

4. **A unified model + bitwise-parity story.** TorchTitan-RL runs the *same*
   torchtitan model definition in both the trainer and inside vLLM, and invests
   heavily in **batch invariance** and trainer-vs-generator logprob parity
   (`test_bitwise_parity.py`, `bit_wise/*` metrics). This addresses a class of
   subtle RL correctness bugs and was not a central theme in Forge.

5. **First-class multi-turn.** The `MessageEnv`/`TokenEnv`/renderer stack and
   prefix-preserving/branching training-sample packing make multi-turn a tested
   core path, versus Forge's single-turn GRPO app.

6. **Simpler weight sync, richer loss set (each way).** TorchTitan-RL's
   single-key + overlap manager is simpler than Forge's ping-pong + shm prefetch
   (Forge's is arguably more allocation-efficient and lower-latency). Conversely
   Forge ships more loss variants (GSPO/CISPO/SAPO) and an in-loop KL reference
   model that TorchTitan-RL does not (yet) include.

Net: **Forge optimized for infrastructure generality and scale-out; TorchTitan-RL
optimizes for legibility, correctness (bitwise parity), a hard off-policy bound,
and multi-turn -- as an in-tree experiment rather than a separate library.**

---

## 14. Glossary

- **Rollout / Episode**: one prompt -> (multi-turn) generation trajectory.
- **Group**: G sibling rollouts of the same prompt; the GRPO baseline unit.
- **Policy version**: monotonically increasing count of optimizer steps; stamps
  weights and completions.
- **Off-policy step / policy age**: how many train-steps a sample lags the current
  policy.
- **Born-fresh**: invariant that a new rollout starts on weights no older than
  `max_offpolicy_steps`.
- **Weight sync**: copying trainer weights into the generator (TorchStore).
- **Continuous batching**: admitting new generation requests into an already-running
  decode batch instead of waiting for it to drain.
- **Bitwise parity**: trainer and generator computing identical logprobs for the
  same tokens (batch-invariant kernels + matched dtype/parallelism).
- **ForgeEngine**: torchtitan's `experiments/forge` engine that Forge's trainer and
  reference model wrap.
- **Adverbs**: Monarch call verbs -- `.route`/`.fanout` (Forge services),
  `.call`/`.call_one` (actor meshes).
</content>
