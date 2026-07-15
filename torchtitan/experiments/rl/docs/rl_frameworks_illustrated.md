# RL Frameworks, Illustrated: TorchTitan-RL vs Forge vs Slime

An educational, slide-style walkthrough of the **APIs** of three PyTorch RL
post-training stacks, with small runnable-looking code snapshots, tiny test
snippets, side-by-side comparisons, and the design concepts behind each.

Read it top to bottom like a deck: each "SLIDE" is one concept, usually with the
same idea shown in all three frameworks so you can compare directly.

The three systems:

| | TorchTitan-RL | Forge | Slime |
|---|---|---|---|
| Repo | `torchtitan/experiments/rl` | `meta-pytorch/torchforge` | `THUDM/slime` |
| Trainer engine | TorchTitan (native) | TorchTitan (`ForgeEngine`) | **Megatron-Core** |
| Rollout engine | **vLLM** | **vLLM** | **SGLang** |
| Orchestration | **Monarch** (1 Controller) | **Monarch** (services/actors) | **Ray** (actor groups) |
| Weight transport | TorchStore (1 key) | TorchStore (ping-pong) | NCCL / IPC / disk / **delta** |
| Off-policy control | FIFO off-policy window | random replay buffer | Data Buffer + `update_weights_interval` |
| Lineage | in-tree successor to Forge | paused; folded into TorchTitan | independent; behind GLM models |

One-liners:
- **TorchTitan-RL**: one legible `Controller`, unified model shared with vLLM,
  hard off-policy bound, bitwise-parity focus. In-tree experiment.
- **Forge**: a general "services + actors" RL *infrastructure* library on top of
  TorchTitan (development paused, consolidating into TorchTitan).
- **Slime**: battle-tested Megatron+SGLang stack behind frontier GLM releases;
  deep single-backend integration, rich weight-sync + agentic rollout.

---

## SLIDE 0 -- How to read the snippets

Snippets are trimmed to the essential API surface (imports and error handling
elided). File references point at the real source so you can dive in. `# ...`
means "code omitted".

---

## SLIDE 1 -- The universal RL loop (the mental model)

Every framework implements the same GRPO cycle; only the *machinery* differs.

```
        +-------------------- weights ---------------------+
        v                                                  |
   [GENERATOR] --answers--> [SCORE] --reward--> [ADVANTAGE] --> [TRAINER]
    (sample G                (rubric/            (r - mean)/std   (clipped
     answers)                 verifier)          per group        policy grad)
```

- **Generate** a group of G answers per prompt.
- **Score** each (reward model / verifier / rubric).
- **Advantage** = center rewards within the group (critic-free baseline).
- **Train** with a clipped importance-ratio policy gradient.
- **Sync** the new weights back to the generator.

Keep this picture; every API below is one box or one arrow.

---

## SLIDE 2 -- Orchestration substrate: how you get compute

**TorchTitan-RL -- one Controller spawns two actors on disjoint meshes:**
```python
# controller.py :: setup_async
self.trainer = trainer_mesh.spawn("trainer", PolicyTrainer, config.trainer, ...)
generator   = generator_mesh.spawn("generator", VLLMGenerator, config.generator, ...)
self.generator_router = config.generator_router.build(generators=generators)
```

**Forge -- actors and load-balanced service pools, placed by a Provisioner:**
```python
# apps/grpo/main.py
generator = await Generator.options(procs=1, num_replicas=4, with_gpus=True).as_service(**cfg.generator)
trainer   = await TitanTrainer.options(procs=8, with_gpus=True).as_actor(**cfg.trainer, loss=loss_fn)
# call verbs: service -> .route()/.fanout(); actor -> .call()/.call_one()
completions = await generator.generate.route(prompt)     # one load-balanced replica
await generator.update_weights.fanout(version)           # broadcast to all replicas
```

**Slime -- one Ray placement group, sliced into train + rollout, SPMD actor groups:**
```python
# train.py
pgs = create_placement_groups(args)                       # 1 PG, carved by bundle offset
rollout_manager, n = create_rollout_manager(args, pgs["rollout"])   # SGLang engines (Ray actors)
actor_model, critic_model = create_training_models(args, pgs, rollout_manager)  # Megatron actors
# actor_model is a RayTrainGroup: one ray.remote(MegatronTrainRayActor) per GPU rank
```

**Concept:** Monarch (Titan/Forge) gives *mesh*-native actors and collective-style
adverbs; Ray (Slime) gives per-process actors grouped into an SPMD
`torch.distributed` world. Forge adds a whole service layer (replicas, routers,
health) that Titan-RL deliberately collapses into a single controller.

---

## SLIDE 3 -- The main training loop

**TorchTitan-RL -- 4 concurrent asyncio loops + a bounded buffer (structured):**
```python
# controller.py :: run  (simplified)
rollout_tasks = [asyncio.create_task(self._rollout_loop(...)) for _ in range(max_active_groups)]
data_task     = asyncio.create_task(self._data_input_loop(...))
batcher_task  = asyncio.create_task(self._batcher_loop(...))
trainer_task  = asyncio.create_task(self._trainer_loop(queue, num_training_steps=N))
# the buffer's capacity IS the off-policy window; the trainer loop is the finite clock
```

**Forge -- two imperative async loops calling services (flat, hackable):**
```python
# apps/grpo/main.py
async def continuous_rollouts():
    while not shutdown:
        prompt, target = (await dataloader.sample.call_one()).values()
        responses = await generator.generate.route(prompt)          # n = group_size
        # ... build episodes, reward, advantages ...
        await replay_buffer.add.call_one(episode)

async def continuous_training():
    while training_step < max_steps:
        batch = await replay_buffer.sample.call_one(curr_policy_version=training_step)
        await trainer.train_step.call(batch)                        # fused step
        await trainer.push_weights.call(training_step)
        await generator.update_weights.fanout(training_step)
```

**Slime -- a synchronous for-loop over rollouts (explicit dataflow):**
```python
# train.py
for rollout_id in range(args.num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))  # SGLang rollout
    if args.offload_rollout: ray.get(rollout_manager.offload.remote())       # free GPUs (colocate)
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))           # Megatron train
    if args.offload_rollout: ray.get(rollout_manager.onload_weights.remote())
    actor_model.update_weights()                                             # push to SGLang
```

**Concept:** Titan-RL hides overlap inside a pipeline of loops; Forge exposes the
loop to you (write your own orchestration with adverbs); Slime keeps it a linear,
inspectable script and time-shares GPUs with offload (`train_async.py` pipelines
one rollout ahead for the disaggregated case).

---

## SLIDE 4 -- Trainer API

**TorchTitan-RL -- `PolicyTrainer`: forward and optimizer are SEPARATE endpoints**
```python
# actors/trainer.py
@endpoint
async def forward_backward(self, training_data, num_global_valid_tokens) -> dict:
    loss, metrics = self.loss_fn(pred, labels, num_global_valid_tokens,
                                 generator_logprobs=..., advantages=..., loss_mask=...)
    loss.backward()                          # no optimizer step here
    return reduced_metrics
@endpoint
async def optim_step(self) -> OptimStepOutput:
    grad_norm = clip_grad_norm_(...); self.optimizers.step(); self.policy_version += 1
```
Why split? So the controller can overlap weight sync between `forward_backward`
and `optim_step`.

**Forge -- `TitanTrainer`: one FUSED `train_step`, wraps TorchTitan `ForgeEngine`**
```python
# forge/actors/trainer/titan.py
@endpoint
async def train_step(self, batches: list[TrainBatch]):
    batch = batches[self.engine.dp_rank]
    loss = self.forward_backward(batch)      # logits = model(**batch.model_inputs); loss = self.loss(logits, **batch.loss_inputs)
    torch.distributed.all_reduce(loss)
    self.engine.optimizers.step(); self.engine.lr_schedulers.step()
    self.engine.checkpointer.save(curr_step=self.step, ...)
```

**Slime -- `MegatronTrainRayActor.train`: Megatron pipeline engine, role = actor/critic**
```python
# slime/ray/train_actor.py (abstract)      slime/backends/megatron_utils/actor.py (impl)
def train(self, rollout_id, rollout_data_ref, external_data=None): ...
def update_weights(self): ...
def sleep(self, tags); def wake_up(self, tags)   # offload / onload GPU memory
# a step runs Megatron's forward_backward_func over packed micro-batches, then optimizer.step()
```

**Concept:** Titan-RL = fine-grained, RL-aware endpoints. Forge = a coarse
train_step that reuses TorchTitan wholesale. Slime = a Megatron SPMD group where
one class serves both actor and critic and offloads via sleep/wake_up.

---

## SLIDE 5 -- Generator / rollout API

**TorchTitan-RL -- `VLLMGenerator.generate`: ONE completion per call (`n=1`)**
```python
# actors/generator.py
completion = await generator.slice(hosts=0, gpus=0).generate.call_one(
    prompt_token_ids, request_id="group=3/rollout=0/turn=0",
    routing_session_id="group=3/rollout=0",
)
# a group of G siblings = G separate n=1 calls, each with base_seed + sample_idx
```

**Forge -- `Generator.generate`: a whole group per call (`n=group_size`)**
```python
# vllm sampling_params: n = group_size
responses: list[Completion] = await generator.generate.route(prompt)   # G answers
```

**Slime -- `RolloutManager.generate`: returns a whole training batch; HTTP to SGLang**
```python
# slime/ray/rollout.py  ->  slime/rollout/sglang_rollout.py
url = f"http://{router_ip}:{router_port}/generate"
output = await post(url, {"sampling_params": sp, "return_logprob": True})   # async HTTP
# group fan-out: one asyncio task per sample; logprobs come inline in meta_info
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
```

**Concept:** Titan-RL drives the vLLM engine loop itself (custom continuous
batching, `RequestDispatcher` for DP/TP) and keeps `n=1` for exact per-sample
control; Forge leans on vLLM `AsyncLLM` with `n=group_size`; Slime treats the
inference engine as an external **SGLang HTTP service** behind a router and lets
you swap the whole rollout function by dotted path (`--rollout-function-path`).

---

## SLIDE 6 -- Weight sync (the biggest differentiator)

**TorchTitan-RL -- single TorchStore key, overwritten; overlap manager**
```python
# trainer:  actors/trainer.py
await ts.put_state_dict(state_dict, "model_state_dict", direct_rdma=False)   # GPU->CPU stage
# generator: actors/generator.py
await ts.get_state_dict("model_state_dict", user_state_dict=model_sd, ...)
self._engine.reset_prefix_cache(...)                                         # drop stale KV
# components/weight_sync.py: start_async_push_pull overlaps push->pull->release with next step
```

**Forge -- ping-pong two-slot keys (reuse allocations), shm prefetch**
```python
# _torchstore_utils.py
def get_param_key(v, name): return f"policy_ver_{v % 2:010d}.{name}"   # slot 0 or 1
# trainer: ts.put_batch([(get_param_key(version, n), p) for n, p in hf_state_dict])
# generator.update_weights: prefetch to POSIX shared memory, then
await self.llm.pause_generation(wait_for_inflight_requests=True, clear_cache=True)
await self.workers.apply_prefetched_weights.call(fetched)     # model.load_weights([...])
await self.llm.resume_generation()
```

**Slime -- FOUR modes (mode x transport), chosen by config**
```python
# actor.py :: init
if update_weight_mode == "delta":            cls = UpdateWeightFromDiskDelta   # zstd byte-diff on disk
elif update_weight_transport == "disk":      cls = UpdateWeightFromDisk        # full HF checkpoint on disk
elif args.colocate:                          cls = UpdateWeightFromTensor      # CUDA IPC (shared GPUs)
else:                                         cls = UpdateWeightFromDistributed # NCCL broadcast
# NCCL mode: a group "slime-pp_{i}" spans training rank0 + all engine GPUs; source rank broadcasts buckets
# delta mode: snapshot baseline, XOR-diff each tensor, zstd-compress, publish only changed shards
```

**Concept:** this is where the philosophies show most.
- Titan-RL: simplest possible (one key), correctness via prefix-cache reset +
  overlap. One staged copy.
- Forge: latency/alloc-optimized (ping-pong avoids per-step alloc/free; shm avoids
  a network hop when generator and trainer share a host).
- Slime: maximal flexibility for scale -- colocated GPUs use zero-copy CUDA IPC;
  disaggregated pools use NCCL broadcast; cross-cluster / cross-vendor use disk;
  huge models use compressed **delta** sync to cut wire bytes.

---

## SLIDE 7 -- Buffer / data flow

**TorchTitan-RL -- FIFO work buffer whose capacity is the off-policy bound**
```python
# components/work_buffer.py  (states: WAITING -> INFLIGHT -> FINALIZED)
max_active = (max_offpolicy_steps + 1) * num_groups_per_train_step
await buffer.wait_for_slot()           # blocks producers when the window is full
await buffer.release_active_groups(num_groups_per_train_step, reason="trained")  # AFTER the pull
# -> "born-fresh": a new rollout always starts <= max_offpolicy_steps behind
```

**Forge -- a genuine replay buffer (random sample, age eviction, resampling)**
```python
# forge/actors/replay_buffer.py
@endpoint
async def add(self, episode): self.buffer.append(BufferEntry(episode))
@endpoint
async def sample(self, curr_policy_version):
    self._evict(curr_policy_version)                       # drop entries older than max_policy_age
    idx = random.sample(range(len(self.buffer)), total)    # random, may resample
    return self.collate(reshaped_episodes)                 # (dp_size, batch_size, ...)
```

**Slime -- a "Data Buffer": prompt source + partial-rollout recycler (FIFO drain)**
```python
# slime/rollout/data_source.py :: RolloutDataSourceWithBuffer
def get_samples(self, num_samples):
    samples = self._get_samples_from_buffer(num_samples)   # pop_first (FIFO), aborted/partial first
    if num_samples - len(samples) > 0:
        samples += super().get_samples(num_samples - len(samples))   # top up from dataset
    return samples
# add_samples() pushes half-finished (aborted) groups back for the next rollout (partial rollout)
```

**Concept:** the buffer encodes each framework's off-policy stance. Titan-RL: a
*hard bound* (window capacity). Forge: a *soft distribution* (random sample +
age eviction). Slime: a *recycling queue* for partial/async rollouts, with
staleness bounded by `--update-weights-interval`.

---

## SLIDE 8 -- Advantage and loss

**TorchTitan-RL -- Dr.GRPO mean baseline; symmetric-clip GRPO**
```python
# rollout/advantage.py
A_i = (r_i - mean(r)) / (std(r)+1e-6 if should_std_normalize else 1.0)     # default: denom = 1
# losses/dapo.py (GRPO = symmetric)
ratio = exp(clamp(trainer_logprob - generator_logprob, -10, 10))
loss  = -min(ratio*A, clamp(ratio, 1-eps, 1+eps)*A) * loss_mask / global_valid_tokens
# non-finite generator logprob -> token DROPPED (no KL / reference model)
```

**Forge -- z-score normalized advantage; GRPO with optional KL (beta + ref model)**
```python
# forge/rl/advantage.py
advantages = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-4)
# forge/rl/loss/grpo.py
L = max(-r*A, -clip(r, 1-eps_low, 1+eps_high)*A) + beta * KL_k3(logp, ref_logp)   # beta>0 needs ReferenceModel
```

**Slime -- estimator dispatch (grpo/gspo/cispo/ppo/reinforce++); clipped PG + dual-clip**
```python
# slime/utils/ppo_utils.py
@torch.compile
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    ratio = (-ppo_kl).exp()
    l1, l2 = -ratio*advantages, -ratio.clamp(1-eps_clip, 1+eps_clip_high)*advantages
    loss = torch.maximum(l1, l2)
    if eps_clip_c is not None:                       # dual-clip PPO
        loss = torch.where(advantages < 0, torch.min(-eps_clip_c*advantages, loss), loss)
# GRPO advantage = scalar reward broadcast, then distributed_masked_whiten across DP; KL k1/k2/k3, GAE, CISPO, GSPO
```

**Concept:** same clipped-PG core everywhere. Differences: **normalization**
(Dr.GRPO no-std vs z-score vs distributed whiten), **KL** (Titan none; Forge
optional ref model; Slime KL-in-reward *and* KL-in-loss), and **breadth** (Slime
also has PPO+GAE with a critic, REINFORCE++, GSPO, CISPO, TIS/ICEPOP off-policy
correction).

---

## SLIDE 9 -- Off-policy / async control (how far can the sampler lag?)

| | Mechanism | Knob | Guarantee |
|---|---|---|---|
| TorchTitan-RL | buffer slot budget released after pull | `max_offpolicy_steps` | HARD bound; born-fresh; consume-time assert |
| Forge | replay-buffer age eviction | `off_by_n` (= `max_policy_age`) | soft; old episodes evicted, staleness is a distribution |
| Slime (sync) | offload + serialized loop (colocate) | none needed | fully on-policy |
| Slime (async) | pipeline 1 rollout ahead (disaggregated) | `--update-weights-interval` | bounded, ~1-step off-policy |

```python
# Titan-RL: consume-time staleness is enforced, not just hoped for
compute_policy_age_metrics(trainer_policy_version, min_policy_versions, max_offpolicy_steps)  # raises if too old
# Slime async (train_async.py): start next rollout BEFORE training current
rollout_next = rollout_manager.generate.remote(rollout_id + 1)
ray.get(actor_model.async_train(rollout_id, rollout_curr))
```

---

## SLIDE 10 -- Multi-turn / agentic

**TorchTitan-RL -- first-class `MessageEnv` / `TokenEnv`, tested branching packing**
```python
# rollout/rollouter.py :: _run_single_rollout
env_step = await env.init()
while not env_step.status.is_terminal():
    completion = await generate_fn(env_step.next_prompt_token_ids, request_id=..., routing_session_id=...)
    env_step   = await env.step(completion)   # tool result / follow-up; TokenEnv enforces terminals
# TrainingSampleBuilder packs prefix-preserving turns into 1 sample; edited history -> branch
```

**Forge -- GRPO app is single-turn; agentic via external harnesses/envs**
```python
# apps/grpo/main.py is single prompt -> group of responses -> reward (GSM8K)
# multi-turn/tool use lives outside the core loop (coder.py, OpenEnv-style envs)
```

**Slime -- multi-turn / agent as a custom generate function (per-sample pluggable)**
```python
# --custom-generate-function-path swaps the inner per-sample generation
custom = load_function(sample.generate_function_path or args.custom_generate_function_path)
sample = await custom(args, sample, sampling_params, ...)   # may loop tools; may return list[Sample]
# session_id enables sticky routing so a multi-turn session hits the same SGLang worker
```

**Concept:** Titan-RL bakes multi-turn into the type system (envs + renderer +
branching packing). Slime keeps it maximally free-form (any Python async function
producing `Sample`s), which is how agentic/tool/sandbox workloads plug in without
forking the trainer. Forge's GRPO example stays single-turn.

---

## SLIDE 11 -- Config surface

**TorchTitan-RL -- one typed `Configurable` dataclass tree, `.build()`**
```python
config = rl_grpo_qwen3_0_6b_varlen()          # a Controller.Config
config.trainer.loss = GRPOLoss.Config(clip_eps=0.2)
config.async_loop.max_offpolicy_steps = 3
controller = config.build()
```

**Forge -- OmegaConf YAML with `services` / `actors` resource blocks**
```yaml
services:   {generator: {procs: 1, num_replicas: 1, with_gpus: true}}
actors:     {trainer: {procs: 8, with_gpus: true}, replay_buffer: {procs: 1, with_gpus: false}}
trainer:    {model: {name: qwen3, flavor: 1.7B}, parallelism: {tensor_parallel_degree: 1}}
replay_buffer: {max_policy_age: 1}   # off_by_n
```

**Slime -- flat CLI args, pass-through to Megatron and SGLang**
```bash
python3 train.py --colocate \
  --advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28 \
  --rollout-batch-size 32 --n-samples-per-prompt 8 --global-batch-size 256 \
  --update-weight-mode full --update-weight-transport nccl \
  --sglang-mem-fraction-static 0.7        # any SGLang arg via --sglang- prefix
```

**Concept:** Titan-RL = typed + programmatic (IDE-checkable). Forge = declarative
YAML separating *resources* from *model config*. Slime = native pass-through
(every Megatron and SGLang flag available, no wrapper) -- powerful but flat.

---

## SLIDE 12 -- Testing the APIs (small, real snippets)

**TorchTitan-RL -- pure-logic pytest with hand-verified numbers (what we added)**
```python
# tests/test_losses.py
def test_on_policy_identity_loss_equals_negative_masked_advantage():
    loss_fn = GRPOLoss.Config(clip_eps=0.2).build()
    logits = torch.zeros(1, 3, 2)                       # uniform -> logprob = -ln(2)
    loss, m = loss_fn(logits, labels=[[0,1,0]], global_valid_tokens=2,
                      generator_logprobs=[[-ln2,-ln2,-ln2]], advantages=[[0,2,-1]],
                      loss_mask=[[False,True,True]])
    assert loss.item() == pytest.approx(-0.5)           # ratio==1 -> -(2 + -1)/2
    assert m["loss/ratio_clipped_frac"].item() == 0.0
```

**Forge -- unittest API-shape / version smoke checks**
```python
# tests/unit_tests/test_generator_version.py
def test_generator_has_required_methods(self):
    from forge.actors.generator import Generator
    for method in ["launch", "setup", "generate", "shutdown"]:
        self.assertTrue(hasattr(Generator, method))
```

**Slime -- pytest CPU loss tests: closed-form value + gradient-flow**
```python
# tests/test_cispo_loss.py
def test_compute_cispo_loss_gradient_flows_only_through_log_probs(...):
    pg_losses, _ = compute_cispo_loss(ppo_kl, log_probs, advantages, eps_clip, eps_clip_high)
    pg_losses.sum().backward()
    assert log_ratios.grad is None or torch.all(log_ratios.grad == 0)   # IS ratio is stop-grad
```

**Concept:** all three unit-test the *loss math on CPU* (the silent-bug-prone
part). Titan-RL and Slime pin exact numeric/gradient behavior; Forge leans on
API-shape + integration tests (`test_policy_update.py`,
`test_titan_fwd_vs_hf_fwd.py`) and end-to-end model runs. Slime additionally has
full end-to-end model tests (`test_qwen2.5_0.5B_short.py`, etc.).

How to actually run the TorchTitan-RL pure tests on a box without the full stack
is documented in `docs/test_plan.md` (a local import shim + `titan312`); the full
suite runs unshimmed in the CI RL image.

---

## SLIDE 13 -- The big comparison matrix

| Concept | TorchTitan-RL | Forge | Slime |
|---|---|---|---|
| Trainer backend | TorchTitan native (`ModelSpec`) | TorchTitan `ForgeEngine` | Megatron-Core |
| Rollout backend | vLLM `LLMEngine` (custom loop) | vLLM `AsyncLLM` | SGLang HTTP server + router |
| Orchestrator | Monarch, 1 `Controller` | Monarch, services + `Provisioner` | Ray, placement groups |
| Trainer API | `forward_backward` + `optim_step` (split) | `train_step` (fused) | `train` + `update_weights` + sleep/wake |
| Generator API | `generate` (n=1) | `generate.route` (n=G) | `RolloutManager.generate` (HTTP) |
| Group sampling | G x n=1 requests, seed offset | vLLM n=group_size | async task per sample |
| Weight transport | TorchStore, 1 key + overlap | TorchStore ping-pong + shm | IPC / NCCL / disk / delta |
| Buffer | FIFO off-policy window | random replay buffer | Data Buffer (FIFO recycler) |
| Off-policy bound | hard (`max_offpolicy_steps`) | soft (`off_by_n` eviction) | interval (`update_weights_interval`) |
| Advantage | Dr.GRPO (no std default) | z-score | whiten; grpo/gspo/cispo/ppo/rpp |
| KL / ref model | none | optional `ReferenceModel` | KL-in-reward + KL-in-loss + OPD |
| Losses | GRPO, DAPO | GRPO, DAPO, GSPO, CISPO, SAPO | PPO, GRPO, GSPO, CISPO, RPP |
| Parallelism (train) | FSDP/TP/EP (no PP/CP) | FSDP/TP (no PP) | TP/PP/CP/EP/vPP (full Megatron) |
| Colocate train+infer | no (disjoint meshes) | no | YES (offload) or disaggregated |
| Multi-turn | first-class + tested | external | custom generate fn (free-form) |
| Config | typed dataclasses | OmegaConf YAML | flat CLI pass-through |
| Bitwise parity | explicit goal + tests | not central | logprob-parity via TIS/rollout-logprobs |
| Maturity | in-tree experiment | paused | production (GLM releases) |

---

## SLIDE 14 -- Design concept deep-dive

**(a) Colocate vs disaggregated.** Slime is the only one that *colocates* the
trainer and the inference engine on the same GPUs and time-shares them via
memory **offload** (`sleep`/`wake_up`, `release/resume_memory_occupation`). This
maximizes GPU utilization on a fixed cluster but forces a synchronous loop.
Titan-RL and Forge always **disaggregate** (separate GPU pools), which enables
async overlap but needs more GPUs. Slime's `train_async.py` also supports
disaggregated + one-step-ahead pipelining.

**(b) Single-backend depth vs multi-backend abstraction.** Slime bets entirely on
SGLang (+Megatron) and passes their flags through verbatim -- it can use
backend-specific features (PD disaggregation, session-affinity routing, delta
sync) without a lowest-common-denominator wrapper. Titan-RL/Forge wrap vLLM
behind their own abstractions (a `GenerateFn` protocol, a service). Trade-off:
depth + upstream velocity (Slime) vs a cleaner internal contract (Titan/Forge).

**(c) Unified model + bitwise parity.** Titan-RL runs the *same* TorchTitan model
definition in both the trainer and inside vLLM and invests in batch-invariant
kernels so trainer and generator logprobs match bitwise. Slime instead corrects
the train/rollout logprob gap statistically (`--use-rollout-logprobs`, TIS,
ICEPOP). Two philosophies for the same silent-bug class: eliminate the gap vs
correct for it.

**(d) Orchestration philosophy.** Forge decomposes "the controller" into reusable
infrastructure (Provisioner + Services + routers + health/recovery) -- powerful,
but parts were aspirational (nested actors disabled, autoscaling metrics-only).
Titan-RL collapses all of that into one readable `Controller`. Slime uses Ray's
mature actor/placement primitives and keeps the driver a plain script.

**(e) The off-policy knob is the soul of each design.** A hard window (Titan), a
soft eviction age (Forge), or a sync interval + offload (Slime) each reflect a
different answer to "how stale may training data be?" -- the central RL-systems
question.

---

## SLIDE 15 -- Pros and cons / when to pick which

**TorchTitan-RL**
- Pros: one legible controller; hard off-policy guarantee; unified model + bitwise
  parity; first-class tested multi-turn; typed config; in-tree with TorchTitan.
- Cons: vLLM-only, no PP/CP yet, single-key weight sync (no double-buffer),
  experimental; generate backend not yet config-pluggable.
- Pick when: you want a clean, correctness-first, hackable in-tree RL loop on
  TorchTitan models and value reproducibility over raw scale features.

**Forge**
- Pros: general services/actors infra (replicas, routing, health); flexible
  placement; multiple losses; clean adverb API; ping-pong + shm weight sync.
- Cons: development paused; some infra aspirational; wraps `ForgeEngine`
  (fused step, partial Trainer-protocol conformance); single-turn GRPO app.
- Pick when: you're studying the services/actors pattern or need its flexible
  multi-replica placement -- but note it's being folded into TorchTitan.

**Slime**
- Pros: production-proven (GLM); full Megatron parallelism (TP/PP/CP/EP); SGLang
  depth; four weight-sync modes incl. delta; colocate-with-offload for GPU
  efficiency; maximally free-form agentic rollout; rich off-policy correction.
- Cons: heavier stack (Megatron + SGLang + Ray); flat CLI config; more moving
  parts; steeper learning curve; opinionated single rollout backend.
- Pick when: you're training large / MoE models at scale, need PP/CP or delta
  sync, want colocation on a fixed cluster, or want battle-tested frontier-grade
  RL infra.

---

## SLIDE 16 -- One-line cheat sheet

- **Generate**: Titan `generate.call_one` (n=1) | Forge `generate.route` (n=G) |
  Slime `rollout_manager.generate.remote` (SGLang HTTP).
- **Train**: Titan `forward_backward`+`optim_step` | Forge `train_step` |
  Slime `actor_model.async_train`.
- **Sync weights**: Titan `put/get_state_dict("model_state_dict")` |
  Forge `push_weights`/`update_weights.fanout` (ping-pong) |
  Slime `actor_model.update_weights()` (IPC/NCCL/disk/delta).
- **Bound staleness**: Titan buffer window | Forge replay age | Slime
  update-weights-interval / colocate-sync.
- **Advantage**: Titan Dr.GRPO | Forge z-score | Slime whiten (+GAE for PPO).

---

## Appendix -- source pointers

TorchTitan-RL: `torchtitan/experiments/rl/{controller,actors/trainer,actors/generator}.py`,
`components/{weight_sync,work_buffer,training_sample_builder}.py`,
`losses/{grpo,dapo}.py`, `rollout/{rollouter,advantage}.py`,
`environment/{message,token}.py`; tests in `tests/`; see also
`docs/rl_architecture_and_forge_comparison.md` and `docs/test_plan.md`.

Forge: `apps/grpo/main.py`, `src/forge/actors/{generator,trainer/titan,replay_buffer,
reference_model}.py`, `src/forge/actors/_torchstore_utils.py`,
`src/forge/controller/{provisioner,service/*}.py`, `src/forge/rl/{advantage,loss/*}.py`.

Slime: `train.py`, `train_async.py`, `slime/ray/{placement_group,actor_group,
train_actor,rollout}.py`, `slime/backends/megatron_utils/{actor,loss,update_weight/*}.py`,
`slime/rollout/{sglang_rollout,data_source}.py`, `slime/utils/ppo_utils.py`.
</content>
