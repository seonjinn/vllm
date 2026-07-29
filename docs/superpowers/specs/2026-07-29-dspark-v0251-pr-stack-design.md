# DSpark vLLM 0.25.1 PR Stack Design

## Goal

Measure whether upstream DSpark correctness and performance changes improve the
existing Qwen3-235B SWE rollout benchmark without changing its model, drafter,
sampling parameters, topology, or CUDA Graph coverage.

## Fixed Baseline

- vLLM base: `v0.25.1` (`752a3a504485790a2e8491cacbb35c137339ad34`)
- Existing runtime image:
  `vllm-openai-v0.25.1-aarch64-ubuntu2404-dspark-jit-5634162.sqsh`
- Hardware: one OCI-HSG node, 4 GB200 GPUs, target TP4
- Target: `Qwen3-235B-A22B-Thinking-2507`
- Drafter: `dspark_235b_general_v1_b8/2`
- Serving: target and draft `FLASH_ATTN`, Triton MoE, BF16, FULL CUDA Graph
- Performance workload: SWE rollout, temperature 1.0, top-p 1.0, top-k -1,
  four prompts per step, 32 generations per prompt, four steps
- Existing reference: job `5668853`, 4,155.57 seconds, mean acceptance length
  1.508, token acceptance 10.159%, 0.924x non-spec throughput

The checkpoint declares `sample_from_anchor=true`. Stock vLLM 0.25.1 converts
Speculators-format DSpark checkpoints to `dspark_bonus_anchor=true`, which
forces `sample_from_anchor=false`. Correcting this layout mismatch is the first
acceptance hypothesis and must be measured before attributing the low
acceptance to CUDA Graph execution or training quality.

## Integration Stages

Each stage is a named commit boundary so results can be attributed to one class
of changes.

1. `stage0-v0251`: unmodified vLLM 0.25.1.
2. `stage1-correctness`:
   - #48167: Blackwell non-causal draft attention fix.
   - #48524: auxiliary-layer-based DFlash/DSpark FC sizing.
   - #48639: load `sample_from_anchor` from Speculators checkpoints.
   - #49617: place DSpark Speculators attributes in the nested config consumed
     by the Qwen DSpark model.
   - #48909: account for bonus-anchor query width in scheduler capacity.
3. `stage2-prefix-cache`:
   - #47926: mask prefix-cache-restored tokens whose draft context KV was not
     constructed.
4. `stage3-tp-throughput`:
   - #49731: replicate the DSpark Markov head across TP ranks to remove a draft
     all-reduce and full-vocabulary gather.
5. `stage4-startup`:
   - #48804: warm DSpark/DFlash Triton kernels at startup.
6. `stage5-dynamic-k`:
   - #47737: permit Dynamic Speculative Decoding schedules with K=0 for DSpark.
7. `stage6-fp8-head`:
   - #47584: opt-in row-wise FP8 DSpark draft LM head.

#48932 is not stacked initially because its loader change overlaps #49617 and
its repetition-penalty optimization is inactive for the current benchmark.
It receives a separate compatibility review only if #49617 is insufficient.

#48692 changes scheduling, request representation, CUDA Graph dispatch, and
attention metadata together. It is evaluated on a separate
`dspark/v0251-adaptive-20260729` branch so failures or regressions cannot
invalidate the fixed-K stack.

## Validation Matrix

### Correctness and Acceptance

Run a short greedy smoke at K=3 and K=5 after every stage:

- server reaches readiness;
- checkpoint weights load without ignored or mismatched tensors;
- all requests complete with expected token counts;
- no illegal memory access, assertion, engine death, or eager-mode downgrade;
- target and DSpark FULL CUDA Graph capture messages are present;
- per-position accepted-token counts are recorded.

For `stage1-correctness`, run prefix caching both on and off. A large
acceptance increase caused by #48639 is interpreted as checkpoint-layout
correction, not a generic throughput optimization.

For `stage2-prefix-cache`, compare:

- stage1 with prefix caching on;
- stage1 with prefix caching off;
- stage2 with prefix caching on.

The prefix-cache hypothesis is confirmed only if stage1-off and stage2-on
recover acceptance relative to stage1-on.

### Performance

Use the exact job-5668853 workload and matched non-spec baseline. Record:

- completion tok/s and tok/s/GPU;
- generation wall time and baseline-relative speedup;
- token acceptance rate;
- mean acceptance length;
- per-position acceptance;
- CUDA Graph capture/replay or fallback telemetry;
- startup time and first-request latency;
- actual and expected output token counts.

For #49731, retain the change when TP4 throughput improves outside repeat
noise. For #48804, judge cold-start and first-request latency separately from
steady-state throughput. For #47737, compare fixed K and batch-size schedules
at matched concurrency. For #47584, compare the opt-in environment variable
off and on in the same build.

## Build and Provenance

All integration work happens in the isolated vLLM worktree
`.worktrees/vllm-dspark-v0251-prstack`. Commits use `seonjinn` as committer and
are pushed to `seonjinn/vllm`.

OCI jobs clone or fetch the exact pushed commit. The first smoke may use a
source-bound editable/precompiled installation because every selected PR is
Python or Triton source. A result promoted to the final comparison must use an
immutable saved container with:

- base image path and SHA-256;
- vLLM base tag and integration commit;
- included PR list and head SHA for open PRs;
- Python, PyTorch, CUDA, FlashAttention, and FlashInfer versions;
- exact benchmark command and output directory.

## Failure Handling

- A cherry-pick conflict is resolved by comparing the PR's base implementation
  with v0.25.1, not by accepting either side wholesale.
- If a stage fails, preserve its branch and logs; do not add another PR until
  the root cause is isolated.
- If three independent repair attempts fail for the same stage, stop stacking
  and keep that PR on a separate branch.
- Correctness fixes may be retained with neutral performance. Performance
  changes are excluded from the recommended runtime if they regress matched
  throughput or destabilize output.

