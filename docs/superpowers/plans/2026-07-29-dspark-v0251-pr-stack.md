# DSpark vLLM 0.25.1 PR Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and benchmark a reproducible vLLM 0.25.1 DSpark PR stack for the existing Qwen3-235B SWE rollout.

**Architecture:** Preserve v0.25.1 as the fixed base and create one commit boundary per correctness or performance stage. Validate source changes locally, validate imports and unit tests in the existing ARM64 CUDA container, then run matched OCI-HSG GPU A/B jobs before promoting a stage.

**Tech Stack:** vLLM 0.25.1, Python 3.12, PyTorch 2.11+cu130, CUDA 13.0, Triton, FlashAttention, SLURM/pyxis, Qwen3-235B, DSpark.

## Global Constraints

- Preserve the job-5668853 target, drafter, TP4 topology, sampling settings, workload, and CUDA Graph capture sizes.
- Commit and push exact source before submitting a GPU job.
- Use `seonjinn` as git committer.
- Store logs and machine-readable results in a dedicated experiment directory.
- Monitor every submitted job for at least five minutes after it starts running.
- Never combine unmatched prefix-cache, K, sampling, model, or container rows in one speedup claim.

---

### Task 1: Record the v0.25.1 baseline and PR manifest

**Files:**
- Create: `tools/dspark_pr_stack/pr_manifest.json`
- Create: `tools/dspark_pr_stack/README.md`
- Test: `tests/tools/test_dspark_pr_stack_manifest.py`

**Interfaces:**
- Consumes: vLLM tag `v0.25.1` and GitHub PR metadata.
- Produces: a machine-readable ordered stage manifest used by build receipts and benchmark reports.

- [ ] **Step 1: Write a manifest schema test**

The test must require the base commit, ordered stages, PR number, source head
SHA, activation mode, and expected benchmark effect.

- [ ] **Step 2: Run the manifest test and verify it fails**

Run:

```bash
python3 -m pytest tests/tools/test_dspark_pr_stack_manifest.py -q
```

Expected: failure because the manifest does not exist.

- [ ] **Step 3: Add the manifest and README**

Record stages 0 through 6 exactly as specified in the design document. Mark
#47584 as opt-in and #48692 as a separate branch.

- [ ] **Step 4: Run the manifest test**

Run:

```bash
python3 -m pytest tests/tools/test_dspark_pr_stack_manifest.py -q
```

Expected: all manifest tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/dspark_pr_stack tests/tools/test_dspark_pr_stack_manifest.py
git commit -s -m "docs: record DSpark v0.25.1 PR stack"
```

### Task 2: Apply the correctness stage

**Files:**
- Modify: files changed by PRs #48167, #48524, #48639, #49617, and #48909.
- Test: upstream tests added or modified by those PRs.

**Interfaces:**
- Consumes: `stage0-v0251`.
- Produces: `stage1-correctness`, with Speculators checkpoint layout and loader semantics corrected.

- [ ] **Step 1: Cherry-pick merged correctness PRs**

Cherry-pick upstream squash commits for #48167, #48524, and #48639 in that
order.

- [ ] **Step 2: Port open loader and scheduler fixes**

Cherry-pick #49617 and #48909 commits in their original order. Resolve conflicts
against v0.25.1 by preserving v0.25.1 APIs and the PR behavior.

- [ ] **Step 3: Run static validation**

Run:

```bash
git diff v0.25.1 --check
python3 -m compileall -q \
  vllm/transformers_utils/configs/speculators \
  vllm/model_executor/models/qwen3_dflash.py \
  vllm/v1/worker/gpu/spec_decode
```

- [ ] **Step 4: Run targeted container tests**

Run the PR-provided config, DSpark, scheduler, and lookahead tests inside the
ARM64 CUDA container. Record the full command and pass/fail count.

- [ ] **Step 5: Tag and push**

```bash
git tag dspark-v0251-stage1-correctness-20260729
git push fork dspark/v0251-prstack-20260729
git push fork dspark-v0251-stage1-correctness-20260729
```

### Task 3: Port and validate prefix-cache masking

**Files:**
- Modify: files changed by #47926.
- Test: `tests/v1/spec_decode/test_dflash_prefix_cache_masking.py`

**Interfaces:**
- Consumes: `stage1-correctness`.
- Produces: `stage2-prefix-cache`.

- [ ] **Step 1: Apply #47926**

Cherry-pick its head commit. When the MRV2 runner API differs from v0.25.1,
trace request admission, cached-token bookkeeping, and draft block-table
preparation before resolving conflicts.

- [ ] **Step 2: Run static and targeted tests**

Run the new prefix-cache kernel test plus the stage1 test set in the ARM64 CUDA
container.

- [ ] **Step 3: Run the three-arm acceptance smoke**

Submit stage1 prefix-on, stage1 prefix-off, and stage2 prefix-on with identical
K, prompts, seeds, and CUDA Graph settings.

- [ ] **Step 4: Tag and push**

Tag the verified commit as `dspark-v0251-stage2-prefix-cache-20260729`.

### Task 4: Apply TP communication optimization

**Files:**
- Modify: files changed by #49731.
- Test: PR-provided DSpark model and numerical-equivalence tests.

**Interfaces:**
- Consumes: `stage2-prefix-cache`.
- Produces: `stage3-tp-throughput`.

- [ ] **Step 1: Cherry-pick #49731 commits**

Preserve the PR's replicated Markov embedding/projection semantics and weight
loading.

- [ ] **Step 2: Run static and container tests**

Verify tensor-parallel weight loading and numerical equivalence.

- [ ] **Step 3: Run matched TP4 A/B**

Run stage2 and stage3 with prefix caching in the configuration selected by
Task 3. Repeat enough times to distinguish the reported 3–4% effect from run
noise.

- [ ] **Step 4: Tag and push**

Tag the retained commit as `dspark-v0251-stage3-tp-throughput-20260729`.

### Task 5: Apply startup warm-up and dynamic-K support

**Files:**
- Modify: files changed by #48804 and #47737.
- Test: their warm-up and Dynamic Speculative Decoding tests.

**Interfaces:**
- Consumes: `stage3-tp-throughput`.
- Produces: `stage4-startup` and `stage5-dynamic-k`.

- [ ] **Step 1: Apply and test #48804**

Measure engine initialization and first-request latency separately from
steady-state throughput.

- [ ] **Step 2: Tag the startup stage**

Tag as `dspark-v0251-stage4-startup-20260729`.

- [ ] **Step 3: Apply and test #47737**

Exercise a schedule containing K=0 and verify that all required FULL CUDA
Graphs are captured without division by zero or eager downgrade.

- [ ] **Step 4: Run fixed-K versus scheduled-K A/B**

Keep model, workload, and maximum K fixed. Change only the batch-size schedule.

- [ ] **Step 5: Tag and push**

Tag as `dspark-v0251-stage5-dynamic-k-20260729`.

### Task 6: Evaluate the opt-in FP8 draft head

**Files:**
- Modify: files changed by #47584.
- Test: `tests/v1/spec_decode/test_dspark_fp8_draft_head.py`

**Interfaces:**
- Consumes: `stage5-dynamic-k`.
- Produces: `stage6-fp8-head`, with an off-by-default runtime option.

- [ ] **Step 1: Cherry-pick #47584**

Confirm that the Qwen3 DSpark loader reaches the new FP8 head path before
running performance tests.

- [ ] **Step 2: Run numerical and capture tests**

Verify draft argmax agreement, target verification isolation, and FULL CUDA
Graph capture.

- [ ] **Step 3: Run environment-variable off/on A/B**

Compare the same stage6 build with `VLLM_DSPARK_FP8_DRAFT_HEAD=0` and `1`.

- [ ] **Step 4: Retain or revert**

Retain only if the Qwen3 path is active and matched throughput does not regress.
Tag a retained result as `dspark-v0251-stage6-fp8-head-20260729`.

### Task 7: Evaluate adaptive DSpark separately

**Files:**
- Modify: files changed by #48692 on branch `dspark/v0251-adaptive-20260729`.
- Test: PR-provided adaptive scheduling, output, attention, and CUDA Graph tests.

**Interfaces:**
- Consumes: the last verified fixed-K correctness stage.
- Produces: an independent adaptive candidate, not a dependency of the fixed-K stack.

- [ ] **Step 1: Create the adaptive branch**

Branch from the last verified correctness/acceptance commit.

- [ ] **Step 2: Port #48692**

Resolve MRV2 API drift component by component. Do not mix repair commits with
the fixed-K branch.

- [ ] **Step 3: Run full targeted tests and smoke**

Require FULL CUDA Graph operation, correct token accounting, and no regression
in fixed-width mode before benchmarking adaptive mode.

- [ ] **Step 4: Run matched fixed versus adaptive benchmark**

Report mean effective K, acceptance, throughput, and concurrency distribution.

### Task 8: Package, report, and update the dashboard

**Files:**
- Create: immutable container receipt under the experiment output.
- Modify: roadmap DSpark HTML/status artifacts after results are fetched.

**Interfaces:**
- Consumes: verified stage results.
- Produces: recommended runtime commit, immutable image, comparison table, and reproducible handoff.

- [ ] **Step 1: Save immutable image and receipt**

Record image SHA-256, source commit, PR manifest, package versions, and build
command.

- [ ] **Step 2: Validate result completeness**

Require expected token counts, matched configurations, complete metrics, and
valid baseline pairing.

- [ ] **Step 3: Update HTML and Markdown reports**

Separate completed, failed, excluded, and still-running stages. Include
conflicts and failed hypotheses.

- [ ] **Step 4: Run report builders and tests**

Run:

```bash
python3 scripts/build_latest_specdec_html_pages.py
python3 scripts/build_pages_index.py
python3 -m py_compile \
  scripts/build_latest_specdec_html_pages.py \
  scripts/build_pages_index.py
```

- [ ] **Step 5: Commit and push the final recommendation**

Commit only the files owned by this experiment and preserve unrelated user
changes.

