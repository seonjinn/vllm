# vLLM v0.20.2 MXFP8 Adaptive JSON Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean custom vLLM 0.20.2 fork branch that contains the
validated adaptive dense MXFP8 implementation, loads qualified exact-shape
tactics from one fail-closed JSON manifest, and reproducibly generates that
manifest from offline shmoo results.

**Architecture:** Port the three validated Python override files onto the clean
`nemo-speed-v0.20.2` base, then isolate JSON parsing and compatibility checks
in a small typed module. The existing adaptive runtime consumes the validated
immutable object during pre-CUDA-Graph preparation, while no config file
preserves the original vLLM MXFP8 path. A separate offline CLI converts runtime
shape traces into a deterministic inventory, benchmarks every eligible
physical shape against runner default `-1`, applies correctness and repeat
gates, and emits the immutable runtime JSON. Runtime code never modifies the
manifest.

**Tech Stack:** Python 3.12, vLLM 0.20.2 fork, PyTorch 2.11/cu130, FlashInfer 0.6.8.post1, pytest, JSON, CUDA Graph.

## Global Constraints

- Base commit is `5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1`.
- Branch is `sna/mxfp8-adaptive-v0.20.2-nemorl`.
- Canonical adaptive source is benchmark commit `4bb11d11b2fdef33cd84b5430d4403428c07a2e1`.
- Only the three validated Python override files are ported; dirty native CUDA experiments are excluded.
- FlashInfer private TRTLLM API compatibility is fixed to `0.6.8.post1`.
- Custom source builds set `VLLM_VERSION_OVERRIDE=0.20.2`; the loader rejects
  another installed version.
- Adaptive layout uses 8x4 for logical `M <= 256` and 128x4 above 256.
- Shape misses use runner tactic `-1`.
- The primary runtime input is `VLLM_MXFP8_DENSE_CONFIG_FILE`.
- File configuration and legacy inline tactic variables are mutually exclusive.
- Relative config names resolve only under the package-owned `tactic_configs` directory.
- Invalid schema, compatibility mismatch, duplicate shape, changed file, or post-prepare configuration change fails closed.
- All new functions and methods have type hints.
- Production code is written only after its failing test has been observed.
- Python commands use `uv`; never use system Python or bare `pip`.
- Commits are signed off.

---

### Task 1: Port the validated adaptive dense MXFP8 runtime

**Files:**
- Modify: `vllm/model_executor/kernels/linear/mxfp8/flashinfer.py`
- Modify: `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`
- Modify: `vllm/utils/flashinfer.py`
- Create: `tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py`
- Create: `tests/kernels/quantization/test_mxfp8_trtllm_weight_shuffle_contract.py`

**Interfaces:**
- Consumes: clean vLLM base `5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1`.
- Produces: compiler-safe adaptive 8x4/128x4 dense MXFP8 execution with exact-shape tactic maps and pre-capture runner state.

- [ ] **Step 1: Add the source-contract and behavioral tests**

Port the tests from benchmark commit
`4bb11d11b2fdef33cd84b5430d4403428c07a2e1`:

```text
experiments/sweep/test_mxfp8_adaptive_layout_v020.py
experiments/sweep/test_mxfp8_trtllm_weight_shuffle_contract.py
```

Change only their path constants so they inspect the vLLM source tree in this
repository. Preserve all layout threshold, tactic parsing, configuration
freeze, workspace separation, weight shuffle, graph specialization, and
fail-closed assertions.

- [ ] **Step 2: Run the tests and verify the clean base fails**

Run:

```bash
uv run --no-project --with pytest python -m pytest \
  tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py \
  tests/kernels/quantization/test_mxfp8_trtllm_weight_shuffle_contract.py -q
```

Expected: failures identify the missing adaptive layout policy, separate
8x4/128x4 runners, exact-shape maps, and corrected B/B-scale shuffle.

- [ ] **Step 3: Port the validated three-file implementation**

Use the tracked contents under:

```text
runtime_overrides/mxfp8_trtllm_adaptive_layout_v020/vllm/
```

at benchmark commit `4bb11d11b2fdef33cd84b5430d4403428c07a2e1`.
Before any later JSON changes, verify the three target SHA256 values are:

```text
b1de017ff41c3714712a56a7575b0f8fdbda9a05ce33b100828a4b76ed1bbd9a
476defbfc9943138b06fdf920c2120ea8ebce3e327f2e42bbeb04f4c992c4015
50e4b527876303c2e3b830745a6b6c1c712b1c6a54752c8509f2417adadedfdb
```

- [ ] **Step 4: Run the focused tests**

Run the Step 2 command.

Expected: all ported tests pass.

- [ ] **Step 5: Check formatting and commit**

Run:

```bash
uvx ruff check \
  vllm/model_executor/kernels/linear/mxfp8/flashinfer.py \
  vllm/model_executor/layers/quantization/utils/mxfp8_utils.py \
  vllm/utils/flashinfer.py \
  tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py \
  tests/kernels/quantization/test_mxfp8_trtllm_weight_shuffle_contract.py
git diff --check
git add \
  vllm/model_executor/kernels/linear/mxfp8/flashinfer.py \
  vllm/model_executor/layers/quantization/utils/mxfp8_utils.py \
  vllm/utils/flashinfer.py \
  tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py \
  tests/kernels/quantization/test_mxfp8_trtllm_weight_shuffle_contract.py
git commit -s -m "perf(mxfp8): port adaptive TRTLLM dense path"
```

### Task 2: Add a typed fail-closed JSON tactic configuration loader

**Files:**
- Create: `vllm/model_executor/kernels/linear/mxfp8/tactic_config.py`
- Create: `vllm/model_executor/kernels/linear/mxfp8/tactic_configs/.gitkeep`
- Create: `tests/kernels/quantization/test_mxfp8_tactic_config.py`

**Interfaces:**
- Produces:

```python
Shape = tuple[int, int, int]

@dataclass(frozen=True)
class Mxfp8DenseRuntimeConfig:
    source_path: Path
    source_sha256: str
    mode: Literal["adaptive"]
    switch_m: int
    gemm_backend: Literal["trtllm"]
    layout: Literal["adaptive"]
    direct_trtllm: bool
    require_direct_trtllm: bool
    quant_backend: Literal["cuda", "flashinfer"]
    require_8x4_quant: bool
    pad_to_128: bool
    default_tactic: int
    tactics_8x4: tuple[tuple[Shape, int], ...]
    tactics_128x4: tuple[tuple[Shape, int], ...]
    compatibility: Mapping[str, object]
    provenance: Mapping[str, object]

def load_mxfp8_dense_runtime_config(
    reference: str,
    *,
    actual_vllm_version: str,
    actual_flashinfer_version: str,
    actual_compute_capability: tuple[int, int],
    package_config_dir: Path | None = None,
) -> Mxfp8DenseRuntimeConfig:
    ...
```

- [ ] **Step 1: Write tests for valid absolute and package-relative files**

Use literal temporary JSON fixtures. Assert that:

```python
config.switch_m == 256
config.tactics_8x4 == (((1, 2048, 8192), 66),)
config.tactics_128x4 == (((1000, 2048, 8192), 70),)
config.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
```

Also assert that a relative reference resolves beneath the injected
`package_config_dir`, not the process current directory.

- [ ] **Step 2: Run the loader tests and verify import failure**

Run:

```bash
uv run --no-project --with pytest python -m pytest \
  tests/kernels/quantization/test_mxfp8_tactic_config.py -q
```

Expected: collection fails because `tactic_config.py` does not exist.

- [ ] **Step 3: Add validation tests**

Add separate tests for:

```text
unsupported schema_version
mode other than adaptive
vLLM base-version mismatch
FlashInfer base-version mismatch
compute-capability mismatch
switch_m <= 0
switch_m not divisible by 128
zero or negative m/n/k
non-integer shape or tactic
duplicate shape within one layout
same shape in both layouts
missing required policy/provenance fields
relative path traversal
```

Every failure must match a field-specific `ValueError` or `RuntimeError`.

- [ ] **Step 4: Implement the immutable loader**

Implement the exact interface above using stdlib `json`, `hashlib`, `Path`,
`dataclasses`, and `packaging.version.Version`. Resolve a relative name with:

```python
candidate = (config_dir / reference).resolve()
if not candidate.is_relative_to(config_dir.resolve()):
    raise ValueError("relative MXFP8 config must stay inside tactic_configs")
```

Sort tactic tuples by shape before constructing the frozen dataclass. Retain
the raw compatibility and provenance mappings as immutable mapping proxies.

- [ ] **Step 5: Run the focused tests**

Run the Step 2 command.

Expected: all loader tests pass.

- [ ] **Step 6: Lint and commit**

Run:

```bash
uvx ruff check \
  vllm/model_executor/kernels/linear/mxfp8/tactic_config.py \
  tests/kernels/quantization/test_mxfp8_tactic_config.py
git diff --check
git add \
  vllm/model_executor/kernels/linear/mxfp8/tactic_config.py \
  vllm/model_executor/kernels/linear/mxfp8/tactic_configs/.gitkeep \
  tests/kernels/quantization/test_mxfp8_tactic_config.py
git commit -s -m "feat(mxfp8): load exact tactics from JSON"
```

### Task 3: Generate the versioned TP4 standalone seed manifest

**Files:**
- Create: `tools/mxfp8/build_tactic_config.py`
- Create: `tests/kernels/quantization/test_build_mxfp8_tactic_config.py`
- Create: `vllm/model_executor/kernels/linear/mxfp8/tactic_configs/nemotron3_ultra_tp4_v0202_standalone_seed.json`

**Interfaces:**
- Consumes legacy `M,N,K:tactic;...` hints and explicit provenance arguments.
- Produces the schema-1 JSON consumed by `load_mxfp8_dense_runtime_config`.
- TP4 seed policy: entries with `M <= 256` go to `8x4`; `128x4` is empty
  because high-M tactics are not NeMo-RL rollout-qualified.

- [ ] **Step 1: Write parser and split-policy tests**

Use this literal input:

```text
1,2048,8192:66;256,8192,2048:71;1000,2048,8192:70
```

Assert that `M <= 256` yields two sorted 8x4 entries and that the high-M
entry is omitted when `--high-m-policy empty` is selected. Assert duplicate,
malformed, and non-positive shapes fail.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
uv run --no-project --with pytest python -m pytest \
  tests/kernels/quantization/test_build_mxfp8_tactic_config.py -q
```

Expected: collection fails because the builder module is absent.

- [ ] **Step 3: Implement the builder**

Provide:

```python
def parse_legacy_hints(raw_hints: str) -> tuple[tuple[Shape, int], ...]:
    ...

def build_manifest(
    entries: tuple[tuple[Shape, int], ...],
    *,
    switch_m: int,
    high_m_policy: Literal["empty", "include"],
    compatibility: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    ...
```

The CLI writes stable two-space-indented, key-sorted JSON ending in one
newline.

- [ ] **Step 4: Run the builder tests**

Run the Step 2 command.

Expected: all builder tests pass.

- [ ] **Step 5: Materialize the real seed manifest**

Read the hint file from benchmark commit
`4bb11d11b2fdef33cd84b5430d4403428c07a2e1`:

```text
experiments/sweep/data/microbench/mxfp8_topology_corrected_tactics_20260713/tp4/v020/all_dense_hints.txt
```

Use:

```text
schema_version=1
mode=adaptive
vllm_version=0.20.2
vllm_base_commit=5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1
flashinfer_version=0.6.8.post1
compute_capability=10.0
gpu_family=GB200
model=Nemotron 3 Ultra MXFP8
tensor_parallel_size=4
switch_m=256
high_m_policy=empty
source_manifest_sha256=ba4e5caf540911a20bb0024fb26578103268e551ab70d8b366203ef2da81d67b
source_hint_sha256=dcc927901d6559d31d8f9374932563ccda39614add7c428f85e9401398ab9193
container_sha256=9942dd61b7805b7fd225b37a92d5948344ff1782995b0716d0c052f10dd18ee0
qualification_repeat_count=3
minimum_cosine_similarity=0.999
minimum_speedup_vs_default=1.02
qualification_scope=standalone_serving_seed
```

Verify the output has 63 8x4 entries and zero 128x4 entries.

- [ ] **Step 6: Validate through the production loader and commit**

Run:

```bash
uv run --no-project --with pytest python -m pytest \
  tests/kernels/quantization/test_build_mxfp8_tactic_config.py \
  tests/kernels/quantization/test_mxfp8_tactic_config.py -q
uvx ruff check \
  tools/mxfp8/build_tactic_config.py \
  tests/kernels/quantization/test_build_mxfp8_tactic_config.py
git diff --check
git add \
  tools/mxfp8/build_tactic_config.py \
  tests/kernels/quantization/test_build_mxfp8_tactic_config.py \
  vllm/model_executor/kernels/linear/mxfp8/tactic_configs/nemotron3_ultra_tp4_v0202_standalone_seed.json
git commit -s -m "chore(mxfp8): add qualified TP4 seed tactics"
```

### Task 4: Connect the JSON object to adaptive pre-capture state

**Files:**
- Modify: `vllm/utils/flashinfer.py`
- Modify: `tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py`
- Modify: `tests/kernels/quantization/test_mxfp8_tactic_config.py`

**Interfaces:**
- Consumes: `load_mxfp8_dense_runtime_config`.
- Produces: one-file adaptive runtime selection, frozen file hash, startup
  provenance logging, and legacy-inline compatibility.

- [ ] **Step 1: Add failing integration tests**

Add tests that verify:

```text
VLLM_MXFP8_DENSE_CONFIG_FILE alone supplies the complete adaptive policy
file configuration rejects either legacy inline tactic variable
the fingerprint includes the resolved path and SHA256
changing file bytes after preparation is rejected
the loaded 8x4 and 128x4 entry counts are logged
no config file preserves legacy/original behavior
legacy inline configuration still works when no file is set
```

- [ ] **Step 2: Run and verify the tests fail for missing integration**

Run:

```bash
uv run --no-project --with pytest python -m pytest \
  tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py \
  tests/kernels/quantization/test_mxfp8_tactic_config.py -q
```

Expected: new file-based integration assertions fail while existing adaptive
tests remain green.

- [ ] **Step 3: Load and freeze the JSON configuration**

In `_mxfp8_trtllm_configuration_fingerprint`, when
`VLLM_MXFP8_DENSE_CONFIG_FILE` is non-empty:

```python
runtime_config = load_mxfp8_dense_runtime_config(
    config_reference,
    actual_vllm_version=vllm.__version__,
    actual_flashinfer_version=importlib.metadata.version("flashinfer-python"),
    actual_compute_capability=torch.cuda.get_device_capability(),
)
```

Populate the existing fingerprint fields from the immutable object. Add the
resolved path and file SHA256 to the fingerprint so the existing freeze check
detects mutation. Reject non-empty legacy tactic variables before loading.

- [ ] **Step 4: Emit one startup provenance record**

Log at INFO once per process:

```text
MXFP8 dense config path=<resolved> sha256=<hash> mode=adaptive
switch_m=256 tactics_8x4=<count> tactics_128x4=<count>
qualification_scope=<scope>
```

Include the config SHA256 in every adaptive shape-trace JSONL record.

- [ ] **Step 5: Run the full CPU contract suite**

Run:

```bash
uv run --no-project --with pytest python -m pytest \
  tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py \
  tests/kernels/quantization/test_mxfp8_trtllm_weight_shuffle_contract.py \
  tests/kernels/quantization/test_mxfp8_tactic_config.py \
  tests/kernels/quantization/test_build_mxfp8_tactic_config.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Lint and commit**

Run:

```bash
uvx ruff check \
  vllm/utils/flashinfer.py \
  vllm/model_executor/kernels/linear/mxfp8/tactic_config.py \
  tools/mxfp8/build_tactic_config.py \
  tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py \
  tests/kernels/quantization/test_mxfp8_tactic_config.py \
  tests/kernels/quantization/test_build_mxfp8_tactic_config.py
git diff --check
git add \
  vllm/utils/flashinfer.py \
  tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py \
  tests/kernels/quantization/test_mxfp8_tactic_config.py
git commit -s -m "feat(mxfp8): configure adaptive state from JSON"
```

### Task 5: Add the offline trace-to-shmoo qualification pipeline

**Files:**
- Create: `tools/mxfp8/offline_shmoo.py`
- Create: `tests/kernels/quantization/test_mxfp8_offline_shmoo.py`

**Interfaces:**
- Consumes: adaptive dense shape-trace JSONL and raw per-tactic benchmark JSONL.
- Produces: a deterministic shape inventory, per-shape qualification results,
  and a schema-1 runtime manifest compatible with
  `load_mxfp8_dense_runtime_config`.

The tool has four explicit stages:

```text
inventory: trace JSONL -> unique physical (layout,m,n,k) shapes + frequencies
shmoo:     inventory -> correctness/timing observations for runner -1 and tactics
promote:   repeated observations -> one qualified tactic or default -1 per shape
validate:  generated manifest -> production loader + deterministic regeneration
```

- [ ] **Step 1: Add failing pure tests for trace inventory**

Use literal JSONL fixtures. Assert deterministic aggregation by
`(layout, m, n, k)`, summed call frequency, retained config SHA256, and
rejection of malformed records, mixed config hashes, unsupported layouts,
non-positive dimensions, and an empty eligible inventory.

- [ ] **Step 2: Add failing pure tests for qualification**

Use literal benchmark observations with at least three repeats per candidate.
Assert:

```text
runner default tactic is always -1
only numerically correct observations are eligible
all required repeats must be present
median timing chooses the fastest candidate
promotion requires speedup >= 1.02 versus default -1
ties are resolved by the lower tactic ID
unqualified shapes remain absent from both tactic tables
8x4 and 128x4 shapes cannot cross tables
```

- [ ] **Step 3: Run and verify failure**

Run:

```bash
uv run --no-project --with pytest python -m pytest \
  tests/kernels/quantization/test_mxfp8_offline_shmoo.py -q
```

Expected: collection fails because `tools/mxfp8/offline_shmoo.py` is absent.

- [ ] **Step 4: Implement deterministic inventory and promotion**

Provide typed pure functions:

```python
def load_shape_inventory(paths: Sequence[Path]) -> tuple[ShapeRecord, ...]:
    ...

def qualify_observations(
    inventory: Sequence[ShapeRecord],
    observations: Sequence[BenchmarkObservation],
    *,
    minimum_repeat_count: int,
    minimum_cosine_similarity: float,
    minimum_speedup_vs_default: float,
) -> tuple[QualifiedShape, ...]:
    ...

def build_qualified_manifest(
    qualified_shapes: Sequence[QualifiedShape],
    *,
    compatibility: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    ...
```

Stable JSON output uses sorted keys, two-space indentation, and one trailing
newline. Raw trace and shmoo input SHA256 values are included in provenance.
An inventory with zero eligible dense MXFP8 shapes exits nonzero with a clear
message so a Qwen run that bypasses the dense path cannot silently produce an
empty “optimized” table.

- [ ] **Step 5: Port the validated GPU shmoo runner**

Adapt only the reusable direct-TRTLLM benchmark logic from benchmark commit
`4bb11d11b2fdef33cd84b5430d4403428c07a2e1`:

```text
experiments/sweep/microbench_mxfp8_trtllm_tactic_sweep.py
```

The `shmoo` command:

```text
uses the exact traced physical M/N/K and recorded 8x4 or 128x4 layout
benchmarks runner default -1 and every enumerated valid tactic
uses BF16/reference output for correctness
runs at least three independent repeats
records warmup, iteration count, seed, device, versions, and container digest
never pads a traced 8x4 shape to 128 rows
writes append-only raw JSONL and supports safe resume by exact identity
```

GPU-specific imports are lazy so inventory, promotion, validation, and unit
tests run on CPU-only developer machines.

- [ ] **Step 6: Validate generated manifests through production code**

The `validate` command loads the emitted file through
`load_mxfp8_dense_runtime_config`, verifies the exact model, tensor parallel
size, versions, compute capability, and source hashes, then regenerates the
file in memory and fails if bytes differ. Provide `--check` for CI.

- [ ] **Step 7: Run focused verification and commit**

Run:

```bash
uv run --no-project --with pytest python -m pytest \
  tests/kernels/quantization/test_mxfp8_offline_shmoo.py \
  tests/kernels/quantization/test_mxfp8_tactic_config.py \
  tests/kernels/quantization/test_build_mxfp8_tactic_config.py -q
uvx ruff check \
  tools/mxfp8/offline_shmoo.py \
  tests/kernels/quantization/test_mxfp8_offline_shmoo.py
git diff --check
git add \
  tools/mxfp8/offline_shmoo.py \
  tests/kernels/quantization/test_mxfp8_offline_shmoo.py
git commit -s -m "feat(mxfp8): add offline tactic qualification pipeline"
```

### Task 6: Verify the custom fork handoff

**Files:**
- Create: `docs/mxfp8-adaptive-nemorl.md`

**Interfaces:**
- Produces: reproducible branch/build/runtime instructions for the NeMo-RL
  integration plan.

- [ ] **Step 1: Document the immutable build contract**

Document:

```text
fork URL and exact custom branch/commit
base commit
FlashInfer 0.6.8.post1 private API dependency
VLLM_USE_PRECOMPILED=1
VLLM_PRECOMPILED_WHEEL_LOCATION must be an ABI-matching vLLM 0.20 wheel
VLLM_VERSION_OVERRIDE=0.20.2
VLLM_MXFP8_DENSE_CONFIG_FILE relative and absolute forms
original versus adaptive A/B behavior
JSON schema and fail-closed conditions
TP/model/version specificity of tactic IDs
offline inventory, shmoo, promotion, validation, and zero-hit failure workflow
```

- [ ] **Step 2: Run fresh CPU verification**

Run:

```bash
uv run --no-project --with pytest python -m pytest \
  tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py \
  tests/kernels/quantization/test_mxfp8_trtllm_weight_shuffle_contract.py \
  tests/kernels/quantization/test_mxfp8_tactic_config.py \
  tests/kernels/quantization/test_build_mxfp8_tactic_config.py \
  tests/kernels/quantization/test_mxfp8_offline_shmoo.py -q
uvx ruff check \
  vllm/model_executor/kernels/linear/mxfp8/flashinfer.py \
  vllm/model_executor/layers/quantization/utils/mxfp8_utils.py \
  vllm/model_executor/kernels/linear/mxfp8/tactic_config.py \
  vllm/utils/flashinfer.py \
  tools/mxfp8/build_tactic_config.py \
  tools/mxfp8/offline_shmoo.py \
  tests/kernels/quantization/test_mxfp8_adaptive_layout_v020.py \
  tests/kernels/quantization/test_mxfp8_trtllm_weight_shuffle_contract.py \
  tests/kernels/quantization/test_mxfp8_tactic_config.py \
  tests/kernels/quantization/test_build_mxfp8_tactic_config.py \
  tests/kernels/quantization/test_mxfp8_offline_shmoo.py
git diff --check
```

Expected: pytest and Ruff exit zero with no failures.

- [ ] **Step 3: Commit the handoff document**

```bash
git add docs/mxfp8-adaptive-nemorl.md
git commit -s -m "docs: describe NeMo-RL adaptive MXFP8 handoff"
```

- [ ] **Step 4: Record the exact branch head**

Run:

```bash
git status --short
git rev-parse HEAD
git log --oneline --decorate -5
```

Expected: clean worktree and a named commit ready to be used as the immutable
NeMo-RL custom-vLLM ref.
