# Adaptive MXFP8 dense kernels for NeMo-RL

This branch adds qualified, exact-shape TRTLLM tactic selection for dense MXFP8
linear layers in vLLM 0.20.2. Runtime selection consumes one immutable JSON
manifest. Offline tools create the manifest; the runtime never edits it.

The NeMo-RL target is:

```text
examples/configs/recipes/llm/performance/
grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
```

This is the MXFP8 rollout recipe, not the BF16
`grpo-qwen3-30ba3b-4n4g.yaml` recipe.

## Immutable source and binary contract

Use these source coordinates:

```text
fork:                          https://github.com/seonjinn/vllm.git
branch:                        sna/mxfp8-adaptive-v0.20.2-nemorl
source-install runtime pin:    bc5881924556fcf830f8158815d5a62cef0fbcba
wheel builder introduced:      9070391094c14af61dad0a3872113b09fe0619eb
approved wheel source pin:     e8bb60a254c7d823b3e1b2d30bdc90e43d0e770d
functional implementation:    e32ce4fdd30ef313e10bf3a328352ead2a4c0054
upstream vLLM base:            5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1
vLLM public version:           0.20.2
FlashInfer public version:     0.6.8.post1
```

The functional implementation commit is a historical lower bound, not a final
build pin. It does not include later offline-bootstrap, NeMo-RL handoff, wheel
builder, or review fixes. The source-install runtime pin predates the wheel
builder and is only for the existing source-install workflow. Fetch the
published branch to make the approved object available, then verify its
ancestry and build that immutable source-install runtime commit on every node.
Later documentation-only branch commits do not change this pin:

```bash
git fetch origin sna/mxfp8-adaptive-v0.20.2-nemorl
git checkout --detach bc5881924556fcf830f8158815d5a62cef0fbcba
VLLM_BUILD_COMMIT=$(git rev-parse HEAD)
git merge-base --is-ancestor \
  e32ce4fdd30ef313e10bf3a328352ead2a4c0054 \
  "$VLLM_BUILD_COMMIT"
printf 'VLLM_BUILD_COMMIT=%s\n' "$VLLM_BUILD_COMMIT"
```

Do not build from the branch name or the historical functional commit alone.
The experiment record and container must name `VLLM_BUILD_COMMIT`. Do not use
this source-install pin for the team-distributable wheel: the builder first
appears at `9070391094c14af61dad0a3872113b09fe0619eb`, and its hardened approved
wheel source pin is documented separately below.

The direct TRTLLM runner and tactic enumeration are FlashInfer private APIs.
Changing FlashInfer can change the API, tactic IDs, or both. Requalify instead
of reusing a manifest with another FlashInfer public version.

Source installation over a precompiled vLLM binary requires:

```bash
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION=/absolute/path/to/vllm-0.20.2-abi-matching.whl
export VLLM_VERSION_OVERRIDE=0.20.2
```

`VLLM_PRECOMPILED_WHEEL_LOCATION` must identify a vLLM 0.20 wheel built for the
same Python, PyTorch, CUDA, architecture, and native-extension ABI as the
runtime container. `VLLM_VERSION_OVERRIDE=0.20.2` is required because the
manifest loader rejects another public vLLM version.

### Team-distributable wheel

Use `tools/mxfp8/build_custom_wheel.py` when teammates need one installable
artifact instead of an editable source checkout. The builder deliberately
repackages the exact official vLLM wheel instead of invoking
`setup.py bdist_wheel`.

The standard precompiled setup path is designed for Python-only source and
editable installs. It extracts a selected list of native artifacts into a
source tree; a local wheel build can also emit a host `linux_aarch64` tag
unless its platform tag is explicitly rewritten. Neither behavior proves that
the result retained the complete official native payload and
`manylinux_2_35_aarch64` compatibility. The custom builder instead:

- requires a clean, exact Git `HEAD` on Linux `aarch64`;
- requires the base wheel and output directory to be outside the source
  checkout;
- verifies that the source commit descends from upstream base
  `5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1`;
- requires vLLM's official precompiled-build environment variables;
- verifies the official base wheel's filename, metadata, WHEEL tag, RECORD,
  and fixed SHA256;
- preserves every official payload member, including every native extension,
  except tracked runtime files intentionally overlaid or deleted by the exact
  custom source commit; base metadata changes are limited to the WHEEL build
  header and regenerated RECORD;
- reads tracked runtime files from NUL-delimited Git tree records, retains
  regular-file modes, and rejects symlinks, gitlinks, other non-regular
  entries, non-UTF-8 paths, and paths containing control characters;
- overlays only tracked `vllm` Python and declared runtime package data, not
  `tools`, tests, benchmarks, or other developer files, and removes
  renamed-away or deleted runtime paths;
- includes every tracked
  `vllm/model_executor/kernels/linear/mxfp8/tactic_configs/*.json`;
- keeps package metadata at exactly `Version: 0.20.2` for NeMo's strict
  version gate;
- adds a wheel build tag, embeds the full custom commit and base-wheel digest
  in package and dist-info provenance JSON, and regenerates RECORD;
- derives and validates the exact completed member transformation before
  publication, including every unchanged base member's bytes, file mode, and
  ZIP compression method, and rejects missing, added, or modified payload;
- revalidates exact `HEAD`, index, worktree, and untracked-file cleanliness
  immediately before create-only publication;
- emits deterministic adjacent metadata and SHA256 files without overwriting
  existing artifacts and fsyncs temporary files and publication directories.

The only accepted base artifact is:

```text
filename: vllm-0.20.2-cp38-abi3-manylinux_2_35_aarch64.whl
sha256:   76ccf4c0554556c06f6b0fb1643742d4cf97dcc69f6ef3f04556d0764126035a
```

The official release asset uses
`manylinux_2_35_aarch64` in its filename and
`Tag: cp38-abi3-linux_aarch64` in its internal WHEEL metadata. The builder
validates and preserves both existing identities exactly; it does not silently
rewrite one to resemble the other.

Build on the Linux `aarch64` host that will supply the wheel to the NeMo
container. Use a directory outside the Git checkout for inputs and outputs so
that the clean-tree gate remains meaningful:

```bash
set -euo pipefail

VLLM_ROOT=/workspace/vllm
WHEEL_WORK=/shared/vllm-mxfp8-wheel
VLLM_BASE_WHEEL="$WHEEL_WORK/vllm-0.20.2-cp38-abi3-manylinux_2_35_aarch64.whl"
WHEEL_OUT="$WHEEL_WORK/custom"
VLLM_WHEEL_COMMIT=e8bb60a254c7d823b3e1b2d30bdc90e43d0e770d

test "$(uname -s)" = Linux
test "$(uname -m)" = aarch64
git -C "$VLLM_ROOT" fetch origin sna/mxfp8-adaptive-v0.20.2-nemorl
git -C "$VLLM_ROOT" checkout --detach "$VLLM_WHEEL_COMMIT"
test "$(git -C "$VLLM_ROOT" status --porcelain --untracked-files=all)" = ""
test "$(git -C "$VLLM_ROOT" rev-parse HEAD)" = "$VLLM_WHEEL_COMMIT"
test "$(git -C "$VLLM_ROOT" rev-parse "$VLLM_WHEEL_COMMIT^{commit}")" = \
  "$VLLM_WHEEL_COMMIT"
git -C "$VLLM_ROOT" merge-base --is-ancestor \
  9070391094c14af61dad0a3872113b09fe0619eb \
  "$VLLM_WHEEL_COMMIT"
git -C "$VLLM_ROOT" merge-base --is-ancestor \
  5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1 \
  "$VLLM_WHEEL_COMMIT"

mkdir -p "$WHEEL_WORK"
curl --fail --location \
  --output "$VLLM_BASE_WHEEL" \
  https://github.com/vllm-project/vllm/releases/download/v0.20.2/vllm-0.20.2-cp38-abi3-manylinux_2_35_aarch64.whl
printf '%s  %s\n' \
  76ccf4c0554556c06f6b0fb1643742d4cf97dcc69f6ef3f04556d0764126035a \
  "$VLLM_BASE_WHEEL" |
  sha256sum --check

export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_BASE_WHEEL"
uv run --no-project python "$VLLM_ROOT/tools/mxfp8/build_custom_wheel.py" \
  --repo-root "$VLLM_ROOT" \
  --source-commit "$VLLM_WHEEL_COMMIT" \
  --output-dir "$WHEEL_OUT"
```

The wheel filename is:

```text
vllm-0.20.2-1mxfp8g<SHA12>-cp38-abi3-manylinux_2_35_aarch64.whl
```

The build tag distinguishes the artifact without changing the installed
version. The complete 40-character commit remains authoritative in both:

```text
vllm/mxfp8_wheel_provenance.json
vllm-0.20.2.dist-info/mxfp8-provenance.json
```

Verify and install exactly that wheel:

```bash
set -euo pipefail

CUSTOM_WHEEL=$(
  find "$WHEEL_OUT" -maxdepth 1 -type f \
    -name 'vllm-0.20.2-1mxfp8g*-cp38-abi3-manylinux_2_35_aarch64.whl' \
    -print -quit
)
test -n "$CUSTOM_WHEEL"
test -f "${CUSTOM_WHEEL}.metadata.json"
test -f "${CUSTOM_WHEEL}.sha256"
(
  cd "$WHEEL_OUT"
  sha256sum --check "$(basename "$CUSTOM_WHEEL").sha256"
)
unzip -p "$CUSTOM_WHEEL" vllm/mxfp8_wheel_provenance.json
unzip -p "$CUSTOM_WHEEL" vllm-0.20.2.dist-info/METADATA |
  grep -Fx 'Version: 0.20.2'
uv pip install --reinstall "$CUSTOM_WHEEL" --torch-backend=auto
```

Copy the wheel together with its `.metadata.json` and `.sha256` files. A
teammate installs only the wheel; the adjacent files are the review and
transfer-integrity record. Record the full embedded `source_commit`, not the
12-character filename abbreviation, in every experiment.

The builder publishes the three adjacent files with create-only filesystem
links and rolls back the files it linked if an ordinary publication error
occurs. The three directory entries are not one crash-atomic transaction:
process or host failure can leave a partial set. After any interruption, do not
install from that directory until all three exact names exist and the SHA256
check succeeds. Treat a partial set as invalid; remove only those exact partial
artifact names or select a fresh output directory, then rerun the builder.

Record the following in every experiment:

- custom source commit;
- precompiled wheel SHA256;
- container SHA256;
- FlashInfer public version;
- model identifier and tensor-parallel size;
- GPU family and compute capability;
- manifest bytes and SHA256.

## Runtime contract

The adaptive path is enabled by one variable:

```bash
export VLLM_MXFP8_DENSE_CONFIG_FILE=/absolute/shared/path/qwen3_qualified.json
```

An absolute path must exist at the same location in every internal Ray worker.
A relative value resolves only below vLLM's package-owned directory:

```text
vllm/model_executor/kernels/linear/mxfp8/tactic_configs/
```

For example:

```bash
export VLLM_MXFP8_DENSE_CONFIG_FILE=qwen3_30ba3b_tp1_v0202_qualified.json
```

vLLM 0.20 natively forwards all `VLLM_*` variables to its internal Ray
workers. Do not add this key to `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`. Explicitly
listed non-`VLLM_*` variables remain additive through that setting.

File configuration is mutually exclusive with legacy inline tactic variables.
The file supplies the complete adaptive layout, backend, quantization, padding,
default-tactic, and exact-shape tactic policy.

The runtime fails closed when:

- JSON is malformed, contains duplicate keys, or has unsupported fields;
- the file changes after preparation;
- model or tensor-parallel size differs;
- vLLM or FlashInfer public version differs;
- GPU compute capability differs;
- direct TRTLLM or required 8x4 quantization policy is disabled;
- a shape or tactic has an invalid type or duplicate identity;
- required provenance is absent or invalid;
- file configuration is mixed with legacy inline tactic hints;
- configuration is first loaded during CUDA Graph capture.

A tactic miss is not an error. It uses runner default `-1`.

## Manifest schema

The loader accepts schema version 1 with these top-level objects:

```json
{
  "schema_version": 1,
  "mode": "adaptive",
  "compatibility": {},
  "policy": {},
  "tactics": {
    "8x4": [],
    "128x4": []
  },
  "provenance": {}
}
```

Qualified manifests fix this policy:

```text
gemm_backend=trtllm
layout=adaptive
switch_m=256
direct_trtllm=true
require_direct_trtllm=true
quant_backend=cuda
require_8x4_quant=true
pad_to_128=false
default_tactic=-1
```

Tactic keys are exact physical `(M,N,K)` shapes and retain their traced
`8x4` or `128x4` layout. Low-M 8x4 shapes are never padded to M=128 for
qualification.

Tactic IDs are specific to at least:

- model identifier;
- tensor-parallel size;
- vLLM public version and base implementation;
- FlashInfer public version and private runner ABI;
- compute capability and GPU family;
- quantization/layout policy;
- qualified container.

Do not reuse an Ultra TP4 table for Qwen TP1, or a standalone-serving seed for
a NeMo-RL rollout, without a new trace and qualification.

## MXFP8 A/B

Use the exact NeMo-RL MXFP8 recipe for both arms.

Original-selection baseline:

```bash
unset VLLM_MXFP8_DENSE_CONFIG_FILE
```

Also keep legacy adaptive layout and inline tactic overrides unset. With no
file or legacy overrides, this fork preserves original vLLM MXFP8 selection.

Qualified adaptive arm:

```bash
export VLLM_MXFP8_DENSE_CONFIG_FILE=/shared/qwen3_30ba3b_tp1_v0202_qualified.json
```

Keep every other NeMo-RL config, container, checkpoint, prompt set, seed, node
layout, and measurement window identical. Compare rollout latency/throughput
and end-to-end step time, and verify the adaptive trace reports qualified
tactic hits.

## Offline qualification

The workflow has five explicit stages:

```text
trace bootstrap -> inventory -> shmoo -> promote -> validate
```

### 0. Create the trace bootstrap

The only supported empty-manifest generator is the fixed
`trace-bootstrap-qwen3-30ba3b-tp1` subcommand. It fixes runner default `-1`,
empty tactic tables, the adaptive policy, base commit, and this identity:

```text
model=Qwen/Qwen3-30B-A3B
tensor_parallel_size=1
vllm_version=0.20.2
flashinfer_version=0.6.8.post1
compute_capability=10.0
gpu_family=GB200
qualification_scope=nemo_rl_qwen3_30ba3b_mxfp8_rollout_trace_bootstrap
```

First materialize three named immutable inputs. `RECIPE_SNAPSHOT` is the exact
Git blob of the target NeMo-RL recipe at the recorded NeMo-RL commit.
`EMPTY_HINTS` is intentionally zero bytes because no tactic is qualified
before tracing. `CONTAINER_IMAGE` is the exact image later used by the rollout
and shmoo:

```bash
set -euo pipefail

NEMO_RL_ROOT=/workspace/NeMo-RL
NEMO_RL_COMMIT=$(git -C "$NEMO_RL_ROOT" rev-parse HEAD)
RECIPE_PATH=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
PROVENANCE_DIR=/shared/qwen3_tp1_bootstrap_inputs
RECIPE_SNAPSHOT="$PROVENANCE_DIR/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml"
EMPTY_HINTS="$PROVENANCE_DIR/qwen3_30ba3b_tp1_trace_bootstrap.empty_hints"
CONTAINER_IMAGE=/shared/containers/nemo-rl-qwen3-30ba3b-mxfp8-rollout.sqsh

mkdir -p "$PROVENANCE_DIR"
git -C "$NEMO_RL_ROOT" show \
  "$NEMO_RL_COMMIT:$RECIPE_PATH" >"$RECIPE_SNAPSHOT"
: >"$EMPTY_HINTS"
chmod 0444 "$RECIPE_SNAPSHOT" "$EMPTY_HINTS"

SOURCE_MANIFEST_SHA256=$(sha256sum -- "$RECIPE_SNAPSHOT" | cut -d ' ' -f1)
SOURCE_HINT_SHA256=$(sha256sum -- "$EMPTY_HINTS" | cut -d ' ' -f1)
CONTAINER_SHA256=$(sha256sum -- "$CONTAINER_IMAGE" | cut -d ' ' -f1)
test "$SOURCE_HINT_SHA256" = \
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
printf 'NEMO_RL_COMMIT=%s\n' "$NEMO_RL_COMMIT"
printf '%s  %s\n' "$SOURCE_MANIFEST_SHA256" "$RECIPE_SNAPSHOT"
printf '%s  %s\n' "$SOURCE_HINT_SHA256" "$EMPTY_HINTS"
printf '%s  %s\n' "$CONTAINER_SHA256" "$CONTAINER_IMAGE"
```

Preserve those three files with the experiment. Generate the bootstrap:

```bash
BOOTSTRAP_MANIFEST=/shared/qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json
BOOTSTRAP_SHA256=$(
  python tools/mxfp8/offline_shmoo.py \
    trace-bootstrap-qwen3-30ba3b-tp1 \
    --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
    --source-hint-sha256 "$SOURCE_HINT_SHA256" \
    --container-sha256 "$CONTAINER_SHA256" \
    --output "$BOOTSTRAP_MANIFEST"
)
printf '%s  %s\n' "$BOOTSTRAP_SHA256" "$BOOTSTRAP_MANIFEST" \
  >"$BOOTSTRAP_MANIFEST.sha256"
sha256sum --check "$BOOTSTRAP_MANIFEST.sha256"
```

The subcommand accepts no model, topology, version, policy, or scope override.
It validates canonical bytes through the production loader before atomically
publishing a new output and refuses to overwrite any existing path. This
explicit trace-bootstrap scope is not a qualified performance manifest.

Enable tracing with that bootstrap manifest:

```bash
export VLLM_MXFP8_DENSE_CONFIG_FILE="$BOOTSTRAP_MANIFEST"
export VLLM_MXFP8_DENSE_SHAPE_TRACE=1
export VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR=/shared/qwen_trace
```

The bootstrap file's raw SHA256 is recorded in every eligible trace row.
Inventory rejects a different hash. Both adaptive-dispatch and dense-shape
events carry an explicit physical layout.

### 1. Inventory

Every process writes its PID and hostname into
`adaptive_dispatch_${hostname}_${pid}.jsonl` or
`dense_shapes_${hostname}_${pid}.jsonl`. On 4n4g, collect every file from
every process. This Bash array uses null delimiters and deterministic sorting;
the quoted `find` patterns are interpreted by `find`, not left for shell glob
expansion:

```bash
set -euo pipefail

mapfile -d '' -t TRACE_FILES < <(
  find /shared/qwen_trace -maxdepth 1 -type f \
    \( -name 'adaptive_dispatch_*_*.jsonl' \
       -o -name 'dense_shapes_*_*.jsonl' \) \
    -print0 |
    sort -z
)
if (( ${#TRACE_FILES[@]} == 0 )); then
  printf '%s\n' \
    'not-applicable: no dense MXFP8 trace files were emitted' >&2
  exit 3
fi

TRACE_ARGS=()
for trace_file in "${TRACE_FILES[@]}"; do
  TRACE_ARGS+=(--trace "$trace_file")
done

INVENTORY_ERROR=/shared/qwen3_tp1_inventory.stderr
set +e
python tools/mxfp8/offline_shmoo.py inventory \
  --bootstrap-manifest "$BOOTSTRAP_MANIFEST" \
  "${TRACE_ARGS[@]}" \
  --output /shared/qwen3_tp1_inventory.json \
  2>"$INVENTORY_ERROR"
inventory_status=$?
set -e
if (( inventory_status != 0 )); then
  cat "$INVENTORY_ERROR" >&2
  if grep -Fq 'zero eligible dense MXFP8 trace records' \
      "$INVENTORY_ERROR"; then
    printf '%s\n' \
      'not-applicable: trace files contain zero eligible records' >&2
    exit 3
  fi
  exit "$inventory_status"
fi
```

Inputs are content-hashed deterministically. Paths and CLI order do not affect
the aggregate provenance digest.

The current runtime trace is unique-shape oriented. A missing frequency
defaults to 1 and does not claim true dynamic call frequency.

### 2. GPU shmoo

Run on the same GB200 container and topology used for rollout:

```bash
python tools/mxfp8/offline_shmoo.py shmoo \
  --inventory /shared/qwen3_tp1_inventory.json \
  --output /shared/qwen3_tp1_observations.jsonl \
  --repeat-count 3 \
  --base-seed 1234 \
  --warmup 10 \
  --iterations 80 \
  --workspace-mb 256 \
  --minimum-cosine-similarity 0.999 \
  --vllm-version 0.20.2 \
  --flashinfer-version 0.6.8.post1 \
  --container-sha256 "$CONTAINER_SHA256"
```

The tool benchmarks runner default `-1` and every valid tactic, compares
against BF16, uses CUDA-event timing, appends JSONL, and resumes only exact
shape/tactic/repeat identities with the deterministic seed plan.

`get_valid_tactics` failures abort with the private-runner diagnostic. They are
never represented as a successful empty enumeration.

### 3. Promote

```bash
python tools/mxfp8/offline_shmoo.py promote \
  --inventory /shared/qwen3_tp1_inventory.json \
  --observations /shared/qwen3_tp1_observations.jsonl \
  --output /shared/qwen3_30ba3b_tp1_v0202_qualified.json \
  --qualification-output /shared/qwen3_tp1_qualification.json \
  --repeat-count 3 \
  --minimum-cosine-similarity 0.999 \
  --minimum-speedup-vs-default 1.02 \
  --qualification-scope nemo_rl_qwen3_30ba3b_mxfp8_rollout \
  --vllm-version 0.20.2 \
  --vllm-base-commit 5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1 \
  --flashinfer-version 0.6.8.post1 \
  --compute-capability 10.0 \
  --gpu-family GB200 \
  --model Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 1
```

Every required default repeat must pass. A candidate is promoted only when all
required repeats are finite and correct and the median speedup versus default
is at least 1.02. Exact timing ties select the lower tactic ID. Unpromoted
shapes remain absent and use `-1`.

### 4. Validate and reproduce bytes

```bash
python tools/mxfp8/offline_shmoo.py validate \
  --manifest /shared/qwen3_30ba3b_tp1_v0202_qualified.json \
  --inventory /shared/qwen3_tp1_inventory.json \
  --observations /shared/qwen3_tp1_observations.jsonl \
  --repeat-count 3 \
  --minimum-cosine-similarity 0.999 \
  --minimum-speedup-vs-default 1.02 \
  --qualification-scope nemo_rl_qwen3_30ba3b_mxfp8_rollout \
  --vllm-version 0.20.2 \
  --flashinfer-version 0.6.8.post1 \
  --compute-capability 10.0 \
  --model Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 1 \
  --check
```

`--check` regenerates the manifest through the same qualification functions
and byte-compares it to the target. Canonical manual tactic edits fail.

Generated outputs refuse to overwrite existing files or alias raw inputs.
Shmoo JSONL alone supports append/resume.

## Zero-hit gate

Qwen3-30B-A3B is MoE, and ignored dense projections or fused MoE execution may
bypass this dense MXFP8 path. No matching trace files and inventory's
`zero eligible dense MXFP8 trace records` error are both
`not-applicable`, not successful empty qualification.

Treat zero hits as a negative applicability result:

- do not generate or label an empty table as optimized;
- do not claim rollout improvement from this kernel path;
- inspect fused MoE and ignored-module routing before expanding scope;
- preserve the original-selection baseline.
