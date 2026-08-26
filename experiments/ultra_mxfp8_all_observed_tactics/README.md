# Nemotron 3 Ultra All-Observed MXFP8 Tactic Lookup

This experiment repeats the vLLM 0.20.2 all-observed workflow on vLLM 0.27.1.
It does not assume that serving uses only power-of-two token counts. It records
every exact dense MXFP8 `(M, N, K)` seen by Python tactic selection during the
Ultra capture and baseline workloads, profiles every valid tactic for each
shape under CUDA Graph replay and cold L2, then compares the generated lookup
with the normal vLLM autotuner.

## Contract

- Model: Nemotron 3 Ultra MXFP8
- Hardware: one GB200 node per paired allocation
- Parallelism: TP4, DP1, expert parallel enabled
- Dense backend: CuTeDSL, FlashInfer CUTLASS, or FlashInfer TRTLLM
- MoE backend: FlashInfer TRTLLM
- Workloads: 1K/10K, 10K/1K, and 1K/1K ISL/OSL
- Shape capture concurrency: 1, 2, 4, 8, 16, 32; ten request waves
- Eager shape probe: 1K/64 and 10K/64 ISL/OSL to expose dynamic scheduler
  mixtures without paying eager-mode cost for long decode
- CUDA Graph shape capture: full 1K/10K, 10K/1K, and 1K/1K workloads at
  concurrency 1, 2, 4, 8, 16, and 32
- Final baseline/lookup A/B concurrency: 1, 8, 32; ten request waves
- Paired A/B design: `A` is normal autotuner selection and `B` is the exact
  offline lookup. Each backend runs fresh-server `A -> B` and `B -> A` pairs.
  The two legs of each pair share one node allocation; the reverse-order pair
  may run on another node. Hostname and pair order are recorded.
- Final execution: CUDA Graph capture verified from the vLLM completion marker
- Trace completeness: every capture and lookup phase must contain distributed
  ranks 0-3
- Source contract: clean, detached experiment and FlashInfer worktrees pinned
  by SHA; installed vLLM runtime pinned by the actual container image SHA256;
  model config and weight-index files pinned by SHA256
- Correctness: finite and elementwise oracle checks, followed by deterministic
  generated-text parity within both A/B orders and across all four arms

The short eager capture discovers irregular runtime values such as `M=1001`
without spending hours decoding in eager mode. Ten waves expose
continuous-batching mixtures. A separate CUDA Graph capture contributes the
full long-workload shape set. The offline lookup uses an exact
`(M, N, K, runner)` key; a miss always delegates to the normal FlashInfer
autotuner. Final measurements use two same-node pairs in opposite execution
orders. The report includes each pair ratio, their geometric mean, and
unique-key and selection-call-weighted lookup coverage. CUDA Graph replay does
not re-enter Python tactic selection, so this count is not an
execution-weighted kernel coverage metric.

## Run

Push this branch before submission, then initialize the remote source under
`/home` and launch the dependency chain:

```bash
BACKEND=cute-dsl ./experiments/ultra_mxfp8_all_observed_tactics/submit_pipeline.sh
BACKEND=cutlass ./experiments/ultra_mxfp8_all_observed_tactics/submit_pipeline.sh
BACKEND=trtllm-128x4 ./experiments/ultra_mxfp8_all_observed_tactics/submit_pipeline.sh
BACKEND=trtllm-8x4 ./experiments/ultra_mxfp8_all_observed_tactics/submit_pipeline.sh
```

CuTeDSL, CUTLASS, and the layout-matched TRTLLM arm use `128x4`. The second
TRTLLM arm uses fixed `8x4`. All four are required in the validated summary.
Interpret the first three as a backend-only comparison and `trtllm-8x4` as a
backend-and-layout recipe comparison.

After all four pipelines complete, aggregate their printed result roots:

```bash
python experiments/ultra_mxfp8_all_observed_tactics/summarize_results.py \
  --result cute-dsl=/path/to/cute-dsl-result \
  --result cutlass=/path/to/cutlass-result \
  --result trtllm-128x4=/path/to/trtllm-128x4-result \
  --result trtllm-8x4=/path/to/trtllm-8x4-result \
  --output-json=/path/to/summary.json \
  --output-csv=/path/to/e2e.csv
```

Aggregation fails unless all four arms have common invariant metadata, matching
workloads and generated outputs, complete two-order pairs, complete rank
coverage, a content-bound lookup manifest, and positive lookup hits.

An arm that fails empirically before producing valid serving, oracle, and lookup
artifacts can be represented by an evidence-backed status JSON instead of a
`--result`. At most one arm may use this status; the other three must be measured.
The four expected arm names are still required:

```bash
python experiments/ultra_mxfp8_all_observed_tactics/summarize_results.py \
  --result cute-dsl=/path/to/cute-dsl-result \
  --unsupported cutlass=/path/to/cutlass-status.json \
  --result trtllm-128x4=/path/to/trtllm-128x4-result \
  --result trtllm-8x4=/path/to/trtllm-8x4-result \
  --output-json=/path/to/summary.json \
  --output-csv=/path/to/e2e.csv
```

The unsupported status has this exact outer and failure schema. Every attempt
must use a unique job ID and have at least one evidence record linked by the
same `job_id` and `mode`. Evidence paths must be unique and may be absolute or
relative to the status JSON directory; every referenced file must exist and
match its lowercase SHA256 digest. Canonical output records use the resolved
absolute evidence paths.

```json
{
  "backend": "cutlass",
  "status": "empirically_unsupported",
  "recipe": {
    "backend_name": "cutlass",
    "linear_backend": "flashinfer_cutlass",
    "oracle_backend": "cutlass",
    "scale_layout": "128x4",
    "trtllm_layout": "8x4"
  },
  "provenance": {
    "source_commit": "...",
    "flashinfer_commit": "...",
    "expected_vllm_version": "...",
    "flashinfer": "...",
    "flashinfer_file": "...",
    "vllm_version": "...",
    "vllm_file": "...",
    "vllm_compiled_file": "...",
    "gpu_name": "...",
    "driver_version": "...",
    "container": "...",
    "container_sha256": "...",
    "container_size": "...",
    "container_mtime": "...",
    "model": "...",
    "model_config_sha256": "...",
    "model_index_sha256": "...",
    "model_weights_manifest_sha256": "...",
    "tp": "...",
    "moe_backend": "...",
    "attention_backend": "...",
    "kv_cache_dtype": "...",
    "max_model_len": "...",
    "max_num_batched_tokens": "...",
    "max_num_seqs": "...",
    "gpu_memory_utilization": "...",
    "enable_chunked_prefill": "...",
    "enable_prefix_caching": "...",
    "mamba_cache_mode": "...",
    "mamba_ssm_cache_dtype": "...",
    "cudagraph_capture_sizes": "...",
    "prompt_multiplier": "..."
  },
  "failure": {
    "stage": "server_startup",
    "reason_code": "backend_initialization_failed",
    "message": "Observed failure message",
    "attempts": [
      {"job_id": "123", "mode": "eager", "outcome": "failed"},
      {"job_id": "124", "mode": "cuda_graph", "outcome": "failed"}
    ],
    "evidence": [
      {
        "job_id": "123",
        "mode": "eager",
        "path": "logs/123.stderr",
        "sha256": "<64 lowercase hex characters>"
      },
      {
        "job_id": "124",
        "mode": "cuda_graph",
        "path": "logs/124.stderr",
        "sha256": "<64 lowercase hex characters>"
      }
    ]
  }
}
```

Unsupported provenance is compared with measured arms only for the stable
source, runtime, container, model, GPU, and recipe fields shown above. It must
not contain `cudagraph_configured`, `cudagraph_capture_status`,
`cudagraph_capture_evidence`, or `cudagraph_capture_marker`; those fields imply
a completed serving run. Final A/B `workloads`, `concurrencies`, and
`enforce_eager` are not required or compared for an unsupported arm; the status
may retain their observed capture-attempt values instead.

The failure stage must be `server_startup` or `serving_capture`. Attempts must
cover both `eager` and `cuda_graph` modes, use unique job IDs, and use a failure
outcome such as `failed`, `timed_out`, `cancelled_after_stall`, `engine_dead`,
`out_of_memory`, or `initialization_error`. Every attempt must be covered by a
distinct checksummed evidence path. Provenance accepts only the stable fields
above plus the optional observed-attempt fields `enforce_eager`, `workloads`,
and `concurrencies`.

The canonical JSON keeps four full records in expected backend order under
`backends`. Measured records have `status: measured`; the unsupported record has
no `e2e`, `oracle`, or `lookup` sections. A partial study reports
`study_status: complete_with_unsupported_arm`, ordered `measured_backends` and
`unsupported_backends` name lists, and `metric_comparison_status: partial`.
The E2E CSV contains measured arms only. A four-measured study remains
backward compatible with the original `{"backends": [...]}` JSON and E2E CSV
schema. `build_study_summary()` still exposes `study_status: complete` for
callers that explicitly request the richer in-memory representation.

Durable traces, lookup metadata, oracle reports, and benchmark JSON files are
written below the printed Lustre result root. JIT, autotune, and Python caches
are created under `/raid/scratch` and removed when each job exits.

The server loads the pinned vLLM Python tree but resolves the compiled vLLM
extension from the SHA256-pinned v0.27.1 container. It uses the container's installed
FlashInfer package and compiled modules. The separate FlashInfer checkout pins
only the offline-profiler scripts; workers copy those scripts to node-local
scratch so the checkout cannot shadow the installed `flashinfer` package.

Serving jobs use `gb200`. The longer exhaustive oracle defaults to the
eight-hour `gb200-backfill` partition; override `ORACLE_PARTITION` when needed.

## Interpretation Limits

Order reversal reduces order bias, but two pairs do not provide a confidence
interval. Same-node pairing controls each A/B ratio; absolute comparisons
between backend arms can still include node-allocation variance. CUDA Graph
completion proves capture occurred, not that every kernel replayed a graph.
Oracle microbenchmark gains are supporting evidence; only paired serving
metrics support E2E claims.
