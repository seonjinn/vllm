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
