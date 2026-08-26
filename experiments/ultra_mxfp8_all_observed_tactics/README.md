# Nemotron 3 Ultra All-Observed MXFP8 Tactic Lookup

This experiment repeats the vLLM 0.20.2 all-observed workflow on vLLM 0.27.1.
It does not assume that serving uses only power-of-two token counts. It records
every exact dense MXFP8 `(M, N, K)` seen by Python tactic selection during the
Ultra capture and baseline workloads, profiles every valid tactic for each
shape under CUDA Graph replay and cold L2, then compares the generated lookup
with the normal vLLM autotuner.

## Contract

- Model: Nemotron 3 Ultra MXFP8
- Hardware: one GB200 node
- Parallelism: TP4, DP1, expert parallel enabled
- Dense backend: CuTeDSL, FlashInfer CUTLASS, or FlashInfer TRTLLM
- MoE backend: FlashInfer TRTLLM
- Workloads: 1K/10K, 10K/1K, and 1K/1K ISL/OSL
- Shape capture concurrency: 1, 2, 4, 8, 16, 32; ten request waves
- Eager shape probe: 1K/64 and 10K/64 ISL/OSL to expose dynamic scheduler
  mixtures without paying eager-mode cost for long decode
- CUDA Graph baseline and shape capture: full 1K/10K, 10K/1K, and 1K/1K
  workloads
- Final baseline/lookup A/B concurrency: 1, 8, 32; ten request waves
- Final execution: CUDA Graph enabled
- Source contract: clean, detached experiment and FlashInfer worktrees pinned
  by SHA; installed vLLM runtime pinned by the container image SHA256
- Correctness: finite and elementwise oracle checks, followed by deterministic
  generated-text parity between baseline and lookup

The short eager capture discovers irregular runtime values such as `M=1001`
without spending hours decoding in eager mode. Ten waves expose
continuous-batching mixtures. The baseline run executes the full CUDA Graph
workloads, records every shape, and supplies the before measurement. The
offline lookup uses an exact `(M, N, K, runner)` key. A miss always delegates
to the normal FlashInfer autotuner. The final lookup run repeats the baseline
workloads after the oracle is built. The report includes unique-key and
selection-call-weighted lookup coverage. CUDA Graph replay does not re-enter
Python tactic selection, so this count is not an execution-weighted kernel
coverage metric. A single baseline/lookup pair is exploratory; sub-percent E2E
claims require repeated, order-balanced runs on the same node.

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
TRTLLM arm uses fixed `8x4`. Report the first three as a backend comparison;
report `trtllm-8x4` separately as a backend-and-layout recipe comparison.

Durable traces, lookup metadata, oracle reports, and benchmark JSON files are
written below the printed Lustre result root. JIT, autotune, and Python caches
are created under `/raid/scratch` and removed when each job exits.

Serving jobs use `gb200`. The longer exhaustive oracle defaults to the
eight-hour `gb200-backfill` partition; override `ORACLE_PARTITION` when needed.
