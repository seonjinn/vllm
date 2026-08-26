# Nemotron 3 Ultra All-Observed MXFP8 Tactic Lookup

This experiment repeats the vLLM 0.20.2 all-observed workflow on vLLM 0.27.1.
It does not assume that serving uses only power-of-two token counts. It records
every exact dense MXFP8 `(M, N, K)` dispatched by the Ultra workload, profiles
every valid CuTeDSL tactic for each shape under CUDA Graph replay and cold L2,
then compares the generated lookup with the normal vLLM autotuner.

## Contract

- Model: Nemotron 3 Ultra MXFP8
- Hardware: one GB200 node
- Parallelism: TP4, DP1, expert parallel enabled
- Dense backend: vLLM `auto` (CuTeDSL on GB200)
- MoE backend: FlashInfer TRTLLM
- Workloads: 1K/10K, 10K/1K, and 1K/1K ISL/OSL
- Shape capture concurrency: 1, 2, 4, 8, 16, 32; ten request waves
- Eager shape probe: 1K/64 and 10K/64 ISL/OSL to expose dynamic scheduler
  mixtures without paying eager-mode cost for long decode
- CUDA Graph shape capture: full 1K/10K, 10K/1K, and 1K/1K workloads
- Final A/B concurrency: 1, 8, 32; ten request waves
- Final execution: CUDA Graph enabled

The short eager capture discovers irregular runtime values such as `M=1001`
without spending hours decoding in eager mode. Ten waves expose
continuous-batching mixtures. The graph capture runs the full workloads and
adds shapes chosen during graph construction. The offline lookup uses an exact
`(M, N, K, runner)` key. A miss always delegates to the normal FlashInfer
autotuner.

## Run

Push this branch before submission, then initialize the remote source under
`/home` and launch the dependency chain:

```bash
./experiments/ultra_mxfp8_all_observed_tactics/submit_pipeline.sh
```

Durable traces, lookup metadata, oracle reports, and benchmark JSON files are
written below the printed Lustre result root. JIT, autotune, and Python caches
are created under `/raid/scratch` and removed when each job exits.

Serving jobs use `gb200`. The longer exhaustive oracle defaults to the
eight-hour `gb200-backfill` partition; override `ORACLE_PARTITION` when needed.
