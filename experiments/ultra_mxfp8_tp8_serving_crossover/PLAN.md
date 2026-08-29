# Nemotron 3 Ultra TP8 MXFP8 Serving Crossover Plan

## Objective

Measure the production serving-shape crossover among CuTeDSL, fixed TRTLLM
8x4, fixed TRTLLM 128x4, adaptive TRTLLM, and adaptive TRTLLM with an exact
offline tactic lookup.

## Fixed Setup

- Model: Nemotron 3 Ultra MXFP8
- Hardware: two GB200 nodes, four GPUs per node
- Parallelism: TP8, DP1, EP8
- Workload: ISL 1K, OSL 10K, continuous request supply
- Runtime: vLLM 0.27.1 with CUDA Graph enabled
- MoE backend: FlashInfer TRTLLM
- Requests: ten waves per measured concurrency
- Concurrency: 1, 2, 4, 8, 16, 32, 128, 512

## Stages

1. Run the adaptive TRTLLM shape census at C1-C32 with ten waves.
2. Run C128 and C512 with one wave as a KV-cache and wall-time capacity smoke.
3. Merge all TP8 rank traces and build the exact observed `(M, N, K)` set.
4. Profile every legal tactic for both 8x4 and 128x4 under CUDA Graph replay
   and cold L2. Build a TP8-only exact lookup; do not reuse TP4 tables.
5. Run paired full-model A/B measurements for all five arms. Promote the high
   concurrency points to ten waves only after the capacity smoke passes.
6. Run deterministic generated-text parity and GSM8K correctness checks for
   the selected production policy.

## Validation Gates

- Every trace artifact is keyed by host and PID so two nodes cannot overwrite
  each other.
- Every completed worker writes a count snapshot and completion marker.
- Every serving rank 0-7 must appear in the merged trace.
- A lookup miss must use the normal FlashInfer autotuner.
- Exact tactics must pass finite, cosine, and elementwise checks before use.
- Final performance claims require same-node, reverse-order paired runs.

## Artifacts

- Source and launch files stay in `/home`.
- JIT, autotune, Ray, and Python caches stay in `/raid/scratch`.
- Durable JSON, logs, traces, and summaries stay in `/lustre`.
