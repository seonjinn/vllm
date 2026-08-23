# BF16 versus MoE-only MXFP8 kernel trace

This experiment captures matched rank-0 ntrace records for BF16 and MoE-only
MXFP8 Nemotron 3 Ultra generation.

## Contract

- Runtime: vLLM 0.27.1 with the pinned ntrace profiler overlay
- Hardware: two GB200 nodes, TP8/DP1/PP1, expert parallel enabled
- Weight lineage: one BF16 checkpoint; the MXFP8 checkpoint quantizes only
  routed expert `up_proj` and `down_proj` weights from that checkpoint
- KV cache: BF16
- MoE backend: FlashInfer TRTLLM for both arms
- Dense linear backend: vLLM default
- CUDA graph mode: `FULL_AND_PIECEWISE`, capture size 8
- Concurrency: 8, one fixed request wave after one warmup wave
- Trace scope: rank 0, one measured benchmark iteration

Two workload phases are captured:

| Phase | ISL | OSL | Purpose |
| --- | ---: | ---: | --- |
| prefill | 10,000 | 1 | isolate prompt processing |
| decode | 1,000 | 256 | expose steady autoregressive decoding |

ntrace changes absolute latency. Use these traces for kernel order, call-stack
attribution, and matched relative composition. Use unprofiled throughput and
the existing long-context Nsight runs for production latency claims.

## Launch

Run scheduler validation first:

```bash
SBATCH_TEST_ONLY=1 ./submit.sh
```

Then submit all four arms:

```bash
SBATCH_TEST_ONLY=0 ./submit.sh
```

The launcher writes a `metadata.env` file and one rank-0 Parquet artifact set
under each arm. It does not create per-kernel files.

## Analyze

Install `pyarrow`, then pass the four copied run directories explicitly:

```bash
python analyze_runs.py \
  --prefill-bf16 /path/to/prefill_bf16 \
  --prefill-mxfp8 /path/to/prefill_mxfp8 \
  --decode-bf16 /path/to/decode_bf16 \
  --decode-mxfp8 /path/to/decode_mxfp8 \
  --output analysis.json
```

The JSON contains benchmark and metadata provenance, clipped rank-0 kernel
counts and durations, category shares, bounded global and per-stream kernel
sequences, and per-phase MXFP8-minus-BF16 deltas. The global sequence is
observed timestamp order and is explicitly non-causal across streams; each
per-stream sequence preserves causal stream order.

`gpu_sum_ns` includes concurrent kernel overlap, while `gpu_union_ns` merges
that overlap. When `ntrace_memops_rank0.parquet` is present,
`memop_union_ns` is its clipped interval union. `activity_union_ns` merges
kernel and available memop intervals, `no_recorded_activity_ns` is the
iteration-window union not covered by that activity, and `overlap_factor` is
`gpu_sum_ns / gpu_union_ns`. Missing memops are reported explicitly and do not
prevent kernel analysis.

Decode arms are rejected unless profiler replay provenance validates exactly
`OSL - 1 = 255` dominant-graph replays. The report includes dominant graph
nodes per replay plus replay-span and anchor-period p50/p95/p99 values. Stack-
qualified routed/shared MoE ownership is not inferred because that requires
model-specific clone-aware rules. Use `--sequence-limit` to change the default
200 emitted segments per sequence. Missing or contradictory required metadata
is fatal; unavailable optional benchmark fields are listed without failing
kernel analysis.

Run the synthetic Parquet tests without modifying the project environment:

```bash
TMPDIR=/tmp uv run --isolated --no-project --python 3.12 \
  --with pytest --with pyarrow pytest -q -p no:cacheprovider \
  test_analyze_runs.py
```
