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
