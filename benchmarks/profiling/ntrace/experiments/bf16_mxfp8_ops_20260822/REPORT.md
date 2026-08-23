# BF16 versus MoE-only MXFP8 ntrace report

## Result

The matched four-arm capture completed on GB200 with vLLM 0.27.1. All arms
used TP8/DP1/PP1, expert parallelism, BF16 KV cache, FlashInfer TRTLLM MoE,
CUDA Graph `FULL_AND_PIECEWISE`, and one C8 request wave after warmup.

| Workload | BF16 | MXFP8 | Change |
| --- | ---: | ---: | ---: |
| 10K to 1 prefill-workload kernel sum, rank 0 | 1,618.50 ms | 1,491.98 ms | -7.8% |
| 10K to 1 MoE GEMM kernel sum, rank 0 | 340.20 ms | 196.57 ms | -42.2% |
| Decode graph p50 replay span | 13.143 ms | 10.548 ms | -19.7% |
| Decode graph p50 anchor cadence | 35.057 ms | 31.732 ms | -9.5% |
| Decode graph kernel sum per replay | 14.065 ms | 11.637 ms | -17.3% |
| Decode graph MoE GEMM sum per replay | 4.728 ms | 2.824 ms | -40.3% |
| Decode graph MXFP8 activation quantize per replay | 0 ms | 0.110 ms | +0.110 ms |

MXFP8 adds 48 CUDA Graph nodes per decode replay: 1,455 nodes versus 1,407
for BF16. The 48 added nodes are the profiler-visible MXFP8 activation
quantization kernels, one per routed MoE layer. Their 0.110 ms summed cost per
replay is much smaller than the 1.904 ms reduction in routed-expert GEMM time.

The prefill request wave completed the exact `80,000 input / 8 output` token
contract in both arms. The generation wave completed the exact
`8,000 input / 2,048 output` contract and contained exactly 255 CUDA Graph
decode replays in both arms.

## Hierarchical GPU-work breakdown

The hierarchy recovers the expected 48 Mamba, 12 Attention, and 48 MoE blocks
in each decode graph. Every kernel node has exactly one owner, and the assigned
durations reconcile with the raw kernel sum. Times below are exclusive sums of
GPU kernel durations, not additive wall time; shared-expert work overlaps the
main stream.

| Decode module / replay | BF16 | MXFP8 | Change |
| --- | ---: | ---: | ---: |
| MoE | 10.726 ms (76.3%) | 8.280 ms (71.2%) | -22.8% |
| Mamba | 2.748 ms (19.5%) | 2.758 ms (23.7%) | +0.3% |
| Attention | 0.573 ms (4.1%) | 0.581 ms (5.0%) | +1.4% |
| Other | 0.018 ms (0.1%) | 0.019 ms (0.2%) | +3.4% |

The largest MoE changes are routed W13 plus activation (`2.361 -> 1.407 ms`),
routed W2 (`2.367 -> 1.417 ms`), and the module-owned TP all-reduce interval
(`2.390 -> 1.655 ms`). MXFP8 adds `0.110 ms` of activation quantization.
Mamba and Attention are outside this MoE-only quantization scope and remain
near parity. Attention includes QKV projection, FMHA, KV-cache update, O
projection, setup, normalization, and its TP collective. Mamba includes input
and output projections, causal convolution, selective-state and cache updates,
gate/residual work, normalization, and its TP collective.

The 10K-to-1 prefill wave contains six model-pass signatures due to chunked
prefill. Across the full request wave, MoE GPU work changes from `915.763` to
`774.209 ms`, Mamba from `606.197` to `621.640 ms`, and Attention from `95.755`
to `95.329 ms`.

## Observed decode order

The dominant CUDA stream repeats this layer-level order:

1. Residual and normalization kernels.
2. Dense BF16 projection kernels and Mamba or attention kernels.
3. Routed-expert selection.
4. BF16: W13 BF16 BMM, activation, W2 BF16 BMM.
5. MXFP8: activation quantize, W13 MXFP8 BMM, fused activation path, W2 MXFP8 BMM.
6. Expert finalize, TP collective, and residual update.

Shared-expert dense kernels overlap on a second stream. The report therefore
treats the global timestamp order as observed rather than causal. Order within
one CUDA stream is causal.

## Interpretation limits

- ntrace changes absolute latency. The trace is for kernel sequence and matched
  composition, not production throughput.
- Kernel sum includes overlap across streams. Kernel union and activity union
  single-count overlapping intervals.
- `window - activity_union` is `no recorded rank-0 activity`; it is not proof
  of GPU idle or CPU overhead.
- The prefill arm is a 10K-to-1 first-token request wave. The decode arm is a
  1K-to-256 generation wave; only the validated 255 CUDA Graph replays are
  labeled decode model steps.
- The trace covers rank 0 of eight TP ranks. The separate unprofiled and Nsight
  runs provide production throughput and multi-rank evidence.

## Artifacts

- SLURM jobs: `2764382`, `2764383`, `2764384`, `2764385`
- Copied traces and benchmark JSON:
  `/Users/sna/MXFP8_generation/deliverables/bf16_mxfp8_ntrace_ops_20260822/matched_shape_fix`
- Machine-readable comparison: `matched_shape_fix/analysis.json`
- Per-arm llm-analyzer reports: `matched_shape_fix/<arm>/breakdown/`

## Validation

- Four-arm metadata and exact-token contract: passed
- Decode replay count: BF16 `255/255`, MXFP8 `255/255`
- Synthetic analyzer tests: 38 passed
- Ruff: passed
- Pyright: passed
- `git diff --check`: passed
