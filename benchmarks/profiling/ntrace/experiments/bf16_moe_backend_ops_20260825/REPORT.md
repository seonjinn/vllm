# BF16 Triton versus FlashInfer-TRTLLM ntrace report

## Result

FlashInfer-TRTLLM is faster because its monolithic routed-expert path reduces
both expert-compute time and the number and cost of routing/finalization
kernels. The matched rank-0 CUDA Graph trace attributes essentially the entire
kernel-time reduction to MoE: `12.015 -> 9.999 ms` per decode replay. Mamba and
Attention remain at parity.

Both arms used vLLM 0.27.1, BF16 Nemotron 3 Ultra, TP8/DP1/PP1, expert
parallelism, FlashInfer attention, concurrency 8, and CUDA Graph
`FULL_AND_PIECEWISE`. Each completed the exact `8,000 input / 2,048 output`
token contract and contains exactly 255 steady decode graph replays.

| Matched ntrace run | Triton | FlashInfer-TRTLLM | Change |
| --- | ---: | ---: | ---: |
| Request-wave latency | 15.419 s | 12.985 s | -15.8% |
| Output throughput | 132.82 tok/s | 157.73 tok/s | +18.8% |
| Mean TTFT | 4,441.9 ms | 3,644.1 ms | -18.0% |
| Mean TPOT | 43.02 ms | 36.60 ms | -14.9% |
| Decode graph p50 replay span | 15.140 ms | 12.560 ms | -17.0% |
| Decode graph p50 anchor cadence | 41.289 ms | 34.594 ms | -16.2% |
| GPU kernel sum per replay | 15.330 ms | 13.323 ms | -13.1% |
| Graph nodes per replay | 1,695 | 1,407 | -288 (-17.0%) |

Absolute latency includes ntrace overhead. The matched deltas, graph structure,
and operation composition are the intended evidence.

## Per-replay module breakdown

Times are exclusive sums of rank-0 GPU kernel durations. Concurrent streams
can make these sums larger than elapsed wall time.

| Module | Triton | FlashInfer-TRTLLM | Change |
| --- | ---: | ---: | ---: |
| MoE | 12.015 ms | 9.999 ms | -16.8% |
| Mamba | 2.723 ms | 2.736 ms | +0.5% |
| Attention | 0.574 ms | 0.570 ms | -0.6% |
| Other | 0.018 ms | 0.018 ms | -1.4% |

MoE saves `2.015 ms`, while the full kernel sum saves `2.007 ms`. The small
difference is explained by Mamba being `0.013 ms` slower in this sample.

### Whole-replay GEMM and BMM portion

The additive denominator is the exclusive GPU kernel-duration sum per decode
replay: `15.330 ms` for Triton and `13.323 ms` for FlashInfer-TRTLLM. This is
the defensible denominator for an operation portion that reconciles to 100%.
It is not an exclusive wall-clock split because kernels on concurrent streams
can overlap.

| GEMM/BMM family | Triton | Portion | FlashInfer-TRTLLM | Portion | Time change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Routed expert GEMM/BMM | 5.378 ms | 35.1% | 4.265 ms | 32.0% | -20.7% |
| Dense GEMM | 3.547 ms | 23.1% | 3.732 ms | 28.0% | +5.2% |
| **All GEMM/BMM** | **8.925 ms** | **58.2%** | **7.997 ms** | **60.0%** | **-10.4%** |

The GEMM/BMM absolute time falls by 10.4%, while its portion increases from
58.2% to 60.0%. This is not a contradiction: routing, support, and other
non-GEMM overhead shrink more than the GEMM/BMM total.

| GEMM/BMM operation | Triton | Portion | FlashInfer-TRTLLM | Portion | Time change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Routed W13 + activation | 2.925 ms | 19.1% | 2.113 ms | 15.9% | -27.8% |
| Routed W2 | 2.453 ms | 16.0% | 2.152 ms | 16.2% | -12.3% |
| Router projection | 0.938 ms | 6.1% | 0.944 ms | 7.1% | +0.7% |
| Mamba input projection | 0.792 ms | 5.2% | 0.798 ms | 6.0% | +0.8% |
| Shared W13/up-gate | 0.558 ms | 3.6% | 0.692 ms | 5.2% | +23.9% |
| Mamba output projection | 0.384 ms | 2.5% | 0.383 ms | 2.9% | -0.1% |
| Shared gate/combine GEMM | 0.373 ms | 2.4% | 0.373 ms | 2.8% | +0.1% |
| Shared W2/down | 0.299 ms | 1.9% | 0.332 ms | 2.5% | +11.2% |
| Attention QKV projection | 0.138 ms | 0.9% | 0.142 ms | 1.1% | +2.7% |
| Attention output projection | 0.066 ms | 0.4% | 0.067 ms | 0.5% | +1.8% |

The dominant gain is routed W13 (`-0.812 ms`) followed by routed W2
(`-0.301 ms`). Shared-expert W13 and W2 regress by a combined `0.167 ms`, but
this is much smaller than the `1.113 ms` routed-expert saving.

### Mamba operation breakdown

| Mamba operation per replay | Triton | FlashInfer-TRTLLM | Change |
| --- | ---: | ---: | ---: |
| Input projection | 0.792 ms | 0.798 ms | +0.8% |
| TP all-reduce | 0.547 ms | 0.541 ms | -1.0% |
| Output projection | 0.384 ms | 0.383 ms | -0.1% |
| Input norm | 0.248 ms | 0.242 ms | -2.5% |
| Selective state update | 0.211 ms | 0.208 ms | -1.4% |
| State/cache index update | 0.201 ms | 0.222 ms | +10.5% |
| Output gate/residual | 0.177 ms | 0.188 ms | +6.0% |
| Causal convolution | 0.164 ms | 0.154 ms | -6.4% |
| **Mamba total** | **2.723 ms** | **2.736 ms** | **+0.5%** |

The sub-operation shifts cancel out. The largest relative regression,
state/cache index update, is only `0.021 ms` in absolute time. There is no
evidence that changing the MoE backend materially changes Mamba performance.

### Attention operation breakdown

| Attention operation per replay | Triton | FlashInfer-TRTLLM | Change |
| --- | ---: | ---: | ---: |
| QKV projection | 0.138 ms | 0.142 ms | +2.7% |
| TP all-reduce | 0.128 ms | 0.132 ms | +3.7% |
| Attention core | 0.110 ms | 0.108 ms | -1.2% |
| Output projection | 0.066 ms | 0.067 ms | +1.8% |
| Attention setup/copy | 0.052 ms | 0.048 ms | -8.2% |
| Input norm | 0.046 ms | 0.045 ms | -2.4% |
| KV cache update | 0.035 ms | 0.028 ms | -19.0% |
| **Attention total** | **0.574 ms** | **0.570 ms** | **-0.6%** |

KV cache update has the largest relative improvement, but saves only
`0.0066 ms` per replay. Small increases in QKV projection, output projection,
and TP all-reduce offset it. Attention therefore remains at parity and does
not explain the end-to-end speedup.

## MoE root cause

| MoE bucket per replay | Triton | FlashInfer-TRTLLM | Saved | Share of total kernel saving |
| --- | ---: | ---: | ---: | ---: |
| Routed expert W13 + W2 | 5.378 ms | 4.265 ms | 1.113 ms | 55.5% |
| Routing, finalize, and routed support | 1.520 ms | 0.682 ms | 0.838 ms | 41.8% |
| MoE TP all-reduce | 2.598 ms | 2.373 ms | 0.226 ms | 11.3% |
| Router projection | 0.938 ms | 0.944 ms | -0.007 ms | -0.3% |
| Shared experts | 1.388 ms | 1.543 ms | -0.155 ms | -7.7% |
| MoE input norm | 0.193 ms | 0.192 ms | 0.001 ms | 0.0% |

The positive and negative contributions reconcile to the `2.007 ms` total
kernel saving.

The trace shows three concrete mechanisms:

1. Expert GEMMs are faster. Triton launches 48 W13 and 48 W2
   `fused_moe_kernel` nodes per replay. FlashInfer-TRTLLM launches the same
   logical 48 + 48 expert operations as tuned SM100 BF16 BMM kernels, but their
   combined time is 20.7% lower.
2. The monolithic TRTLLM path eliminates 288 graph nodes per replay, exactly
   six nodes for each of the 48 MoE layers. Triton exposes four routing nodes,
   one setup node, and two additional routed-support nodes per layer;
   FlashInfer-TRTLLM exposes one routing node and folds the rest into the
   monolithic path.
3. The shorter MoE path also reduces the MoE-owned TP all-reduce sum by 8.7%.
   This is observed timing, not proof that the collective implementation
   changed; earlier producer completion and less overlap/serialization can
   change collective duration inside the graph.

Shared-expert work is 11.2% slower with FlashInfer-TRTLLM, but its `0.155 ms`
regression is much smaller than the routed-path savings. Router projection,
Mamba, and Attention are near parity, which isolates the advantage to the MoE
backend rather than the common attention backend or the rest of the model.

## Observed kernel paths

- Triton routed experts: `moe_align_block_size_kernel` -> W13
  `fused_moe_kernel` -> activation/support kernels -> W2 `fused_moe_kernel` ->
  `moe_sum_vec_dynamic_kernel`.
- FlashInfer-TRTLLM routed experts: `moe::dev::routing::*` -> tuned W13/W2
  `bmm_Bfloat16_*_sm100f` -> `moe::dev::finalize::*`.
- Both arms use the same common FlashInfer FMHA, Mamba kernels, dense BF16
  projections, and TP8 topology.

## Artifacts and validation

- SLURM jobs: Triton `2787474`, FlashInfer-TRTLLM `2787475`
- Local artifact root:
  `/Users/sna/MXFP8_generation/deliverables/bf16_moe_backend_ntrace_20260825/rayv2_decode`
- Combined one-page HTML:
  `/Users/sna/MXFP8_generation/deliverables/bf16_moe_backend_ntrace_20260825/bf16_moe_backend_comparison.html`
- Per-arm interactive hierarchy:
  `<artifact root>/<backend>/breakdown/silicon_breakdown.html`
- Per-arm trace files: records, stacks, graph nodes, and memops Parquet
- Decode replay validation: `255/255` in both arms
- Recovered model structure: 48 Mamba, 12 Attention, and 48 MoE blocks in both
  arms
- Hierarchical attribution reconciles every graph node and its kernel-duration
  sum

## All-rank validation

The follow-up captures completed for ranks 0-7 in both arms:

- Triton job: `2788246`
- FlashInfer-TRTLLM job: `2788247`
- Exact output tokens: `2,048/2,048` in both arms
- Decode graph nodes on every rank: Triton `1,695`, FlashInfer-TRTLLM `1,407`
- MoE kernel-time change across ranks: `-20.2%` to `-19.7%`, mean `-19.9%`

The MoE reduction is therefore not a rank-0-only effect. The eight-rank
simultaneous CUPTI capture does perturb collective waiting: total kernel sum
and graph span vary strongly by rank even though the MoE time is stable. Its
request throughput and cross-rank total span are not used as production
performance evidence. Never sum rank timing; production throughput remains
tied to an unprofiled matched sweep.
