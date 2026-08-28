# Adaptive MXFP8 TRTLLM CUDA Graph Validation

## Setup

- GPU: 4 x GB200
- Model: Nemotron 3 Ultra MXFP8
- Runtime: vLLM 0.27.1, FlashInfer 0.6.16.post3
- Parallelism: TP4, DP1, EP4
- Linear and MoE backend: `flashinfer_trtllm`
- CUDA Graph: `FULL_AND_PIECEWISE`, capture sizes 1, 8, and 32
- Source commit: `ca5ea0e1889f81c1cb724b3a7b0ee39222df0804`

## Result

Adaptive 8x4/128x4 selection works with CUDA Graph when both conditions below
are met:

1. Disable PDL in the MXFP8 activation quantizers.
2. Preallocate a persistent workspace large enough to avoid the TRTLLM GEMM
   fallback allocation. The validated run used 1 GiB per GPU.

Job `2816005` completed PIECEWISE and FULL graph capture, started the API
server, and served all eight 1K/1K requests without failure.

| Configuration | Job | Result | Output tok/s | tok/s/GPU |
| --- | ---: | --- | ---: | ---: |
| Fixed 8x4 reference | 2812861 | Pass | 505.98 | 126.50 |
| Adaptive, heuristic tactics, PDL off | 2815585 | Pass | 465.19 | 116.30 |
| Adaptive, tuned tactics, PDL off, 32 MiB workspace | 2815877 | Hang | - | - |
| Adaptive, tuned tactics, PDL off, 1 GiB workspace | 2816005 | Pass | 507.06 | 126.76 |

The tuned adaptive run is 9.0% faster than the heuristic-only adaptive run and
0.21% faster than the fixed-8x4 reference at C8. Treat the latter as parity:
the two runs used different request wave counts and GPU memory utilization, and
this workload is dominated by decode shapes that select 8x4.

## Root-Cause Evidence

The failure occurs after FlashInfer autotuning and before CUDA Graph capture,
at the first device synchronization. The following observations separate the
two contributing conditions:

- PDL off plus skipped dense autotuning completes graph capture and generation.
- PDL off plus cached tuned tactics still hangs with the default 32 MiB
  workspace.
- A 1 GiB persistent workspace with PDL still enabled previously hung.
- PDL off plus the 1 GiB persistent workspace completes fresh tuning, graph
  capture, and generation.

FlashInfer's TRTLLM GEMM binding allocates a local CUDA tensor when the supplied
32 MiB workspace is smaller than the selected tactic's requirement. It launches
the GEMM asynchronously and returns immediately. Repeated model-level GEMMs can
therefore outlive and reuse that fallback storage. Preallocating the shared
workspace keeps the selected tactics on the persistent-buffer path.

## Verification

- Warmup/cache tests: 10 passed
- MXFP8 linear-kernel tests: 26 passed
- End-to-end CUDA Graph run: 8 successful requests, 0 failed requests
- Job `2816005`: `COMPLETED`, exit code `0:0`, elapsed `00:10:34`

The end-to-end run establishes runtime stability. A matched accuracy dataset
run is still required before making a model-quality claim.

## Artifacts

- Passing tuned run:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_mxfp8_adaptive_cg_pdl_off_ws1g_cached_20260827/adaptive/2816005`
- Passing heuristic run:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_mxfp8_adaptive_cg_pdl_off_skip_tune_20260827/adaptive/2815585`
- Full 239-entry tactic cache:
  `/lustre/fsw/coreai_dlalgo_llm/users/sna/vllm-v0271-results/ultra_mxfp8_adaptive_cg_pdl_off_ws1g_cached_20260827/seed-cache/d91a0e2ddac036e5e0eb9ffd616d39f2ff240565544e7a0006660f54e5371574/autotune_configs.json`
