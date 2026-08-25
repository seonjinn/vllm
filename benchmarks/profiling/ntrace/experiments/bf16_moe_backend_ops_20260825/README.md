# BF16 MoE backend ntrace comparison

This experiment tests why FlashInfer-TRTLLM is faster than Triton for BF16
Nemotron 3 Ultra generation.

## Contract

- Runtime: vLLM 0.27.1 with the pinned ntrace worker overlay
- Image: `vllm_openai_v0271_aarch64_20260813_2688476.sqsh`
- Image SHA256: `e7be53f2754097c88f7c801da92f6d94794ec4d78d9df937fcd315a6994297f0`
- Model: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`
- Hardware: Lyris, two GB200 nodes, TP8/DP1/PP1, expert parallel enabled
- MoE backends: Triton and FlashInfer-TRTLLM
- Attention backend: FlashInfer in both arms
- CUDA Graph: `FULL_AND_PIECEWISE`, capture size 8
- Workload: 1,000 input tokens, 256 generated tokens, concurrency 8
- Traffic: one fixed request wave after one exact warmup wave
- Initial trace scope: rank 0; expand to ranks 0-7 after smoke validation

The shorter 256-token output preserves 255 steady decode graph replays while
keeping the CUPTI trace bounded. Production throughput conclusions remain tied
to the separate unprofiled 1K/10K sweep.

## Launch

The ntrace source revision is `compute/e2etrain/ntrace@165ae08`. Build its
native CUDA 13 runtime first. OCI-HSG exposes container integration on `srun`,
so use the dedicated wrapper instead of passing container flags to `sbatch`:

```bash
SBATCH_TEST_ONLY=1 ./build_runtime_oci_hsg.sh
SBATCH_TEST_ONLY=0 ./build_runtime_oci_hsg.sh
```

After the build succeeds, validate profiling-job scheduling:

```bash
SBATCH_TEST_ONLY=1 ./submit.sh
```

Submit the two rank-0 arms only after the scheduling checks pass:

```bash
SBATCH_TEST_ONLY=0 ./submit.sh
```

Each arm writes a metadata manifest and rank-scoped Parquet files. ntrace
perturbs absolute latency, so the primary result is matched operation
composition and kernel-time delta, not profiled requests per second.

## Result

The validated rank-0 result and root-cause analysis are in [REPORT.md](REPORT.md).
The successful SLURM jobs are `2787474` (Triton) and `2787475`
(FlashInfer-TRTLLM).
