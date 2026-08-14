# Nemotron 3 Ultra MXFP8 MoE Backend Comparison

This experiment compares vLLM v0.27.1 MXFP8 MoE backends on one GB200 node.
Only `--moe-backend` changes between runs.

## Fixed setup

- Model: Nemotron 3 Ultra MXFP8 local checkpoint
- GPUs: 4 GB200 GPUs
- Parallelism: TP4, DP1, expert parallel enabled
- Workload: random 1K input / 1K output tokens
- Concurrency: 1, 8, 32
- Requests: 10 waves per concurrency
- Dense linear backend: `auto`
- KV cache dtype: `auto`
- Chunked prefill and prefix caching: enabled

## Backend matrix

The native GB200 W8A8 candidates are `flashinfer_trtllm` and `humming`. `auto`
is included to identify the default selection. `marlin` is measured separately
because it is a W8A16 fallback. `deep_gemm` and `triton` are submitted as
capability checks: Ultra's non-gated MoE is expected to reject DeepGEMM, and the
v0.27.1 native Triton MXFP8 implementation is expected to reject GB200. Startup
failures are preserved as results instead of being silently omitted.

## Submit

Stage the immutable v0.27.1 image first:

```bash
sbatch --test-only \
  --account=coreai_dlalgo_llm \
  --partition=gb200 \
  --qos=user-restrictions \
  --export=ALL,SOURCE_IMAGE=docker.io/vllm/vllm-openai:v0.27.1,OUTPUT_PREFIX=vllm_openai_v0271_aarch64,CONTAINER_DIR=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers,SOURCE_COMMIT=6e448d0ea9bf3d88d898b65449ca6dc2aec170ac \
  experiments/ultra_mxfp8_moe_backends/stage_enroot_image.sbatch
```

After the image passes its smoke check:

```bash
SBATCH_TEST_ONLY=1 experiments/ultra_mxfp8_moe_backends/submit.sh
experiments/ultra_mxfp8_moe_backends/submit.sh
```

Each result directory contains the server log, one benchmark JSON per
concurrency, the exact command line, package versions, and the SLURM output.
