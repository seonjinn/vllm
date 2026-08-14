# Nemotron 3 Ultra MXFP8 TRTLLM Layout Comparison

This experiment compares fixed 8x4 and adaptive 8x4/128x4 scale-factor
layouts for the FlashInfer TRTLLM MXFP8 dense linear backend on vLLM v0.27.1.

## Fixed setup

- Model: Nemotron 3 Ultra MXFP8 local checkpoint
- GPUs: one GB200 node, four GPUs
- Parallelism: TP4, DP1, expert parallel enabled
- Dense linear backend: `flashinfer_trtllm`
- MoE backend: `flashinfer_trtllm`
- Workloads: random 1K/1K, 10K/1K, and 1K/10K input/output tokens
- Concurrency: 1, 8, 32
- Requests: 10 waves per workload and concurrency
- Adaptive switch: 8x4 when flattened M is at most 256, otherwise 128x4

The default row uses the TRTLLM backend's fixed 8x4 layout. The adaptive row
changes only `VLLM_MXFP8_TRTLLM_LAYOUT=adaptive`; all serving and workload
settings remain identical.

A contextual run uses `LINEAR_BACKEND=auto`. On GB200, vLLM v0.27.1 resolves
that setting to FlashInfer CuteDSL for MXFP8 dense linear layers. This row
answers whether adaptive TRTLLM beats the released v0.27.1 default; it is not
part of the fixed-versus-adaptive TRTLLM A/B.

## Submit on Lyris

The remote source checkout must match `SOURCE_COMMIT`. Validate scheduling
before submitting:

```bash
SBATCH_TEST_ONLY=1 experiments/ultra_mxfp8_trtllm_adaptive/submit.sh
experiments/ultra_mxfp8_trtllm_adaptive/submit.sh
```

Each result directory records the exact source commit, mounted file hashes,
container, command, server log, and detailed benchmark JSON files.
