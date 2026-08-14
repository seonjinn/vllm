# Nemotron 3 Ultra backend sweep on vLLM v0.27.1

This experiment isolates backend selection on GB200:

- MXFP8 dense: vary `--linear-backend`, keep MoE at `flashinfer_trtllm`.
- MXFP8 MoE: vary `--moe-backend`, keep dense linear at `auto` (CuteDSL).
- BF16 MoE is submitted from the companion `vllm-benchmark` worktree because
  the BF16 checkpoint needs two nodes and the existing Ray harness.

Start with `PROFILE=smoke` to reject unsupported backends before launching the
three-workload, three-concurrency `PROFILE=full` matrix. Every result is checked
for failed requests and exact input/output token counts.

```bash
SBATCH_TEST_ONLY=1 PROFILE=smoke ./experiments/ultra_backend_sweep_v0271/submit_mxfp8.sh
PROFILE=smoke ./experiments/ultra_backend_sweep_v0271/submit_mxfp8.sh
```
