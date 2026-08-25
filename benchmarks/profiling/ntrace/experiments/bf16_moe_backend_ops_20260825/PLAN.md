# Plan

1. Build the pinned `compute/e2etrain/ntrace` CUDA 13 runtime on OCI-HSG.
2. Validate the two TP8 requests with `sbatch --test-only`.
3. Capture rank 0 for BF16 Triton and FlashInfer-TRTLLM at C8.
4. Validate exact tokens and CUDA Graph replay coverage.
5. Expand to ranks 0-7 only after the rank-0 smoke passes.
6. Compare exclusive GPU kernel time by module, operation, kernel family, and
   collective subtype; reconcile every category to the raw kernel sum.
