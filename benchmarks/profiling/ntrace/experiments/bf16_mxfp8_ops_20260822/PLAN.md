# Plan

1. Validate the four SLURM requests with `sbatch --test-only`.
2. Capture BF16 and MoE-only MXFP8 prefill/decode traces.
3. Verify exact-token completion and ntrace graph replay coverage.
4. Run the ntrace breakdown and compare reports.
5. Add the measured kernel sequence and timing to the BF16 versus MXFP8 HTML
   explainer, with trace perturbation and backend scope stated explicitly.
