# Nemotron 3 Ultra TP8 Serving-Shape Crossover

This experiment extends the all-observed MXFP8 tactic workflow from TP4 to
TP8 across two GB200 nodes. The first submission records the exact dense MXFP8
GEMM shapes produced by a production-like 1K/10K serving workload.

The initial submission is intentionally split:

- `low`: C1, C2, C4, C8, C16, and C32 with ten request waves.
- `high-smoke`: C128 and C512 with one request wave.

The high-concurrency smoke prevents an unsupported KV-cache point or a
5,120-request, 51.2-million-output-token C512 run from consuming a long
allocation. A successful smoke is promoted to ten waves in the final A/B.

## Dry Run

```bash
PRINT_PLAN=1 ./experiments/ultra_mxfp8_tp8_serving_crossover/submit_shape_census.sh
```

## Submit

```bash
PHASES="low high-smoke" \
  ./experiments/ultra_mxfp8_tp8_serving_crossover/submit_shape_census.sh
```

The script verifies clean and pinned vLLM, benchmark-harness, FlashInfer,
container, and model inputs. It runs `sbatch --test-only` before each actual
submission and prints the job ID and result directory.

See [PLAN.md](PLAN.md) for the full five-arm study and validation gates.
