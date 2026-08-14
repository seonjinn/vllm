#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:15:00

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"

# The quoted script executes after entering the container.
# shellcheck disable=SC2016
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="/lustre:/lustre" \
  --mpi=pmix \
  bash -lc '
set -euo pipefail
uname -m
vllm --version
python3 - <<"PY"
import importlib.util

import flashinfer
import torch
import vllm

print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flashinfer", getattr(flashinfer, "__version__", "unknown"))
print("gpu_count", torch.cuda.device_count())
print("gpu_names", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
for package in ("deep_gemm", "humming"):
    print(package, "available" if importlib.util.find_spec(package) else "missing")
assert vllm.__version__ == "0.27.1"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
PY
'
