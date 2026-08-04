from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def _load_flashinfer_module() -> ModuleType:
    module_name = "vllm.model_executor.kernels.linear.mxfp8.flashinfer"
    module_path = os.environ.get("MXFP8_FLASHINFER_MODULE_FILE")
    if module_path is None:
        return importlib.import_module(module_name)

    importlib.import_module("vllm.model_executor.kernels.linear.mxfp8")
    spec = importlib.util.spec_from_file_location(module_name, Path(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MXFP8 kernel module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=("flashinfer_cutlass", "flashinfer_cutedsl", "flashinfer_trtllm"),
    )
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    return parser.parse_args()


def _quantize_weight(
    flashinfer: ModuleType,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, scale = flashinfer.mxfp8_quantize(
        weight,
        backend="cuda",
        sf_swizzle_layout=flashinfer.SfLayout.layout_linear,
    )
    return quantized, scale.view(weight.shape[0], weight.shape[1] // 32)


def _cosine_similarity(output: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            output.float().flatten(),
            reference.float().flatten(),
            dim=0,
        ).item()
    )


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise RuntimeError("SM100 or SM103 is required")

    flashinfer = importlib.import_module("flashinfer")
    kernel_module = _load_flashinfer_module()
    config_module = importlib.import_module(
        "vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel"
    )
    kernel_types = {
        "flashinfer_cutlass": kernel_module.FlashInferCutlassMxfp8LinearKernel,
        "flashinfer_cutedsl": kernel_module.FlashInferCutedslMxfp8LinearKernel,
        "flashinfer_trtllm": kernel_module.FlashInferTrtllmMxfp8LinearKernel,
    }

    torch.manual_seed(7)
    x = torch.randn((args.m, args.k), device="cuda", dtype=torch.bfloat16) * 0.1
    weight_bf16 = (
        torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16) * 0.02
    )
    weight, weight_scale = _quantize_weight(flashinfer, weight_bf16)

    layer = torch.nn.Module()
    layer.weight = Parameter(weight, requires_grad=False)
    layer.weight_scale = Parameter(weight_scale, requires_grad=False)
    kernel = kernel_types[args.backend](config_module.Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    prepared_names = tuple(
        name
        for name in ("weight_for_apply", "weight_scale_for_apply")
        if hasattr(layer, name)
    )
    prepared_pointers = {
        name: getattr(layer, name).data_ptr() for name in prepared_names
    }
    weight_pointer = layer.weight.data_ptr()
    scale_pointer = layer.weight_scale.data_ptr()

    compiled_apply = torch.compile(
        lambda input_: kernel.apply_weights(layer, input_),
        fullgraph=True,
        dynamic=False,
    )
    with flashinfer.autotune(False):
        for _ in range(3):
            compiled_apply(x)
        torch.cuda.synchronize()

        static_x = x.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = compiled_apply(static_x)
        graph.replay()
        torch.cuda.synchronize()

    first_reference = x @ weight_bf16.t()
    first_similarity = _cosine_similarity(static_output, first_reference)
    if first_similarity <= 0.95 or not torch.isfinite(static_output).all():
        raise AssertionError(f"invalid first output: similarity={first_similarity}")

    next_weight_bf16 = (
        torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16) * 0.02
    )
    next_weight, next_scale = _quantize_weight(flashinfer, next_weight_bf16)
    with torch.no_grad():
        layer.weight.copy_(next_weight)
        layer.weight_scale.copy_(next_scale)
    kernel.process_weights_after_loading(layer)

    if layer.weight.data_ptr() != weight_pointer:
        raise AssertionError("checkpoint weight pointer changed across refit")
    if layer.weight_scale.data_ptr() != scale_pointer:
        raise AssertionError("checkpoint scale pointer changed across refit")
    for name, pointer in prepared_pointers.items():
        if getattr(layer, name).data_ptr() != pointer:
            raise AssertionError(f"prepared pointer changed across refit: {name}")

    static_x.copy_(x)
    graph.replay()
    torch.cuda.synchronize()
    second_reference = x @ next_weight_bf16.t()
    second_similarity = _cosine_similarity(static_output, second_reference)
    if second_similarity <= 0.95 or not torch.isfinite(static_output).all():
        raise AssertionError(f"invalid refit output: similarity={second_similarity}")

    print(
        f"backend={args.backend} status=PASS "
        f"first_similarity={first_similarity:.6f} "
        f"refit_similarity={second_similarity:.6f} "
        f"prepared_buffers={','.join(prepared_names) or 'none'}"
    )


if __name__ == "__main__":
    main()
