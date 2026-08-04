from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from types import ModuleType

import torch
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
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


def _capture_graph(
    kernel: object,
    layer: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    compiled_apply = torch.compile(
        lambda input_: kernel.apply_weights(layer, input_),
        fullgraph=True,
        dynamic=False,
    )
    for _ in range(5):
        compiled_apply(x)
    torch.cuda.synchronize()

    static_x = x.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled_apply(static_x)
    graph.replay()
    torch.cuda.synchronize()
    return graph, static_x


def _measure_graph_us(
    graph: torch.cuda.CUDAGraph,
    iterations: int,
    repeats: int,
) -> tuple[float, list[float]]:
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) * 1000 / iterations))
    return statistics.median(samples), samples


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise RuntimeError("SM100 or SM103 is required")
    if args.backend == "flashinfer_trtllm":
        os.environ.setdefault("VLLM_MXFP8_DENSE_TRTLLM_TACTIC", "-1")

    flashinfer = importlib.import_module("flashinfer")
    kernel_module = _load_flashinfer_module()
    config_module = importlib.import_module(
        "vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel"
    )
    config_api = importlib.import_module("vllm.config")
    kernel_types = {
        "flashinfer_cutlass": kernel_module.FlashInferCutlassMxfp8LinearKernel,
        "flashinfer_cutedsl": kernel_module.FlashInferCutedslMxfp8LinearKernel,
        "flashinfer_trtllm": kernel_module.FlashInferTrtllmMxfp8LinearKernel,
    }
    shapes = [
        (m, n, k)
        for n, k in ((1280, 8192), (8192, 1280))
        for m in (1, 8, 32, 256, 1024)
    ]

    torch.manual_seed(17)
    rows: list[dict[str, object]] = []
    for m, n, k in shapes:
        torch._dynamo.reset()
        weight_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
        weight, weight_scale = _quantize_weight(flashinfer, weight_bf16)
        layer = torch.nn.Module()
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        with config_api.set_current_vllm_config(config_api.VllmConfig()):
            kernel = kernel_types[args.backend](config_module.Mxfp8LinearLayerConfig())
        kernel.process_weights_after_loading(layer)

        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.1
        graph, static_x = _capture_graph(kernel, layer, x)
        static_x.copy_(x)
        median_us, samples_us = _measure_graph_us(
            graph,
            args.iterations,
            args.repeats,
        )
        tflops = 2 * m * n * k / median_us / 1e6
        row = {
            "backend": args.backend,
            "m": m,
            "n": n,
            "k": k,
            "median_us": median_us,
            "samples_us": samples_us,
            "tflops": tflops,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "cuda_graph": True,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del graph, static_x, x, layer, weight, weight_scale, weight_bf16
        torch.cuda.empty_cache()

    payload = {
        "backend": args.backend,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
