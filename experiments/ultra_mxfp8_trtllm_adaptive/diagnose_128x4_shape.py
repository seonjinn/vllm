# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time

import torch


def _mark(message: str) -> None:
    print(f"[{time.monotonic():.6f}] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--path",
        choices=("direct-128x4", "adaptive-op"),
        default="direct-128x4",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    layout = "adaptive" if args.path == "adaptive-op" else "128x4"
    os.environ["VLLM_MXFP8_TRTLLM_LAYOUT"] = layout
    os.environ["VLLM_MXFP8_TRTLLM_SWITCH_M"] = "256"

    from flashinfer import (
        SfLayout,
        mxfp8_quantize,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
    )
    from flashinfer.autotuner import autotune

    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        mxfp8_trtllm_linear,
    )
    from vllm.utils import flashinfer as vllm_flashinfer

    torch.accelerator.set_device_index(0)
    torch.manual_seed(0)
    m, n, k = args.m, args.n, args.k
    padded_n = ((n + 127) // 128) * 128
    _mark(f"START path={args.path} shape=({m},{n},{k}) padded_n={padded_n}")

    weight_bf16 = torch.randn((padded_n, k), dtype=torch.bfloat16, device="cuda")
    weight, weight_scale = mxfp8_quantize(
        weight_bf16,
        sf_swizzle_layout=SfLayout.layout_linear,
    )
    del weight_bf16
    weight = shuffle_matrix_a(weight, 128).reshape(padded_n, k)
    weight_scale = shuffle_matrix_sf_a(
        weight_scale.view(padded_n, k // 32),
        128,
        num_elts_per_sf=32,
    ).reshape(-1)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    torch.accelerator.synchronize()
    _mark("WEIGHT_READY")

    if args.path == "adaptive-op":
        small_x = torch.randn((256, k), dtype=torch.bfloat16, device="cuda")
        with autotune(tune_mode=True):
            _mark("ADAPTIVE_SMALL_TUNE_BEGIN")
            small_output = mxfp8_trtllm_linear(small_x, weight, weight_scale, n)
            torch.accelerator.synchronize()
            _mark(f"ADAPTIVE_SMALL_TUNE_DONE output={tuple(small_output.shape)}")

            _mark("ADAPTIVE_LARGE_TUNE_BEGIN")
            output = mxfp8_trtllm_linear(x, weight, weight_scale, n)
            torch.accelerator.synchronize()
            _mark(f"ADAPTIVE_LARGE_TUNE_DONE output={tuple(output.shape)}")

        _mark("ADAPTIVE_SELECTED_REPLAY_BEGIN")
        output = mxfp8_trtllm_linear(x, weight, weight_scale, n)
        torch.accelerator.synchronize()
        _mark(f"ADAPTIVE_SELECTED_REPLAY_DONE output={tuple(output.shape)}")

        static_x = torch.randn_like(x)
        _mark("ADAPTIVE_CUDA_GRAPH_CAPTURE_BEGIN")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = mxfp8_trtllm_linear(static_x, weight, weight_scale, n)
        _mark("ADAPTIVE_CUDA_GRAPH_CAPTURE_DONE")
        graph.replay()
        torch.accelerator.synchronize()
        _mark(f"ADAPTIVE_CUDA_GRAPH_REPLAY_DONE output={tuple(graph_output.shape)}")
        return

    with autotune(tune_mode=True):
        _mark("TUNE_QUANTIZE_BEGIN")
        x_mxfp8, x_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_128x4(x)
        torch.accelerator.synchronize()
        _mark("TUNE_QUANTIZE_DONE")

        _mark("TUNE_GEMM_BEGIN")
        output = vllm_flashinfer.mm_mxfp8(
            x_mxfp8,
            weight.t(),
            x_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
            backend="trtllm",
            use_8x4_sf_layout=False,
        )
        torch.accelerator.synchronize()
        _mark(f"TUNE_GEMM_DONE output={tuple(output.shape)}")

    _mark("SELECTED_REPLAY_BEGIN")
    x_mxfp8, x_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_128x4(x)
    output = vllm_flashinfer.mm_mxfp8(
        x_mxfp8,
        weight.t(),
        x_scale,
        weight_scale,
        out_dtype=torch.bfloat16,
        backend="trtllm",
        use_8x4_sf_layout=False,
    )
    torch.accelerator.synchronize()
    _mark(f"SELECTED_REPLAY_DONE output={tuple(output.shape)}")

    static_x = torch.randn_like(x)
    _mark("CUDA_GRAPH_CAPTURE_BEGIN")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_x, graph_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_128x4(static_x)
        graph_output = vllm_flashinfer.mm_mxfp8(
            graph_x,
            weight.t(),
            graph_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
            backend="trtllm",
            use_8x4_sf_layout=False,
        )
    _mark("CUDA_GRAPH_CAPTURE_DONE")
    graph.replay()
    torch.accelerator.synchronize()
    _mark(f"CUDA_GRAPH_REPLAY_DONE output={tuple(graph_output.shape)}")


if __name__ == "__main__":
    main()
