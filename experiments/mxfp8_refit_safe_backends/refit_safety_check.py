from __future__ import annotations

import sys
import types
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter

import vllm.model_executor.kernels.linear.mxfp8.flashinfer as flashinfer_module
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    FlashInferTrtllmMxfp8LinearKernel,
)


def _make_layer(n: int, k: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = Parameter(torch.arange(n * k).reshape(n, k), requires_grad=False)
    layer.weight_scale = Parameter(torch.ones((n, k // 32)), requires_grad=False)
    return layer


def _check_refit(
    kernel_type: type,
    process: Callable[[torch.nn.Module], None],
    layer: torch.nn.Module,
    prepared_names: tuple[str, ...],
) -> None:
    assert kernel_type.preserves_checkpoint_weight_scale_for_refit is True
    weight = layer.weight
    weight_scale = layer.weight_scale

    process(layer)
    prepared = {name: getattr(layer, name) for name in prepared_names}
    layer.weight.data.add_(1)
    layer.weight_scale.data.mul_(2)
    process(layer)

    assert layer.weight is weight
    assert layer.weight_scale is weight_scale
    for name, value in prepared.items():
        assert getattr(layer, name) is value


def main() -> None:
    original_swizzle = flashinfer_module.swizzle_mxfp8_scale
    original_exact = flashinfer_module.prepare_mxfp8_trtllm_exact_tactic_state
    original_high_m = flashinfer_module.prepare_mxfp8_trtllm_high_m_tactic_state
    original_flashinfer = sys.modules.get("flashinfer")
    try:
        flashinfer_module.swizzle_mxfp8_scale = lambda scale, *, M, K: scale + M + K

        cutlass = object.__new__(FlashInferCutlassMxfp8LinearKernel)
        _check_refit(
            FlashInferCutlassMxfp8LinearKernel,
            cutlass.process_weights_after_loading,
            _make_layer(4, 64),
            ("weight_scale_for_apply",),
        )

        cutedsl = object.__new__(FlashInferCutedslMxfp8LinearKernel)
        _check_refit(
            FlashInferCutedslMxfp8LinearKernel,
            cutedsl.process_weights_after_loading,
            _make_layer(4, 64),
            ("weight_for_apply", "weight_scale_for_apply"),
        )

        sys.modules["flashinfer"] = types.SimpleNamespace(
            shuffle_matrix_a=lambda weight, _: weight + 3,
            shuffle_matrix_sf_a=lambda scale, *_args, **_kwargs: scale + 5,
        )
        flashinfer_module.prepare_mxfp8_trtllm_exact_tactic_state = lambda _: None
        flashinfer_module.prepare_mxfp8_trtllm_high_m_tactic_state = lambda _: None
        trtllm = object.__new__(FlashInferTrtllmMxfp8LinearKernel)
        trtllm._cutedsl_fallback = None
        _check_refit(
            FlashInferTrtllmMxfp8LinearKernel,
            trtllm.process_weights_after_loading,
            _make_layer(4, 256),
            ("weight_for_apply", "weight_scale_for_apply"),
        )
    finally:
        flashinfer_module.swizzle_mxfp8_scale = original_swizzle
        flashinfer_module.prepare_mxfp8_trtllm_exact_tactic_state = original_exact
        flashinfer_module.prepare_mxfp8_trtllm_high_m_tactic_state = original_high_m
        if original_flashinfer is None:
            sys.modules.pop("flashinfer", None)
        else:
            sys.modules["flashinfer"] = original_flashinfer

    print("refit-safe MXFP8 linear backend checks passed")


if __name__ == "__main__":
    main()
