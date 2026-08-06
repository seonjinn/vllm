# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    convert_moe_weights_to_flashinfer_trtllm_block_layout,
    rotate_weights_for_fi_trtllm_fp8_per_tensor_moe,
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import input_to_float8
from vllm.model_executor.models.llama4 import Llama4MoE
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

try:
    from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "flashinfer not supported for vLLM on ROCm", allow_module_level=True
        )

if not has_flashinfer_cutlass_fused_moe() or not current_platform.has_device_capability(
    90
):
    pytest.skip(
        "Supported for sm >= 90",
        allow_module_level=True,
    )

NUM_EXPERTS = [16]
TOP_KS = [1]

MNK_FACTORS = [
    (256, 8192, 5120),
    (127, 4096, 5120),
    (10, 8192, 5120),
    (10, 4096, 5120),
    (1, 8192, 5120),
    (1, 4096, 5120),
]

vllm_config = VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))


def _convert_moe_weights_to_trtllm_block_layout_reference(
    cache_permute_indices: dict[torch.Size, torch.Tensor],
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    is_gated_act_gemm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the v0.25.1 expert-by-expert BF16 conversion."""
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    block_k = 128
    num_experts = w13_weight.shape[0]

    def copy_permuted_expert_to_block_layout(
        out: torch.Tensor,
        expert_uint8: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> None:
        expert_blocks = expert_uint8.view(
            expert_uint8.shape[0], out.shape[0], block_k
        ).permute(1, 0, 2)
        torch.index_select(
            expert_blocks,
            1,
            source_indices.to(expert_uint8.device),
            out=out,
        )

    w13_rows, w13_cols = w13_weight[0].view(torch.uint8).shape
    w2_rows, w2_cols = w2_weight[0].view(torch.uint8).shape
    w13_out = torch.empty(
        (num_experts, w13_cols // block_k, w13_rows, block_k),
        dtype=torch.uint8,
        device=w13_weight.device,
    )
    w2_out = torch.empty(
        (num_experts, w2_cols // block_k, w2_rows, block_k),
        dtype=torch.uint8,
        device=w2_weight.device,
    )

    for expert_idx in range(num_experts):
        w13_expert = w13_weight[expert_idx].view(torch.uint8)
        w13_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            w13_expert,
            epilogue_tile_m,
            is_gated_act_gemm=is_gated_act_gemm,
        )
        if is_gated_act_gemm:
            w13_indices = (w13_indices + w13_rows // 2) % w13_rows
        copy_permuted_expert_to_block_layout(
            w13_out[expert_idx], w13_expert, w13_indices
        )

        w2_expert = w2_weight[expert_idx].view(torch.uint8)
        w2_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            w2_expert,
            epilogue_tile_m,
        )
        copy_permuted_expert_to_block_layout(
            w2_out[expert_idx], w2_expert, w2_indices
        )

    return w13_out.view(torch.bfloat16), w2_out.view(torch.bfloat16)


def quant_fp8_per_tensor_batches(a):
    num_batches = a.size(0)
    a_quant = []
    a_scales = []

    for i in range(num_batches):
        a_fp8, a_global_sf = input_to_float8(a[i])
        if a_global_sf.numel() == 1:
            a_global_sf = a_global_sf.view(1, 1)
        a_quant.append(a_fp8)
        a_scales.append(a_global_sf)

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


def check_accuracy(ref_output, actual_output, atol=0.1, rtol=0.85, percent=0.925):
    close = torch.isclose(ref_output, actual_output, atol=atol, rtol=rtol)
    match_ratio = close.float().mean()
    assert match_ratio >= percent, (
        f"Match ratio {match_ratio:.4f} is below the threshold {percent:.4f}"
    )

    mismatch_percent = 1.0 - match_ratio.item()
    assert mismatch_percent <= 1 - percent, (
        f"Mismatch percentage {mismatch_percent:.4f} is above the threshold "
        f"{1 - percent:.4f}"
    )


@dataclass
class TestData:
    hidden_states: torch.Tensor
    w13_quantized: torch.Tensor
    w2_quantized: torch.Tensor
    a1_scale: torch.Tensor
    a2_scale: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    layer: torch.nn.Module

    @staticmethod
    def make_moe_tensors_8bit(
        m: int,
        k: int,
        n: int,
        e: int,
        is_trtllm: bool,
        activation: MoEActivation = MoEActivation.SILU,
        topk: int = 1,
    ) -> "TestData":
        is_gated = activation.is_gated

        hidden_states = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
        w13 = (
            torch.randn(
                (e, (2 * n) if is_gated else n, k), device="cuda", dtype=torch.bfloat16
            )
            / 10
        )
        w2 = torch.randn((e, k, n), device="cuda", dtype=torch.bfloat16) / 10

        # Scale to fp8
        _, a1_scale = input_to_float8(hidden_states)
        a2_scale = torch.scalar_tensor(1.0).to(device="cuda").to(dtype=torch.float32)
        w13_quantized, w13_weight_scale = quant_fp8_per_tensor_batches(w13)
        w2_quantized, w2_weight_scale = quant_fp8_per_tensor_batches(w2)

        layer = torch.nn.Module()
        layer.orig_dtype = torch.bfloat16
        layer.w13_weight = w13_quantized.clone()
        layer.w2_weight = w2_quantized.clone()
        layer.w13_input_scale = a1_scale
        layer.w2_input_scale = a2_scale
        layer.w13_weight_scale = w13_weight_scale
        layer.w2_weight_scale = w2_weight_scale
        layer.activation = activation
        # Setup dummy config.
        layer.moe_parallel_config = mk.FusedMoEParallelConfig.make_no_parallel()

        # flashinfer expects swapped rows for w13
        if is_gated:
            layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
        if is_trtllm:
            rotate_weights_for_fi_trtllm_fp8_per_tensor_moe(
                layer.w13_weight, layer.w2_weight, is_gated
            )
        layer.custom_routing_function = Llama4MoE.custom_routing_function
        layer.routing_method_type = RoutingMethodType.Llama4
        layer.renormalize = False
        layer.intermediate_size_per_partition = n
        layer.ep_rank = 0
        layer.local_num_experts = e

        layer.moe = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            moe_parallel_config=layer.moe_parallel_config,
            in_dtype=hidden_states.dtype,
            routing_method=layer.routing_method_type,
            activation=activation,
            device=w13_quantized.device,
            max_num_tokens=next_power_of_2(m),
        )

        return TestData(
            hidden_states=hidden_states,
            w13_quantized=w13_quantized,
            w2_quantized=w2_quantized,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
            layer=layer,
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
def test_flashinfer_per_tensor_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    monkeypatch,
):
    if not current_platform.has_device_capability(100):
        pytest.skip("Test is only supported for sm >= 100")
    set_random_seed(7)
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(
            m, k, n, e, is_trtllm=True, activation=activation
        )

        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = Llama4MoE.custom_routing_function(
            hidden_states=td.hidden_states,
            gating_output=score,
            topk=topk,
            renormalize=False,
        )

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=td.w13_weight_scale,
            w2_scale=td.w2_weight_scale,
            a1_scale=td.a1_scale,
            a2_scale=td.a2_scale,
            per_act_token_quant=False,
        )

        output = fused_experts(
            td.hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
            quant_config=quant_config,
        )

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=td.layer.moe,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=True,
            ),
            TrtLlmFp8ExpertsMonolithic(
                moe_config=td.layer.moe,
                quant_config=quant_config,
            ),
        )

        flashinfer_output = kernel.apply_monolithic(
            hidden_states=td.hidden_states,
            w1=td.layer.w13_weight,
            w2=td.layer.w2_weight,
            router_logits=score,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
            routed_scaling_factor=1.0,
        )

        check_accuracy(
            ref_output=output,
            actual_output=flashinfer_output,
            atol=0.1,
            rtol=0.85,
            percent=0.925,
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
def test_flashinfer_cutlass_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    monkeypatch,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(
            m, k, n, e, is_trtllm=False, activation=activation
        )

        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = Llama4MoE.custom_routing_function(
            hidden_states=td.hidden_states,
            gating_output=score,
            topk=topk,
            renormalize=False,
        )

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=td.w13_weight_scale,
            g1_alphas=(td.w13_weight_scale * td.a1_scale).squeeze(),
            w2_scale=td.w2_weight_scale,
            g2_alphas=(td.w2_weight_scale * td.a2_scale).squeeze(),
            a1_scale=td.a1_scale,
            a1_gscale=td.a1_scale,
            a2_scale=td.a2_scale,
            a2_gscale=1.0 / td.a2_scale,
            per_act_token_quant=False,
        )

        output = fused_experts(
            td.hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
            quant_config=quant_config,
        )

        td.layer.dp_size = 1

        def get_fused_moe_quant_config(n: torch.nn.Module) -> FusedMoEQuantConfig:
            return quant_config

        td.layer.get_fused_moe_quant_config = get_fused_moe_quant_config
        td.layer.quant_method = td.layer

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=torch.bfloat16,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        kernel = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            FlashInferExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
        )

        flashinfer_cutlass_output = kernel.apply(
            td.hidden_states,
            td.layer.w13_weight,
            td.layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
        )

        check_accuracy(
            ref_output=output,
            actual_output=flashinfer_cutlass_output,
            atol=0.1,
            rtol=0.85,
            percent=0.925,
        )


@pytest.mark.parametrize(
    "num_experts,intermediate,hidden",
    [
        (8, 2048, 1536),
        (64, 4096, 4096),
    ],
)
def test_convert_moe_weights_to_flashinfer_trtllm_block_layout(
    num_experts, intermediate, hidden
):
    w13 = torch.randn(
        (num_experts, 2 * intermediate, hidden), dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.randn(
        (num_experts, hidden, intermediate), dtype=torch.bfloat16, device="cuda"
    )

    cache: dict[torch.Size, torch.Tensor] = {}
    w13_converted, w2_converted = convert_moe_weights_to_flashinfer_trtllm_block_layout(
        cache, w13, w2
    )

    assert w13_converted.ndim == 4, (
        f"Expected 4D tensor, got shape {w13_converted.shape}"
    )
    assert w2_converted.ndim == 4, f"Expected 4D tensor, got shape {w2_converted.shape}"

    assert w13_converted.numel() == w13.numel(), "W13 element count should be preserved"
    assert w2_converted.numel() == w2.numel(), "W2 element count should be preserved"

    assert w13_converted.dtype == torch.bfloat16
    assert w2_converted.dtype == torch.bfloat16

    assert w13_converted.shape[0] == num_experts
    assert w2_converted.shape[0] == num_experts


@pytest.mark.parametrize("is_gated_act_gemm", [True, False])
def test_convert_moe_weights_to_flashinfer_trtllm_block_layout_batched(
    is_gated_act_gemm: bool, monkeypatch
):
    num_experts = 4
    intermediate = 128
    hidden = 128
    w13 = torch.randn(
        (num_experts, 2 * intermediate, hidden), dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.randn(
        (num_experts, hidden, intermediate), dtype=torch.bfloat16, device="cuda"
    )

    expected_w13, expected_w2 = _convert_moe_weights_to_trtllm_block_layout_reference(
        {}, w13, w2, is_gated_act_gemm
    )

    original_index_select = torch.index_select
    index_select_dims = []

    def record_index_select(*args, **kwargs):
        index_select_dims.append(args[1])
        return original_index_select(*args, **kwargs)

    monkeypatch.setattr(torch, "index_select", record_index_select)
    actual_w13, actual_w2 = convert_moe_weights_to_flashinfer_trtllm_block_layout(
        {}, w13, w2, is_gated_act_gemm
    )

    assert torch.equal(actual_w13, expected_w13)
    assert torch.equal(actual_w2, expected_w2)
    assert index_select_dims == [2, 2]
