import argparse
import itertools

import torch
import triton
import torch.nn.functional as F
import triton.language as tl
from sgl_kernel import moe_biased_grouped_topk
from vllm import _custom_ops as ops

USE_RANDOM_PERM = False

def get_compiler_backend() -> str:
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return "hpu_backend"

    return "inductor"

@torch.compile(dynamic=True, backend=get_compiler_backend())
def native_moe_biased_grouped_topk(
    logits: torch.Tensor, # [tokens, n_routed_experts]
    e_score_correction_bias: torch.Tensor, # [n_routed_experts]
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float
):
    num_experts = logits.shape[-1]
    group_size = num_experts // n_group
    scores = F.sigmoid(logits)
    scores_with_bias = scores + e_score_correction_bias

    scores_shape = list(scores_with_bias.shape)
    tmp_weight, tmp_idx = torch.topk(scores_with_bias.view(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]),
        k=2,
        dim=-1,
        largest=True,
        sorted=True)
    group_scores = torch.sum(tmp_weight, dim=-1) # [tokens, n_group]
    _, group_idx = torch.topk(group_scores,
                                k=topk_group,
                                dim=-1,
                                largest=True,
                                sorted=True)

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(-1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(
        scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]).reshape(scores_shape)

    scores_with_bias = scores_with_bias.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    _, topk_idx = torch.topk(scores_with_bias,
                                k=topk,
                                dim=-1,
                                largest=True,
                                sorted=True)
    topk_weight = scores.gather(1, topk_idx)  # [tokens, top_k]
    topk_weight_sum = torch.sum(topk_weight, dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / topk_weight_sum * routed_scaling_factor
    return topk_idx, topk_weight

def calculate_diff(tokens: int):
    n_routed_experts = 256
    routed_scaling_factor = 1.0
    n_group, topk_group, topk = 8, 4, 8
    logits = torch.randn([tokens, n_routed_experts], dtype=torch.float32, device="cuda")
    e_score_correction_bias = torch.randn([n_routed_experts], dtype=torch.float32, device="cuda")
    topk_idx_native, topk_weight_native = native_moe_biased_grouped_topk(
        logits.clone(),
        e_score_correction_bias.clone(),
        n_group,
        topk_group,
        topk,
        routed_scaling_factor
    )

    topk_idx = torch.empty([tokens, topk], device=logits.device, dtype=torch.int32)
    topk_weight = torch.empty([tokens, topk], device=logits.device, dtype=torch.float32)

    moe_biased_grouped_topk(
        logits, e_score_correction_bias, n_group, topk_group, topk, routed_scaling_factor, topk_idx, topk_weight
    )
    if torch.allclose(topk_idx, topk_idx_native.type(torch.int32)) and torch.allclose(topk_weight, topk_weight_native, atol=1e-5, rtol=1e-5):
        print(f"✅ SGL and Torch implementations match")

tokens_range = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
configs = tokens_range

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["pytorch", "sgl"],
        line_names=["Pytorch", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe-biased-grouped-topk-performance",
        args={},
    )
)
def benchmark(tokens: int, provider):
    n_routed_experts = 256
    routed_scaling_factor = 1.0
    n_group, topk_group, topk = 8, 4, 8

    dtype = torch.float
    device = torch.device("cuda")
    
    logits = torch.randn([tokens, n_routed_experts], dtype=torch.float32, device="cuda")
    e_score_correction_bias = torch.randn([n_routed_experts], dtype=torch.float32, device="cuda")
    topk_idx = torch.empty([tokens, topk], device=logits.device, dtype=torch.int32)
    topk_weight = torch.empty([tokens, topk], device=logits.device, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "pytorch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: native_moe_biased_grouped_topk(
                logits, e_score_correction_bias, n_group, topk_group, topk, routed_scaling_factor
            ),
            quantiles=quantiles,
        )
    elif provider == "sgl":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_biased_grouped_topk(
                logits, e_score_correction_bias, n_group, topk_group, topk, routed_scaling_factor, topk_idx, topk_weight
            ),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    calculate_diff(tokens=1024)
    benchmark.run(print_data=True)