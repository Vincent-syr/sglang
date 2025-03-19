import itertools

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel.moe import moe_biased_grouped_topk

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

@pytest.mark.parametrize(
    "tokens, n_routed_experts, routed_scaling_factor, n_group, topk_group, topk",
    list(
        itertools.product(
            [1, 8, 9, 256, 1024, 2048, 4096, 8192], # tokens
            [256],  # n_routed_experts
            [1],  # routed_scaling_factor
            [8],  # n_group
            [4], # topk_group
            [8], # topk
        )
    ),
)

def test_moe_biased_grouped_topk_compare_implementations(        
    tokens: int,
    n_routed_experts: int,
    routed_scaling_factor: float,
    n_group: int,
    topk_group: int,
    topk: int
):
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

    assert torch.allclose(topk_idx, topk_idx_native.type(torch.int32))
    assert torch.allclose(topk_weight, topk_weight_native, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
