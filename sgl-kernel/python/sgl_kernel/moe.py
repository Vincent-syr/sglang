import torch


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    token_cnts_buffer,
    cumsum_buffer,
):
    torch.ops.sgl_kernel.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: float,
) -> None:
    torch.ops.sgl_kernel.topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating_output
    )

def moe_biased_grouped_topk(
    scores: torch.Tensor,  # [num_tokens, num_experts]
    correction_bias: torch.Tensor,  # [num_experts]
    num_groups: int,
    topk_group: int,
    topk: bool,
    routed_scaling_factor: float,
    topk_idx: torch.Tensor,  # [num_tokens, topk]
    topk_weight: torch.Tensor,  # [num_tokens, topk]
):
    assert correction_bias.shape[0] == scores.shape[-1], f"{correction_bias.shape[0]} != {scores.shape[-1]}"
    assert topk_idx.shape == topk_weight.shape, f"{topk_idx.shape} != {topk_weight.shape}"
    assert topk_idx.numel() / topk_idx.shape[-1] == scores.numel() / scores.shape[-1], f"{topk_idx.numel() / topk_idx.shape[-1]} != {scores.numel() / scores.shape[-1]}"

    torch.ops.sgl_kernel.moe_biased_grouped_topk(scores, correction_bias, num_groups, topk_group, topk, routed_scaling_factor, topk_idx, topk_weight)
