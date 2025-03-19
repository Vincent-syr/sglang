#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <THC/THCAtomics.cuh>

#include <assert.h>
#include <algorithm>

#include "utils.h"

template<typename T, int32_t TPB, int32_t NUM_EXPERTS>
__global__
void moe_biased_grouped_topk_kernel(
    const T* scores,    // [tokens, num_experts]
    const T* e_score_correction_bias, // [num_experts]
    int32_t tokens,
    int32_t num_groups,
    int32_t topk_group,
    int32_t topk,
    float routed_scaling_factor,
    int32_t* topk_idx,    // [tokens, topk]
    T* topk_weight  // [tokens, topk]
) {
    const int32_t row_id = blockIdx.x;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t group_size = NUM_EXPERTS / num_groups;
    const int32_t elem_per_thread = group_size / WARP_SIZE;

    assert(group_size >= elem_per_thread);

    __shared__ T s_scores[3][NUM_EXPERTS];
    __shared__ T s_topk_weight[NUM_EXPERTS];
    __shared__ int32_t s_topk_idx[NUM_EXPERTS];
    __shared__ T s_group_sum_val[NUM_EXPERTS];
    __shared__ bool s_group_mask[NUM_EXPERTS];

    if (lane_id < num_groups) {
        s_group_mask[lane_id] = false;
    }

    const T* row_scores_ptr = scores + row_id * NUM_EXPERTS;

    for (int32_t i = lane_id; i < NUM_EXPERTS; i += WARP_SIZE) {
        s_scores[0][i] = row_scores_ptr[i];
        s_scores[1][i] = 1.0 / (1.0 + exp(-s_scores[0][i]));
        s_scores[2][i] = s_scores[1][i] + e_score_correction_bias[i];
    }
    __syncthreads();

    // topk 2/32
    for (int32_t group_id = 0; group_id < num_groups; ++group_id) {
        const int32_t group_start = group_id * group_size;
        const int32_t topk_group_start = group_id * 2;
        T* group_scores_ptr = s_scores[2] + group_start;

        T group_sum_val = 0;
        #pragma unroll
        for (int32_t k_idx = 0; k_idx < 2; ++k_idx) {
            int32_t thread_start = group_start + lane_id * elem_per_thread;
            T* s_thread_scores_ptr = s_scores[2] + thread_start;
            T local_max = s_thread_scores_ptr[0];
            int32_t local_idx = thread_start;

            for (int32_t i=1; i < elem_per_thread; ++i) {
                if (s_thread_scores_ptr[i] > local_max) {
                    local_max = s_thread_scores_ptr[i];
                    local_idx = thread_start + i;
                }
            }
            // warp max
            T reducing_max = local_max;
            int32_t reducing_idx = local_idx;

            for (int32_t mask = group_size / 2; mask >= 1; mask >>= 1) {
                T receiving_max = __shfl_xor_sync(0xffffffff, reducing_max, mask, WARP_SIZE);
                int32_t receiving_idx = __shfl_xor_sync(0xffffffff, reducing_idx, mask, WARP_SIZE);
                if (reducing_max < receiving_max || (reducing_max == receiving_max && receiving_idx < reducing_idx)) {
                    reducing_max = receiving_max;
                    reducing_idx = receiving_idx;
                }
            }
            group_sum_val += reducing_max;

            local_idx = reducing_idx - thread_start;
            for (int32_t i = 0; i < elem_per_thread; ++i) {
                if (local_idx == i) {
                    s_thread_scores_ptr[i] = -NumericLimits<T>::max();
                }
            }

            if (lane_id == 0) {
                s_topk_idx[topk_group_start + k_idx] = reducing_idx;
                s_topk_weight[topk_group_start + k_idx] = reducing_max;
            }
            __syncthreads();
        }

        if (lane_id == 0) {
            s_group_sum_val[group_id] = group_sum_val;
            for (int32_t k_idx = 0; k_idx < 2; ++k_idx) {
                int32_t idx = s_topk_idx[topk_group_start + k_idx];
                s_scores[2][idx] = s_topk_weight[topk_group_start + k_idx];
            }
        }
        __syncthreads();
    }

    // topk 4/8,
    for (int32_t k_idx = 0; k_idx < topk_group; ++k_idx) {
        unsigned int active_mask = __ballot_sync(0xffffffff, lane_id < num_groups);

        if (lane_id < num_groups) {
            int32_t reducing_idx = lane_id;
            T reducing_max = s_group_sum_val[lane_id];
            for (int32_t mask = num_groups / 2; mask >= 1; mask >>= 1) {
                T receiving_max = __shfl_xor_sync(active_mask, reducing_max, mask, WARP_SIZE);
                int32_t receiving_idx = __shfl_xor_sync(active_mask, reducing_idx, mask, WARP_SIZE);
                if (reducing_max < receiving_max || (reducing_max == receiving_max && receiving_idx < reducing_idx)) {
                    reducing_max = receiving_max;
                    reducing_idx = receiving_idx;
                }
            }
            if (reducing_idx == lane_id) {
                s_group_sum_val[lane_id] = -NumericLimits<T>::max();
            }
    
            if (lane_id == 0) {
                s_group_mask[reducing_idx] = true;
            }
        }
        __syncthreads();
    }

    // 4. topk 8/8*32
    T row_weight_sum = T(1e-20);
    for (int32_t k_idx = 0; k_idx < topk; ++k_idx) {      
        T row_topk_max = -NumericLimits<T>::max();
        int32_t row_topk_idx = -1;

        for (int32_t group_id = 0; group_id < num_groups; ++group_id) {
            if (!s_group_mask[group_id]) {
                continue;
            }
            const int32_t group_start = group_id * group_size;
            
            T* group_scores_ptr = s_scores[2] + group_start;
            int32_t thread_start = group_start + lane_id * elem_per_thread;
            T* s_thread_scores_ptr = s_scores[2] + thread_start;

            T reducing_max = s_thread_scores_ptr[0];
            int32_t reducing_idx = thread_start;
            // warp max
            for (int32_t mask = group_size / 2; mask >= 1; mask >>= 1) {
                T receiving_max = __shfl_xor_sync(0xffffffff, reducing_max, mask, WARP_SIZE);
                int32_t receiving_idx = __shfl_xor_sync(0xffffffff, reducing_idx, mask, WARP_SIZE);
                if (reducing_max < receiving_max || (reducing_max == receiving_max && receiving_idx < reducing_idx)) {
                    reducing_max = receiving_max;
                    reducing_idx = receiving_idx;
                }
            }
            if (reducing_max > row_topk_max) {
                row_topk_max = reducing_max;
                row_topk_idx = reducing_idx;
            }
        }
        if (lane_id == 0) {
            s_topk_idx[k_idx] = row_topk_idx;
            s_topk_weight[k_idx] = s_scores[1][row_topk_idx];
            s_scores[2][row_topk_idx] = -NumericLimits<T>::max();
            row_weight_sum += s_topk_weight[k_idx];
        }
        __syncthreads();
    }

    // renormalize, broadcast
    row_weight_sum = __shfl_sync(0xffffffff, row_weight_sum, 0, WARP_SIZE);
    if (lane_id < topk) {
        s_topk_weight[lane_id] = s_topk_weight[lane_id] / row_weight_sum * routed_scaling_factor;
        int32_t row_topk_offset = row_id * topk;
        topk_weight[row_topk_offset + lane_id] = s_topk_weight[lane_id];
        topk_idx[row_topk_offset + lane_id] = s_topk_idx[lane_id];
    }
}

void moe_biased_grouped_topk(
    const torch::Tensor& scores,    // [tokens, num_experts]
    const torch::Tensor& e_score_correction_bias, // [num_experts]
    int64_t num_groups,
    int64_t topk_group,
    int64_t topk,
    double routed_scaling_factor,
    torch::Tensor& topk_idx,    // [tokens, topk]
    torch::Tensor& topk_weight  // [tokens, topk]
) {
    const int64_t num_experts = scores.size(-1);
    const int64_t tokens = scores.numel() / num_experts;
    assert(num_experts % num_groups == 0);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int TPB = 32;
    const int BPG = static_cast<int32_t>(tokens);

    TORCH_CHECK(scores.scalar_type() == at::ScalarType::Float, "scores must be an Float tensor");
    TORCH_CHECK(e_score_correction_bias.scalar_type() == at::ScalarType::Float, "e_score_correction_bias must be an Float tensor");
    TORCH_CHECK(topk_weight.scalar_type() == at::ScalarType::Float, "topk_weight must be an Float tensor");

    using T = float;
    switch (num_experts) {
        case 256: {
            moe_biased_grouped_topk_kernel<T, TPB, 256><<<BPG, TPB, 0, stream>>>(
                scores.data_ptr<T>(), 
                e_score_correction_bias.data_ptr<T>(), 
                static_cast<int32_t>(tokens), 
                static_cast<int32_t>(num_groups), 
                static_cast<int32_t>(topk_group), 
                static_cast<int32_t>(topk), 
                static_cast<float>(routed_scaling_factor), 
                topk_idx.data_ptr<int32_t>(), 
                topk_weight.data_ptr<T>()
            );
            break;
        }
        default: {
            printf("num_experts: %ld, not supported\n", num_experts);
            assert(false);
        }
    }
}