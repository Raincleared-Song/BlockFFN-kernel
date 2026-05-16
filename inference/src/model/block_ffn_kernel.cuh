#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define MAX_NUM_BLOCKS 180

template <typename T>
__global__ void sparse_up_kernel(
    int hidden_size,
    int tile_size,
    const int* nnz,
    const int* nz_idx,
    const float4* input,
    const float4* weights,
    T* output) {
    
    using T2 = typename TypeTraits<T>::half2;
    
    int tid = threadIdx.x;
    if (blockIdx.x >= nnz[0]) return;

    int row = nz_idx[blockIdx.x] * gridDim.y * tile_size + blockIdx.y * tile_size + threadIdx.y;
    int row_offset = row * hidden_size;
    float4 partial_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = tid; j < hidden_size; j += blockDim.x) {
        float4 a = input[j], b = weights[row_offset + j];
        T2 prodx = __hmul2(*reinterpret_cast<T2*>(&a.x), *reinterpret_cast<T2*>(&b.x));
        T2 prody = __hmul2(*reinterpret_cast<T2*>(&a.y), *reinterpret_cast<T2*>(&b.y));
        T2 prodz = __hmul2(*reinterpret_cast<T2*>(&a.z), *reinterpret_cast<T2*>(&b.z));
        T2 prodw = __hmul2(*reinterpret_cast<T2*>(&a.w), *reinterpret_cast<T2*>(&b.w));
        partial_sum.x += float(prodx.x) + float(prodx.y);
        partial_sum.y += float(prody.x) + float(prody.y);
        partial_sum.z += float(prodz.x) + float(prodz.y);
        partial_sum.w += float(prodw.x) + float(prodw.y);
    }
    float sum = partial_sum.x + partial_sum.y + partial_sum.z + partial_sum.w;
    // sum all
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (tid == 0) {
        output[row] = T(sum);
    }
}

template <typename T>
__global__ void sparse_down_kernel(
    int block_size,
    int intermediate_size,
    int tile_size,
    const int* nnz,
    const int* nz_idx,
    const float4* input,
    const float4* weights,
    T* output) {

    __shared__ float warp_sum[32];

    using T2 = typename TypeTraits<T>::half2;
    
    int row = blockIdx.x * tile_size + threadIdx.z;
    int col = threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num = nnz[0];
    float4 partial_sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = threadIdx.y; i < num; i += blockDim.y) {
        int offset = nz_idx[i] * block_size;
        float4 a = input[offset + col], b = weights[row * intermediate_size + offset + col];
        T2 prodx = __hmul2(*reinterpret_cast<T2*>(&a.x), *reinterpret_cast<T2*>(&b.x));
        T2 prody = __hmul2(*reinterpret_cast<T2*>(&a.y), *reinterpret_cast<T2*>(&b.y));
        T2 prodz = __hmul2(*reinterpret_cast<T2*>(&a.z), *reinterpret_cast<T2*>(&b.z));
        T2 prodw = __hmul2(*reinterpret_cast<T2*>(&a.w), *reinterpret_cast<T2*>(&b.w));
        partial_sum.x += float(prodx.x) + float(prodx.y);
        partial_sum.y += float(prody.x) + float(prody.y);
        partial_sum.z += float(prodz.x) + float(prodz.y);
        partial_sum.w += float(prodw.x) + float(prodw.y);
    }
    float sum = partial_sum.x + partial_sum.y + partial_sum.z + partial_sum.w;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (tid % 32 == 0) warp_sum[threadIdx.z * 4 + tid / 32] = sum;
    __syncthreads();
    if (tid < 4) {
        sum = warp_sum[threadIdx.z * 4 + tid];
        sum += __shfl_down_sync(0x0000000f, sum, 2);
        sum += __shfl_down_sync(0x0000000f, sum, 1);
    }
    if (tid == 0) {
        output[row] = T(sum);
    }
}

template <typename T>
__global__ void nonzero_kernel(int num_blocks, const T* input, int* nnz, T* nz_val, int* nz_idx) {
    __shared__ uint64_t s_nnz_mask[96];
    __shared__ T s_input[256];

    int col = threadIdx.x;
    int col_g = threadIdx.x / 64;
    int col_v = threadIdx.x % 64;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    uint64_t nnz_mask[3] = {0, 0, 0};
    if (col < num_blocks) {
        T val = input[col];
        s_input[col] = val;
        nnz_mask[col_g] |= (val > T(0)) ? (1ULL << col_v) : 0;
    }
    for (int i = 0; i < 3; i++) {
      nnz_mask[i] |= __shfl_down_sync(0xffffffff, nnz_mask[i], 16);
      nnz_mask[i] |= __shfl_down_sync(0xffffffff, nnz_mask[i], 8);
      nnz_mask[i] |= __shfl_down_sync(0xffffffff, nnz_mask[i], 4);
      nnz_mask[i] |= __shfl_down_sync(0xffffffff, nnz_mask[i], 2);
      nnz_mask[i] |= __shfl_down_sync(0xffffffff, nnz_mask[i], 1);
    }
    if (lane_id == 0) {
        s_nnz_mask[warp_id] = nnz_mask[0];
        s_nnz_mask[32 + warp_id] = nnz_mask[1];
        s_nnz_mask[64 + warp_id] = nnz_mask[2];
    }
    __syncthreads();
    if (warp_id <= 2) {
        nnz_mask[warp_id] = (lane_id < blockDim.x / 32) ? s_nnz_mask[lane_id + warp_id * 32] : 0;
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 16);
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 8);
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 4);
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 2);
        nnz_mask[warp_id] |= __shfl_down_sync(0xffffffff, nnz_mask[warp_id], 1);
        if (lane_id == 0) {
            s_nnz_mask[warp_id] = nnz_mask[warp_id];
        }
    }
    __syncthreads();
    nnz_mask[0] = s_nnz_mask[0];
    nnz_mask[1] = s_nnz_mask[1];
    nnz_mask[2] = s_nnz_mask[2];
    int nnz_offset[2];
    nnz_offset[0] = __popcll(nnz_mask[0]);
    nnz_offset[1] = __popcll(nnz_mask[1]);
    if (col == 0) {
        nnz[0] = nnz_offset[0] + nnz_offset[1] + __popcll(nnz_mask[2]);
    }
    if (col < num_blocks && (nnz_mask[col_g] >> col_v & 1)) {
        int pos = (col_g > 0 ? nnz_offset[0] : 0) + (col_g > 1 ? nnz_offset[1] : 0) + __popcll(nnz_mask[col_g] & ((1ULL << (col_v)) - 1));
        nz_idx[pos] = col;
        nz_val[pos] = s_input[col];
    }
}

template <typename T, typename T2>
__global__ void sparse_norm_silu_kernel(
    int dim,
    const int* nnz,
    const int* nz_idx,
    const T* nz_val,
    const T* projected_mean,
    const T2* norm_weight,
    float norm_eps,
    T2* input  // in-place modification
) {
    int block_id = blockIdx.x;
    if (block_id >= nnz[0]) return;
    
    int block_idx = nz_idx[block_id];
    int tid = threadIdx.x;
    
    __shared__ T2 s_input[1024];
    __shared__ float shared_sum;
    __shared__ float warp_sum[4];
    
    T mean = projected_mean[0];
    
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        T2 val = input[block_idx * dim + i];
        float v1 = float(val.x) - float(mean);
        float v2 = float(val.y) - float(mean);
        s_input[i] = T2(T(v1), T(v2));
        sum1 += v1 * v1;
        sum2 += v2 * v2;
    }
    
    float sum = sum1 + sum2;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    
    if (tid % 32 == 0) warp_sum[tid / 32] = sum;
    __syncthreads();
    
    if (tid < 4) {
        sum = warp_sum[tid];
        sum += __shfl_down_sync(0x0000000f, sum, 2);
        sum += __shfl_down_sync(0x0000000f, sum, 1);
    }
    
    if (tid == 0) {
        shared_sum = rsqrtf(sum / (2 * dim) + norm_eps);
    }
    __syncthreads();
    
    float scale = shared_sum;
    float router_val = nz_val[block_id];
    
    // Apply normalization, weight, and SiLU
    for (int i = tid; i < dim; i += blockDim.x) {
        T2 inp = s_input[i];
        T2 w = norm_weight[i];
        
        float v1 = scale * float(inp.x) * float(w.x);
        float v2 = scale * float(inp.y) * float(w.y);
        
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        v1 = v1 / (1.0f + expf(-v1)) * router_val;
        v2 = v2 / (1.0f + expf(-v2)) * router_val;
        
        input[block_idx * dim + i] = T2(T(v1), T(v2));
    }
}

template <typename T>
void sparse_up(const Stream& stream, int num_blocks, int block_size, int hidden_size, const int* nnz, const int* nz_idx, const T* input, const T* weight, T* output) {
  constexpr int tile_size = 8;
  hidden_size /= (16 / sizeof(T));
  sparse_up_kernel<<<dim3(MAX_NUM_BLOCKS, block_size/tile_size), dim3(32, tile_size), 0, stream.stream>>>(hidden_size, tile_size, nnz, nz_idx, (float4*)input, (float4*)weight, output);
}

template <typename T>
void sparse_down(const Stream& stream, int num_blocks, int block_size, int hidden_size, const int* nnz, const int* nz_idx, const T* input, const T* weight, T* output) {
  constexpr int tile_size = 8;
  block_size /= (16 / sizeof(T));
  sparse_down_kernel<<<hidden_size/tile_size, dim3(block_size, 1024/tile_size/block_size, tile_size), 0, stream.stream>>>(block_size, num_blocks * block_size, tile_size, nnz, nz_idx, (float4*)input, (float4*)weight, output);
}

template <typename T>
void sparse_norm_silu(const Stream& stream, int num_blocks, int block_size, const int* nnz, const int* nz_idx, const T* nz_val, const T* projected_mean, const T* norm_weight, float norm_eps, T* input) {
    using T2 = typename TypeTraits<T>::half2;
    sparse_norm_silu_kernel<T, T2><<<MAX_NUM_BLOCKS, 128, 0, stream.stream>>>(
        block_size/2, nnz, nz_idx, nz_val, projected_mean, (T2*)norm_weight, norm_eps, (T2*)input
    );
}

template <typename T>
void nonzero(const Stream& stream, int num_blocks, const T* input, int* nnz, T* nz_val, int* nz_idx) {
  if (num_blocks == 180) {
    nonzero_kernel<<<1, 192, 0, stream.stream>>>(num_blocks, input, nnz, nz_val, nz_idx);
  } else {
    throw std::invalid_argument("Unsupported num_blocks " + std::to_string(num_blocks));
  }
}

