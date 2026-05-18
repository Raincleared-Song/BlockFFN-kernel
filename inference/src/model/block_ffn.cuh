#pragma once

#include "ffn.cuh"
#include "block_ffn_kernel.cuh"
#include "topk.cuh"

namespace {
template <typename T>
__global__ void topk_softmax_scatter_kernel(int dim, int top, const T* topk_val, const int* topk_pos, T* output) {
    int row = blockIdx.x;
    int row_dim = row * dim;
    int row_top = row * top;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[row_dim + i] = T(0);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -float(TypeTraits<T>::inf());
        for (int i = 0; i < top; i++) {
            mx = fmaxf(mx, float(topk_val[row_top + i]));
        }

        float sum = 0.0f;
        for (int i = 0; i < top; i++) {
            sum += expf(float(topk_val[row_top + i]) - mx);
        }

        for (int i = 0; i < top; i++) {
            int pos = topk_pos[row_top + i];
            float score = expf(float(topk_val[row_top + i]) - mx) / sum;
            output[row_dim + pos] = T(score);
        }
    }
}

template <typename T>
void topk_softmax_scatter(const Stream& stream, int num_tokens, int dim, int top, const T* topk_val, const int* topk_pos, T* output) {
    topk_softmax_scatter_kernel<T><<<num_tokens, 256, 0, stream.stream>>>(dim, top, topk_val, topk_pos, output);
}
}

template <typename T>
struct Router {
    int hidden_size;
    int num_blocks;
    int router_topk;

    Linear<T> *proj;
    SimpleLayerNorm<T> *norm;
    functions::TopK<T> *topk_func;
    T* output;

    Router(int hidden_size, int num_blocks, float rms_norm_eps, int router_topk = 0) {
        this->hidden_size = hidden_size;
        this->num_blocks = num_blocks;
        this->router_topk = router_topk;
        if (this->router_topk < 0 || this->router_topk > this->num_blocks || this->router_topk > 64) {
            throw std::invalid_argument("Unsupported router_topk " + std::to_string(this->router_topk));
        }

        this->proj = new Linear<T>(hidden_size, num_blocks);
        this->norm = new SimpleLayerNorm<T>(num_blocks);
        this->topk_func = this->router_topk > 0 ? new functions::TopK<T>(num_blocks, this->router_topk) : nullptr;
    }

    void init_weight_ptr(Memory* memory) {
        this->proj->init_weight_ptr(memory);
        this->norm->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t proj_end = this->proj->init_output_ptr(memory, num_tokens, offset);
        if (this->topk_func != nullptr) {
            proj_end = this->topk_func->init_output_ptr(memory, num_tokens, proj_end);
        }
        this->output = this->proj->output;
        // norm inplace
        return proj_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("moe_router") != std::string::npos) {
            this->proj->load_to_storage(name, ptr);
        } else if (name.find("router_norm") != std::string::npos) {
            this->norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input) {
        this->proj->prefill(stream, num_tokens, input);
        if (this->topk_func != nullptr) {
            this->topk_func->prefill(stream, num_tokens, this->proj->output);
            topk_softmax_scatter(stream, num_tokens, this->num_blocks, this->router_topk, this->topk_func->topk_val, this->topk_func->topk_pos, this->proj->output);
        } else {
            relu_inplace(stream, num_tokens, this->num_blocks, this->proj->output);
        }
        this->norm->prefill(stream, num_tokens, this->proj->output, this->proj->output);
    }
};

template <typename T>
struct BlockFFN : FFN<T> {
    int hidden_size;
    int intermediate_size;
    int num_blocks, block_size;
    float rms_norm_eps;
    bool use_kernel;
    int router_topk;
    Router<T> *router;
    int *nnz;
    T *nz_val; int *nz_idx;
    T* router_score;

    RMSNorm<T> *ffn_norm;
    Linear<T> *up_proj;
    T* up_proj_mean;
    T* projected_mean;
    Linear<T> *down_proj;
    NormSiLU<T> *norm_silu;

    Linear<T> *shared_gate_proj;
    Linear<T> *shared_up_proj;
    Linear<T> *shared_down_proj;

    T* shared_gated_up;

    BlockFFN(int hidden_size, int intermediate_size, float rms_norm_eps, int block_size, bool use_kernel = false, int router_topk = 0) {
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->num_blocks = intermediate_size / block_size;
        this->block_size = block_size;
        this->rms_norm_eps = rms_norm_eps;
        this->use_kernel = use_kernel;
        this->router_topk = router_topk;

        this->router = new Router<T>(hidden_size, num_blocks, rms_norm_eps, router_topk);

        this->ffn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);

        this->shared_gate_proj = new Linear<T>(hidden_size, block_size);
        this->shared_up_proj = new Linear<T>(hidden_size, block_size);
        this->shared_down_proj = new Linear<T>(block_size, hidden_size);

        this->up_proj = new Linear<T>(hidden_size, intermediate_size);
        this->norm_silu = new NormSiLU<T>(num_blocks, block_size, rms_norm_eps);
        this->down_proj = new Linear<T>(intermediate_size, hidden_size);
    }

    void init_weight_ptr(Memory* memory) {
        this->router->init_weight_ptr(memory);
        this->ffn_norm->init_weight_ptr(memory);
        this->shared_gate_proj->init_weight_ptr(memory);
        this->shared_up_proj->init_weight_ptr(memory);
        this->shared_down_proj->init_weight_ptr(memory);
        this->up_proj->init_weight_ptr(memory);
        this->up_proj_mean = (T*)memory->allocate_for_model(hidden_size * sizeof(T));
        this->norm_silu->init_weight_ptr(memory);
        this->down_proj->init_weight_ptr(memory);
        this->router_score = (T*)memory->allocate_for_model(1024 * num_blocks * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        offset = this->ffn_norm->init_output_ptr(memory, num_tokens, offset);
        offset = this->shared_gate_proj->init_output_ptr(memory, num_tokens, offset);
        offset = this->shared_up_proj->init_output_ptr(memory, num_tokens, offset);
        offset = memory->allocate((void**)&this->shared_gated_up, offset, num_tokens * block_size * sizeof(T));
        offset = this->shared_down_proj->init_output_ptr(memory, num_tokens, offset);
        offset = this->router->init_output_ptr(memory, num_tokens, offset);
        offset = this->up_proj->init_output_ptr(memory, num_tokens, offset);
        offset = memory->allocate((void**)&this->projected_mean, offset, num_tokens * sizeof(T));
        // norm_silu inplace
        offset = this->down_proj->init_output_ptr(memory, num_tokens, offset);
        this->output = this->down_proj->output;
        offset = memory->allocate((void**)&nnz, offset, sizeof(int));
        offset = memory->allocate((void**)&nz_val, offset, num_tokens * this->num_blocks * sizeof(T));
        offset = memory->allocate((void**)&nz_idx, offset, this->num_blocks * sizeof(int));
        return offset;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("router_score") != std::string::npos) {
            cudaMemcpy((void*)this->router_score, ptr, 1024 * num_blocks * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("router") != std::string::npos) {
            this->router->load_to_storage(name, ptr);
        } else if (name.find("expert_up_proj_mean") != std::string::npos) {
            cudaMemcpy((void*)this->up_proj_mean, ptr, hidden_size * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("expert_up_proj") != std::string::npos) {
            this->up_proj->load_to_storage(name, ptr);
        } else if (name.find("expert_down_proj") != std::string::npos) {
            this->down_proj->load_to_storage(name, ptr);
        } else if (name.find("expert_act.rms_norm") != std::string::npos) {
            this->norm_silu->load_to_storage(name, ptr);
        } else if (name.find("shared_experts.gate_proj") != std::string::npos) {
            this->shared_gate_proj->load_to_storage(name, ptr);
        } else if (name.find("shared_experts.up_proj") != std::string::npos) {
            this->shared_up_proj->load_to_storage(name, ptr);
        } else if (name.find("shared_experts.down_proj") != std::string::npos) {
            this->shared_down_proj->load_to_storage(name, ptr);
        } else if (name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn_norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {
        this->ffn_norm->prefill(stream, num_tokens, input, prev_output);
        this->router->prefill(stream, num_tokens, this->ffn_norm->output);

        this->up_proj->prefill(stream, num_tokens, this->ffn_norm->output);
        dot_product(stream, num_tokens, hidden_size, this->ffn_norm->output, this->up_proj_mean, this->projected_mean);
        this->norm_silu->prefill(stream, num_tokens, this->up_proj->output, this->projected_mean, this->up_proj->output);
        batched_mul(stream, num_tokens * this->num_blocks, this->block_size, this->up_proj->output, this->router->output, this->up_proj->output);
        this->down_proj->prefill(stream, num_tokens, this->up_proj->output);

        linear<T>(stream, num_tokens, this->hidden_size, this->block_size*2, this->ffn_norm->output, this->shared_gate_proj->weight, this->shared_gate_proj->output);
        gated_silu_interleaved<T>(stream, num_tokens, this->block_size, this->shared_gate_proj->output, this->shared_gated_up);

        this->shared_down_proj->prefill(stream, num_tokens, this->shared_gated_up);
        elementwise_add(stream, num_tokens, this->hidden_size, this->shared_down_proj->output, this->down_proj->output, this->down_proj->output);
    }

    void decode(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {
        this->ffn_norm->prefill(stream, num_tokens, input, prev_output);
        this->router->prefill(stream, num_tokens, this->ffn_norm->output);
        T* rs = this->router->output;

        dot_product(stream, num_tokens, hidden_size, this->ffn_norm->output, this->up_proj_mean, this->projected_mean);

        if (this->use_kernel) {
            if (num_tokens == 1) {
                nonzero(stream, this->num_blocks, rs, this->nnz, this->nz_val, this->nz_idx);
                sparse_up(stream, this->num_blocks, this->block_size, this->hidden_size, this->nnz, this->nz_idx, this->ffn_norm->output, this->up_proj->weight, this->up_proj->output);
                sparse_norm_silu(stream, this->num_blocks, this->block_size, this->nnz, this->nz_idx, this->nz_val, this->projected_mean, this->norm_silu->weight, this->norm_silu->eps, this->up_proj->output);
                sparse_down(stream, this->num_blocks, this->block_size, this->hidden_size, this->nnz, this->nz_idx, this->up_proj->output, this->down_proj->weight, this->down_proj->output);
            } else {
                throw std::invalid_argument("block_ffn: Unsupported num_tokens " + std::to_string(num_tokens));
            }
        } else {
            this->up_proj->prefill(stream, num_tokens, this->ffn_norm->output);
            this->norm_silu->prefill(stream, num_tokens, this->up_proj->output, this->projected_mean, this->up_proj->output);
            batched_mul(stream, num_tokens * this->num_blocks, this->block_size, this->up_proj->output, rs, this->up_proj->output);
            this->down_proj->prefill(stream, num_tokens, this->up_proj->output);
        }

        linear<T>(stream, num_tokens, this->hidden_size, this->block_size*2, this->ffn_norm->output, this->shared_gate_proj->weight, this->shared_gate_proj->output);
        gated_silu_interleaved<T>(stream, num_tokens, this->block_size, this->shared_gate_proj->output, this->shared_gated_up);

        this->shared_down_proj->prefill(stream, num_tokens, this->shared_gated_up);
        elementwise_add(stream, num_tokens, this->hidden_size, this->shared_down_proj->output, this->down_proj->output, this->down_proj->output);
    }
};
