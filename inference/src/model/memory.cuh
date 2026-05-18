#pragma once
#include "../utils.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

#define ALIGN_SIZE 16

struct Memory {
    int64_t memory_limit;
    uint8_t* memory_pool;
    int64_t model_offset;

    Memory(int64_t memory_limit, void* memory_pool) {
        this->memory_limit = memory_limit;
        this->memory_pool = (uint8_t*)memory_pool;
        this->model_offset = 0;
    }

    void* allocate_for_model(size_t size) {
        uint8_t* ret = memory_pool + model_offset;
        model_offset += size;
        model_offset = ROUND_UP(model_offset, ALIGN_SIZE); // Align to 16 bytes
        if (model_offset > memory_limit) {
            throw std::runtime_error(
                "Memory pool exceeded while allocating model weights: offset " +
                std::to_string(model_offset) + " > limit " + std::to_string(memory_limit)
            );
        }
        return (void*)ret;
    }
    
    int64_t allocate(void** ptr, int64_t offset, size_t size = 0) { // 0 for reuse previous allocated memory, just need start offset, return value is useless
        *ptr = memory_pool + offset;
        offset += size;
        offset = ROUND_UP(offset, ALIGN_SIZE); // Align to 16 bytes
        if (offset > memory_limit) {
            throw std::runtime_error(
                "Memory pool exceeded while allocating runtime buffers: offset " +
                std::to_string(offset) + " > limit " + std::to_string(memory_limit)
            );
        }
        return offset;
    }
};