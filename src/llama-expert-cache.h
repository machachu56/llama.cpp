#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model;

#define LLAMA_EXPERT_CACHE_MAX_LAYERS 512
#define LLAMA_EXPERT_CACHE_MAX_EXPERTS 512

enum llama_expert_storage_tier {
    LLAMA_EXPERT_TIER_GPU  = 0,
    LLAMA_EXPERT_TIER_CPU  = 1,
    LLAMA_EXPERT_TIER_DISK = 2,
};

struct llama_expert_page_key {
    int32_t layer_id;
    int32_t expert_id;

    bool operator==(const llama_expert_page_key & other) const {
        return layer_id == other.layer_id && expert_id == other.expert_id;
    }
};

struct llama_expert_page_key_hash {
    size_t operator()(const llama_expert_page_key & key) const {
        return std::hash<int32_t>()(key.layer_id) ^ (std::hash<int32_t>()(key.expert_id) << 16);
    }
};

struct llama_expert_page {
    llama_expert_page_key key;

    llama_expert_storage_tier tier = LLAMA_EXPERT_TIER_GPU;

    ggml_tensor * gpu_gate = nullptr;
    ggml_tensor * gpu_up   = nullptr;
    ggml_tensor * gpu_down = nullptr;

    void * cpu_gate = nullptr;
    void * cpu_up   = nullptr;
    void * cpu_down = nullptr;

    off_t disk_offset_gate = 0;
    off_t disk_offset_up   = 0;
    off_t disk_offset_down = 0;

    size_t size_bytes = 0;

    uint64_t last_access_counter = 0;
};

struct llama_expert_cache_stats {
    uint64_t n_hits   = 0;
    uint64_t n_misses = 0;
    uint64_t n_evictions = 0;
    size_t   bytes_transferred = 0;
    uint64_t n_gpu_hits  = 0;
    uint64_t n_cpu_hits  = 0;
    uint64_t n_disk_hits = 0;
};

class llama_expert_cache {
public:
    llama_expert_cache(
            const llama_model & model,
            size_t gpu_budget_bytes,
            size_t cpu_budget_bytes,
            const std::string & disk_path,
            ggml_backend_buffer_type_t buft_gpu,
            ggml_backend_t backend_gpu);

    ~llama_expert_cache();

    llama_expert_cache(const llama_expert_cache &) = delete;
    llama_expert_cache & operator=(const llama_expert_cache &) = delete;

    struct fetch_result {
        ggml_tensor * gate;
        ggml_tensor * up;
        ggml_tensor * down;
        bool was_cached;
        llama_expert_storage_tier source_tier;
    };

    fetch_result fetch_expert(int32_t layer_id, int32_t expert_id);

    void prefetch_experts(const std::vector<llama_expert_page_key> & predictions);

    void release_expert(int32_t layer_id, int32_t expert_id);

    bool is_cached(int32_t layer_id, int32_t expert_id) const;

    size_t get_gpu_usage_bytes() const;
    size_t get_gpu_capacity_bytes() const;
    size_t get_cpu_usage_bytes() const;
    size_t get_cpu_capacity_bytes() const;
    size_t get_disk_usage_bytes() const;
    size_t get_n_pages() const;

    llama_expert_cache_stats get_stats() const;
    void reset_stats();

    void clear();

private:
    const llama_model & model;

    size_t gpu_capacity_bytes;
    size_t gpu_usage_bytes = 0;
    size_t cpu_capacity_bytes;
    size_t cpu_usage_bytes = 0;
    size_t disk_usage_bytes = 0;

    ggml_backend_buffer_type_t buft_gpu;
    ggml_backend_t backend_gpu;

    mutable std::mutex mtx;

    uint64_t access_counter = 0;

    std::unordered_map<llama_expert_page_key, llama_expert_page, llama_expert_page_key_hash> pages;

    std::list<llama_expert_page_key> lru_list;
    std::unordered_map<llama_expert_page_key, std::list<llama_expert_page_key>::iterator, llama_expert_page_key_hash> lru_map;

    mutable llama_expert_cache_stats stats;

    ggml_context_ptr ctx_gpu;
    ggml_backend_buffer_ptr buf_gpu;

    ggml_context_ptr ctx_cpu;
    ggml_backend_buffer_ptr buf_cpu;

    int disk_fd = -1;
    off_t disk_next_offset = 0;
    std::string disk_path;

    size_t compute_expert_size_bytes(int32_t layer_id, int32_t expert_id) const;

    bool allocate_page_gpu(llama_expert_page & page);
    bool allocate_page_cpu(llama_expert_page & page);
    bool allocate_page_disk(llama_expert_page & page);
    void deallocate_page(llama_expert_page & page);

    bool load_expert_to_gpu(llama_expert_page & page);
    bool load_expert_to_cpu(llama_expert_page & page);
    bool load_expert_to_disk(llama_expert_page & page);

    bool load_expert_from_disk(llama_expert_page & page);

    void evict_until_free_gpu(size_t bytes_needed);
    void evict_until_free_cpu(size_t bytes_needed);
    void evict_one_gpu();
    void evict_one_cpu();

    void touch_lru(const llama_expert_page_key & key);
    void insert_lru(const llama_expert_page_key & key);
    void remove_lru(const llama_expert_page_key & key);

    bool init_disk_storage();
    void close_disk_storage();
};

using llama_expert_cache_ptr = std::unique_ptr<llama_expert_cache>;
