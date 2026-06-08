#include "llama-expert-cache.h"
#include "llama-model.h"
#include "llama-impl.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

llama_expert_cache::llama_expert_cache(
        const llama_model & model,
        size_t gpu_budget_bytes,
        size_t cpu_budget_bytes,
        const std::string & disk_path,
        ggml_backend_buffer_type_t buft_gpu,
        ggml_backend_t backend_gpu)
    : model(model)
    , gpu_capacity_bytes(gpu_budget_bytes)
    , cpu_capacity_bytes(cpu_budget_bytes)
    , buft_gpu(buft_gpu)
    , backend_gpu(backend_gpu)
    , disk_path(disk_path) {

    GGML_ASSERT(buft_gpu != nullptr);
    GGML_ASSERT(backend_gpu != nullptr);

    LLAMA_LOG_INFO("%s: initializing expert cache with %.2f MB GPU budget, %.2f MB CPU budget\n",
            __func__, gpu_budget_bytes / (1024.0 * 1024.0), cpu_budget_bytes / (1024.0 * 1024.0));

    const size_t max_tensors = LLAMA_EXPERT_CACHE_MAX_LAYERS * LLAMA_EXPERT_CACHE_MAX_EXPERTS * 3;

    ggml_init_params params_gpu = {
        /*.mem_size   =*/ ggml_tensor_overhead() * max_tensors,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ctx_gpu.reset(ggml_init(params_gpu));
    if (!ctx_gpu) {
        throw std::runtime_error("failed to initialize ggml context for expert cache (GPU)");
    }

    buf_gpu.reset(ggml_backend_buft_alloc_buffer(buft_gpu, gpu_budget_bytes));
    if (!buf_gpu) {
        LLAMA_LOG_WARN("%s: failed to allocate %.2f MB GPU buffer, expert paging disabled\n",
                __func__, gpu_budget_bytes / (1024.0 * 1024.0));
        gpu_capacity_bytes = 0;
    } else {
        LLAMA_LOG_INFO("%s: allocated %.2f MB GPU buffer for expert cache\n",
                __func__, gpu_budget_bytes / (1024.0 * 1024.0));
    }

    if (cpu_budget_bytes > 0) {
        ggml_init_params params_cpu = {
            /*.mem_size   =*/ ggml_tensor_overhead() * max_tensors,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        ctx_cpu.reset(ggml_init(params_cpu));
        if (!ctx_cpu) {
            LLAMA_LOG_WARN("%s: failed to initialize ggml context for expert cache (CPU)\n", __func__);
            cpu_capacity_bytes = 0;
        } else {
            auto buft_cpu = ggml_backend_cpu_buffer_type();
            buf_cpu.reset(ggml_backend_buft_alloc_buffer(buft_cpu, cpu_budget_bytes));
            if (!buf_cpu) {
                LLAMA_LOG_WARN("%s: failed to allocate %.2f MB CPU buffer\n",
                        __func__, cpu_budget_bytes / (1024.0 * 1024.0));
                cpu_capacity_bytes = 0;
            } else {
                LLAMA_LOG_INFO("%s: allocated %.2f MB CPU buffer for expert cache\n",
                        __func__, cpu_budget_bytes / (1024.0 * 1024.0));
            }
        }
    }

    if (!disk_path.empty()) {
        if (!init_disk_storage()) {
            LLAMA_LOG_WARN("%s: failed to initialize disk storage at %s\n", __func__, disk_path.c_str());
        }
    }
}

llama_expert_cache::~llama_expert_cache() {
    clear();
    close_disk_storage();
}

bool llama_expert_cache::init_disk_storage() {
    disk_fd = open(disk_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (disk_fd < 0) {
        LLAMA_LOG_ERROR("%s: failed to open disk file %s: %s\n", __func__, disk_path.c_str(), strerror(errno));
        return false;
    }

#ifdef __linux__
    int flags = POSIX_FADV_RANDOM;
    if (posix_fadvise(disk_fd, 0, 0, flags) != 0) {
        LLAMA_LOG_WARN("%s: posix_fadvise failed: %s\n", __func__, strerror(errno));
    }
#endif

    disk_next_offset = 0;
    LLAMA_LOG_INFO("%s: initialized disk storage at %s\n", __func__, disk_path.c_str());
    return true;
}

void llama_expert_cache::close_disk_storage() {
    if (disk_fd >= 0) {
        close(disk_fd);
        disk_fd = -1;
        if (!disk_path.empty()) {
            unlink(disk_path.c_str());
        }
    }
}

size_t llama_expert_cache::compute_expert_size_bytes(int32_t layer_id, int32_t /*expert_id*/) const {
    if (layer_id < 0 || layer_id >= (int32_t) model.layers.size()) {
        return 0;
    }

    const auto & layer = model.layers[layer_id];

    if (!layer.ffn_gate_exps || !layer.ffn_up_exps || !layer.ffn_down_exps) {
        return 0;
    }

    const int64_t n_ff_exp = layer.ffn_gate_exps->ne[0];
    const int64_t n_embd   = layer.ffn_gate_exps->ne[1];
    const size_t  elem_size = ggml_element_size(layer.ffn_gate_exps);

    const size_t gate_size = n_ff_exp * n_embd * elem_size;
    const size_t up_size   = n_ff_exp * n_embd * elem_size;
    const size_t down_size = n_embd * n_ff_exp * elem_size;

    return gate_size + up_size + down_size;
}

bool llama_expert_cache::allocate_page_gpu(llama_expert_page & page) {
    if (!buf_gpu) {
        return false;
    }

    const int32_t layer_id = page.key.layer_id;
    const int32_t expert_id = page.key.expert_id;

    if (layer_id < 0 || layer_id >= (int32_t) model.layers.size()) {
        return false;
    }

    const auto & layer = model.layers[layer_id];

    if (!layer.ffn_gate_exps || !layer.ffn_up_exps || !layer.ffn_down_exps) {
        return false;
    }

    const int64_t n_ff_exp = layer.ffn_gate_exps->ne[0];
    const int64_t n_embd   = layer.ffn_gate_exps->ne[1];

    page.gpu_gate = ggml_new_tensor_2d(ctx_gpu.get(), layer.ffn_gate_exps->type, n_embd, n_ff_exp);
    page.gpu_up   = ggml_new_tensor_2d(ctx_gpu.get(), layer.ffn_up_exps->type, n_embd, n_ff_exp);
    page.gpu_down = ggml_new_tensor_2d(ctx_gpu.get(), layer.ffn_down_exps->type, n_ff_exp, n_embd);

    if (!page.gpu_gate || !page.gpu_up || !page.gpu_down) {
        return false;
    }

    ggml_format_name(page.gpu_gate, "expert_cache_gate_l%d_e%d", layer_id, expert_id);
    ggml_format_name(page.gpu_up,   "expert_cache_up_l%d_e%d",   layer_id, expert_id);
    ggml_format_name(page.gpu_down, "expert_cache_down_l%d_e%d", layer_id, expert_id);

    const size_t gate_size = ggml_nbytes(page.gpu_gate);
    const size_t up_size   = ggml_nbytes(page.gpu_up);
    const size_t down_size = ggml_nbytes(page.gpu_down);

    page.size_bytes = gate_size + up_size + down_size;
    page.tier = LLAMA_EXPERT_TIER_GPU;

    return true;
}

bool llama_expert_cache::allocate_page_cpu(llama_expert_page & page) {
    if (!buf_cpu) {
        return false;
    }

    const int32_t layer_id = page.key.layer_id;
    const int32_t expert_id = page.key.expert_id;

    if (layer_id < 0 || layer_id >= (int32_t) model.layers.size()) {
        return false;
    }

    const auto & layer = model.layers[layer_id];

    if (!layer.ffn_gate_exps || !layer.ffn_up_exps || !layer.ffn_down_exps) {
        return false;
    }

    const int64_t n_ff_exp = layer.ffn_gate_exps->ne[0];
    const int64_t n_embd   = layer.ffn_gate_exps->ne[1];
    const size_t elem_size = ggml_element_size(layer.ffn_gate_exps);

    const size_t gate_size = n_ff_exp * n_embd * elem_size;
    const size_t up_size   = n_ff_exp * n_embd * elem_size;
    const size_t down_size = n_embd * n_ff_exp * elem_size;

    page.cpu_gate = ggml_backend_buffer_get_base(buf_cpu.get());
    if (!page.cpu_gate) {
        return false;
    }

    page.cpu_up   = (uint8_t *) page.cpu_gate + gate_size;
    page.cpu_down = (uint8_t *) page.cpu_up + up_size;

    page.size_bytes = gate_size + up_size + down_size;
    page.tier = LLAMA_EXPERT_TIER_CPU;

    GGML_UNUSED(expert_id);

    return true;
}

bool llama_expert_cache::allocate_page_disk(llama_expert_page & page) {
    if (disk_fd < 0) {
        return false;
    }

    const int32_t layer_id = page.key.layer_id;

    if (layer_id < 0 || layer_id >= (int32_t) model.layers.size()) {
        return false;
    }

    const auto & layer = model.layers[layer_id];

    if (!layer.ffn_gate_exps || !layer.ffn_up_exps || !layer.ffn_down_exps) {
        return false;
    }

    const int64_t n_ff_exp = layer.ffn_gate_exps->ne[0];
    const int64_t n_embd   = layer.ffn_gate_exps->ne[1];
    const size_t elem_size = ggml_element_size(layer.ffn_gate_exps);

    const size_t gate_size = n_ff_exp * n_embd * elem_size;
    const size_t up_size   = n_ff_exp * n_embd * elem_size;
    const size_t down_size = n_embd * n_ff_exp * elem_size;

    page.disk_offset_gate = disk_next_offset;
    page.disk_offset_up   = page.disk_offset_gate + gate_size;
    page.disk_offset_down = page.disk_offset_up + up_size;

    disk_next_offset = page.disk_offset_down + down_size;

    page.size_bytes = gate_size + up_size + down_size;
    page.tier = LLAMA_EXPERT_TIER_DISK;

    return true;
}

void llama_expert_cache::deallocate_page(llama_expert_page & page) {
    page.gpu_gate = nullptr;
    page.gpu_up   = nullptr;
    page.gpu_down = nullptr;
    page.cpu_gate = nullptr;
    page.cpu_up   = nullptr;
    page.cpu_down = nullptr;
    page.size_bytes = 0;
}

bool llama_expert_cache::load_expert_to_gpu(llama_expert_page & page) {
    const int32_t layer_id = page.key.layer_id;
    const int32_t expert_id = page.key.expert_id;

    if (layer_id < 0 || layer_id >= (int32_t) model.layers.size()) {
        return false;
    }

    const auto & layer = model.layers[layer_id];

    if (!layer.ffn_gate_exps || !layer.ffn_up_exps || !layer.ffn_down_exps) {
        return false;
    }

    const int64_t n_ff_exp = layer.ffn_gate_exps->ne[0];
    const int64_t n_embd   = layer.ffn_gate_exps->ne[1];
    const int64_t n_expert = layer.ffn_gate_exps->ne[2];

    if (expert_id < 0 || expert_id >= n_expert) {
        return false;
    }

    const size_t elem_size = ggml_element_size(layer.ffn_gate_exps);
    const size_t expert_stride = n_ff_exp * n_embd * elem_size;

    const uint8_t * gate_src = (const uint8_t *) layer.ffn_gate_exps->data + expert_id * expert_stride;
    const uint8_t * up_src   = (const uint8_t *) layer.ffn_up_exps->data   + expert_id * expert_stride;
    const uint8_t * down_src = (const uint8_t *) layer.ffn_down_exps->data + expert_id * expert_stride;

    ggml_backend_tensor_set(page.gpu_gate, gate_src, 0, ggml_nbytes(page.gpu_gate));
    ggml_backend_tensor_set(page.gpu_up,   up_src,   0, ggml_nbytes(page.gpu_up));
    ggml_backend_tensor_set(page.gpu_down, down_src, 0, ggml_nbytes(page.gpu_down));

    return true;
}

bool llama_expert_cache::load_expert_to_cpu(llama_expert_page & page) {
    const int32_t layer_id = page.key.layer_id;
    const int32_t expert_id = page.key.expert_id;

    if (layer_id < 0 || layer_id >= (int32_t) model.layers.size()) {
        return false;
    }

    const auto & layer = model.layers[layer_id];

    if (!layer.ffn_gate_exps || !layer.ffn_up_exps || !layer.ffn_down_exps) {
        return false;
    }

    const int64_t n_ff_exp = layer.ffn_gate_exps->ne[0];
    const int64_t n_embd   = layer.ffn_gate_exps->ne[1];
    const int64_t n_expert = layer.ffn_gate_exps->ne[2];

    if (expert_id < 0 || expert_id >= n_expert) {
        return false;
    }

    const size_t elem_size = ggml_element_size(layer.ffn_gate_exps);
    const size_t expert_stride = n_ff_exp * n_embd * elem_size;

    const uint8_t * gate_src = (const uint8_t *) layer.ffn_gate_exps->data + expert_id * expert_stride;
    const uint8_t * up_src   = (const uint8_t *) layer.ffn_up_exps->data   + expert_id * expert_stride;
    const uint8_t * down_src = (const uint8_t *) layer.ffn_down_exps->data + expert_id * expert_stride;

    const size_t gate_size = ggml_nbytes(page.gpu_gate);
    const size_t up_size   = ggml_nbytes(page.gpu_up);
    const size_t down_size = ggml_nbytes(page.gpu_down);

    memcpy(page.cpu_gate, gate_src, gate_size);
    memcpy(page.cpu_up,   up_src,   up_size);
    memcpy(page.cpu_down, down_src, down_size);

    return true;
}

bool llama_expert_cache::load_expert_to_disk(llama_expert_page & page) {
    if (disk_fd < 0) {
        return false;
    }

    const int32_t layer_id = page.key.layer_id;
    const int32_t expert_id = page.key.expert_id;

    if (layer_id < 0 || layer_id >= (int32_t) model.layers.size()) {
        return false;
    }

    const auto & layer = model.layers[layer_id];

    if (!layer.ffn_gate_exps || !layer.ffn_up_exps || !layer.ffn_down_exps) {
        return false;
    }

    const int64_t n_ff_exp = layer.ffn_gate_exps->ne[0];
    const int64_t n_embd   = layer.ffn_gate_exps->ne[1];
    const int64_t n_expert = layer.ffn_gate_exps->ne[2];

    if (expert_id < 0 || expert_id >= n_expert) {
        return false;
    }

    const size_t elem_size = ggml_element_size(layer.ffn_gate_exps);
    const size_t expert_stride = n_ff_exp * n_embd * elem_size;

    const uint8_t * gate_src = (const uint8_t *) layer.ffn_gate_exps->data + expert_id * expert_stride;
    const uint8_t * up_src   = (const uint8_t *) layer.ffn_up_exps->data   + expert_id * expert_stride;
    const uint8_t * down_src = (const uint8_t *) layer.ffn_down_exps->data + expert_id * expert_stride;

    const size_t gate_size = n_ff_exp * n_embd * elem_size;
    const size_t up_size   = n_ff_exp * n_embd * elem_size;
    const size_t down_size = n_embd * n_ff_exp * elem_size;

    ssize_t written = pwrite(disk_fd, gate_src, gate_size, page.disk_offset_gate);
    if (written != (ssize_t) gate_size) {
        return false;
    }

    written = pwrite(disk_fd, up_src, up_size, page.disk_offset_up);
    if (written != (ssize_t) up_size) {
        return false;
    }

    written = pwrite(disk_fd, down_src, down_size, page.disk_offset_down);
    if (written != (ssize_t) down_size) {
        return false;
    }

    return true;
}

bool llama_expert_cache::load_expert_from_disk(llama_expert_page & page) {
    if (disk_fd < 0 || !page.gpu_gate) {
        return false;
    }

    const int32_t layer_id = page.key.layer_id;

    if (layer_id < 0 || layer_id >= (int32_t) model.layers.size()) {
        return false;
    }

    const auto & layer = model.layers[layer_id];

    if (!layer.ffn_gate_exps || !layer.ffn_up_exps || !layer.ffn_down_exps) {
        return false;
    }

    const int64_t n_ff_exp = layer.ffn_gate_exps->ne[0];
    const int64_t n_embd   = layer.ffn_gate_exps->ne[1];
    const size_t elem_size = ggml_element_size(layer.ffn_gate_exps);

    const size_t gate_size = n_ff_exp * n_embd * elem_size;
    const size_t up_size   = n_ff_exp * n_embd * elem_size;
    const size_t down_size = n_embd * n_ff_exp * elem_size;

    std::vector<uint8_t> buffer(std::max({gate_size, up_size, down_size}));

    ssize_t bytes_read = pread(disk_fd, buffer.data(), gate_size, page.disk_offset_gate);
    if (bytes_read != (ssize_t) gate_size) {
        return false;
    }
    ggml_backend_tensor_set(page.gpu_gate, buffer.data(), 0, gate_size);

    bytes_read = pread(disk_fd, buffer.data(), up_size, page.disk_offset_up);
    if (bytes_read != (ssize_t) up_size) {
        return false;
    }
    ggml_backend_tensor_set(page.gpu_up, buffer.data(), 0, up_size);

    bytes_read = pread(disk_fd, buffer.data(), down_size, page.disk_offset_down);
    if (bytes_read != (ssize_t) down_size) {
        return false;
    }
    ggml_backend_tensor_set(page.gpu_down, buffer.data(), 0, down_size);

    return true;
}

void llama_expert_cache::evict_one_gpu() {
    if (lru_list.empty()) {
        return;
    }

    for (auto it = lru_list.end(); it != lru_list.begin(); ) {
        --it;
        const llama_expert_page_key & key = *it;

        auto page_it = pages.find(key);
        if (page_it == pages.end()) {
            lru_list.erase(it);
            lru_map.erase(key);
            continue;
        }

        llama_expert_page & page = page_it->second;

        if (page.tier == LLAMA_EXPERT_TIER_GPU) {
            if (cpu_capacity_bytes > 0 && cpu_usage_bytes + page.size_bytes <= cpu_capacity_bytes) {
                if (allocate_page_cpu(page)) {
                    if (load_expert_to_cpu(page)) {
                        gpu_usage_bytes -= page.size_bytes;
                        cpu_usage_bytes += page.size_bytes;
                        page.gpu_gate = nullptr;
                        page.gpu_up   = nullptr;
                        page.gpu_down = nullptr;
                        stats.n_evictions++;
                        return;
                    }
                    deallocate_page(page);
                }
            }

            if (disk_fd >= 0) {
                if (allocate_page_disk(page)) {
                    if (load_expert_to_disk(page)) {
                        gpu_usage_bytes -= page.size_bytes;
                        disk_usage_bytes += page.size_bytes;
                        page.gpu_gate = nullptr;
                        page.gpu_up   = nullptr;
                        page.gpu_down = nullptr;
                        stats.n_evictions++;
                        return;
                    }
                    deallocate_page(page);
                }
            }

            gpu_usage_bytes -= page.size_bytes;
            deallocate_page(page);
            pages.erase(page_it);
            lru_list.erase(it);
            lru_map.erase(key);
            stats.n_evictions++;
            return;
        }
    }
}

void llama_expert_cache::evict_one_cpu() {
    if (lru_list.empty()) {
        return;
    }

    for (auto it = lru_list.end(); it != lru_list.begin(); ) {
        --it;
        const llama_expert_page_key & key = *it;

        auto page_it = pages.find(key);
        if (page_it == pages.end()) {
            lru_list.erase(it);
            lru_map.erase(key);
            continue;
        }

        llama_expert_page & page = page_it->second;

        if (page.tier == LLAMA_EXPERT_TIER_CPU) {
            if (disk_fd >= 0) {
                if (allocate_page_disk(page)) {
                    if (load_expert_to_disk(page)) {
                        cpu_usage_bytes -= page.size_bytes;
                        disk_usage_bytes += page.size_bytes;
                        page.cpu_gate = nullptr;
                        page.cpu_up   = nullptr;
                        page.cpu_down = nullptr;
                        stats.n_evictions++;
                        return;
                    }
                    deallocate_page(page);
                }
            }

            cpu_usage_bytes -= page.size_bytes;
            deallocate_page(page);
            pages.erase(page_it);
            lru_list.erase(it);
            lru_map.erase(key);
            stats.n_evictions++;
            return;
        }
    }
}

void llama_expert_cache::evict_until_free_gpu(size_t bytes_needed) {
    while (gpu_usage_bytes + bytes_needed > gpu_capacity_bytes && !lru_list.empty()) {
        evict_one_gpu();
    }
}

void llama_expert_cache::evict_until_free_cpu(size_t bytes_needed) {
    while (cpu_usage_bytes + bytes_needed > cpu_capacity_bytes && !lru_list.empty()) {
        evict_one_cpu();
    }
}

void llama_expert_cache::touch_lru(const llama_expert_page_key & key) {
    auto it = lru_map.find(key);
    if (it != lru_map.end()) {
        lru_list.erase(it->second);
        lru_list.push_front(key);
        it->second = lru_list.begin();
    }
}

void llama_expert_cache::insert_lru(const llama_expert_page_key & key) {
    lru_list.push_front(key);
    lru_map[key] = lru_list.begin();
}

void llama_expert_cache::remove_lru(const llama_expert_page_key & key) {
    auto it = lru_map.find(key);
    if (it != lru_map.end()) {
        lru_list.erase(it->second);
        lru_map.erase(it);
    }
}

llama_expert_cache::fetch_result llama_expert_cache::fetch_expert(int32_t layer_id, int32_t expert_id) {
    std::lock_guard<std::mutex> lock(mtx);

    fetch_result result = { nullptr, nullptr, nullptr, false, LLAMA_EXPERT_TIER_GPU };

    if (gpu_capacity_bytes == 0) {
        return result;
    }

    const llama_expert_page_key key = { layer_id, expert_id };

    auto it = pages.find(key);
    if (it != pages.end()) {
        llama_expert_page & page = it->second;
        touch_lru(key);
        page.last_access_counter = ++access_counter;

        if (page.tier == LLAMA_EXPERT_TIER_GPU) {
            result.gate = page.gpu_gate;
            result.up   = page.gpu_up;
            result.down = page.gpu_down;
            result.was_cached = true;
            result.source_tier = LLAMA_EXPERT_TIER_GPU;
            stats.n_hits++;
            stats.n_gpu_hits++;
            return result;
        } else if (page.tier == LLAMA_EXPERT_TIER_CPU) {
            evict_until_free_gpu(page.size_bytes);
            if (!allocate_page_gpu(page)) {
                return result;
            }
            if (page.cpu_gate) {
                const size_t gate_size = ggml_nbytes(page.gpu_gate);
                const size_t up_size   = ggml_nbytes(page.gpu_up);
                const size_t down_size = ggml_nbytes(page.gpu_down);
                ggml_backend_tensor_set(page.gpu_gate, page.cpu_gate, 0, gate_size);
                ggml_backend_tensor_set(page.gpu_up,   page.cpu_up,   0, up_size);
                ggml_backend_tensor_set(page.gpu_down, page.cpu_down, 0, down_size);
                page.cpu_gate = nullptr;
                page.cpu_up   = nullptr;
                page.cpu_down = nullptr;
                cpu_usage_bytes -= page.size_bytes;
            }
            gpu_usage_bytes += page.size_bytes;
            page.tier = LLAMA_EXPERT_TIER_GPU;
            result.gate = page.gpu_gate;
            result.up   = page.gpu_up;
            result.down = page.gpu_down;
            result.was_cached = true;
            result.source_tier = LLAMA_EXPERT_TIER_CPU;
            stats.n_hits++;
            stats.n_cpu_hits++;
            return result;
        } else if (page.tier == LLAMA_EXPERT_TIER_DISK) {
            evict_until_free_gpu(page.size_bytes);
            if (!allocate_page_gpu(page)) {
                return result;
            }
            if (!load_expert_from_disk(page)) {
                deallocate_page(page);
                return result;
            }
            disk_usage_bytes -= page.size_bytes;
            gpu_usage_bytes += page.size_bytes;
            page.tier = LLAMA_EXPERT_TIER_GPU;
            result.gate = page.gpu_gate;
            result.up   = page.gpu_up;
            result.down = page.gpu_down;
            result.was_cached = true;
            result.source_tier = LLAMA_EXPERT_TIER_DISK;
            stats.n_hits++;
            stats.n_disk_hits++;
            return result;
        }
    }

    stats.n_misses++;

    const size_t expert_size = compute_expert_size_bytes(layer_id, expert_id);
    if (expert_size == 0) {
        return result;
    }

    if (expert_size > gpu_capacity_bytes) {
        LLAMA_LOG_WARN("%s: expert size %.2f MB exceeds cache capacity %.2f MB\n",
                __func__, expert_size / (1024.0 * 1024.0), gpu_capacity_bytes / (1024.0 * 1024.0));
        return result;
    }

    evict_until_free_gpu(expert_size);

    llama_expert_page page;
    page.key = key;

    if (!allocate_page_gpu(page)) {
        LLAMA_LOG_WARN("%s: failed to allocate page for expert %d layer %d\n",
                __func__, expert_id, layer_id);
        return result;
    }

    if (!load_expert_to_gpu(page)) {
        LLAMA_LOG_WARN("%s: failed to load expert %d layer %d to GPU\n",
                __func__, expert_id, layer_id);
        deallocate_page(page);
        return result;
    }

    gpu_usage_bytes += page.size_bytes;
    stats.bytes_transferred += page.size_bytes;

    page.last_access_counter = ++access_counter;

    pages[key] = page;
    insert_lru(key);

    result.gate = page.gpu_gate;
    result.up   = page.gpu_up;
    result.down = page.gpu_down;
    result.was_cached = false;
    result.source_tier = LLAMA_EXPERT_TIER_GPU;

    LLAMA_LOG_DEBUG("%s: loaded expert %d from layer %d, %.2f KB\n",
            __func__, expert_id, layer_id, page.size_bytes / 1024.0);

    return result;
}

void llama_expert_cache::prefetch_experts(const std::vector<llama_expert_page_key> & predictions) {
    std::lock_guard<std::mutex> lock(mtx);

    if (gpu_capacity_bytes == 0 || predictions.empty()) {
        return;
    }

    for (const auto & key : predictions) {
        if (pages.find(key) != pages.end()) {
            continue;
        }

        const size_t expert_size = compute_expert_size_bytes(key.layer_id, key.expert_id);
        if (expert_size == 0) {
            continue;
        }

        if (gpu_usage_bytes + expert_size > gpu_capacity_bytes) {
            break;
        }

        llama_expert_page page;
        page.key = key;

        if (!allocate_page_gpu(page)) {
            continue;
        }

        if (!load_expert_to_gpu(page)) {
            deallocate_page(page);
            continue;
        }

        gpu_usage_bytes += page.size_bytes;
        stats.bytes_transferred += page.size_bytes;

        page.last_access_counter = 0;

        pages[key] = page;
        insert_lru(key);

        LLAMA_LOG_DEBUG("%s: prefetched expert %d from layer %d\n",
                __func__, key.expert_id, key.layer_id);
    }
}

void llama_expert_cache::release_expert(int32_t layer_id, int32_t expert_id) {
    std::lock_guard<std::mutex> lock(mtx);

    const llama_expert_page_key key = { layer_id, expert_id };
    auto it = pages.find(key);

    if (it != pages.end()) {
        touch_lru(key);
    }
}

bool llama_expert_cache::is_cached(int32_t layer_id, int32_t expert_id) const {
    std::lock_guard<std::mutex> lock(mtx);

    const llama_expert_page_key key = { layer_id, expert_id };
    return pages.find(key) != pages.end();
}

size_t llama_expert_cache::get_gpu_usage_bytes() const {
    std::lock_guard<std::mutex> lock(mtx);
    return gpu_usage_bytes;
}

size_t llama_expert_cache::get_gpu_capacity_bytes() const {
    return gpu_capacity_bytes;
}

size_t llama_expert_cache::get_cpu_usage_bytes() const {
    std::lock_guard<std::mutex> lock(mtx);
    return cpu_usage_bytes;
}

size_t llama_expert_cache::get_cpu_capacity_bytes() const {
    return cpu_capacity_bytes;
}

size_t llama_expert_cache::get_disk_usage_bytes() const {
    std::lock_guard<std::mutex> lock(mtx);
    return disk_usage_bytes;
}

size_t llama_expert_cache::get_n_pages() const {
    std::lock_guard<std::mutex> lock(mtx);
    return pages.size();
}

llama_expert_cache_stats llama_expert_cache::get_stats() const {
    std::lock_guard<std::mutex> lock(mtx);
    return stats;
}

void llama_expert_cache::reset_stats() {
    std::lock_guard<std::mutex> lock(mtx);
    stats = {};
}

void llama_expert_cache::clear() {
    std::lock_guard<std::mutex> lock(mtx);

    for (auto & [key, page] : pages) {
        deallocate_page(page);
    }

    pages.clear();
    lru_list.clear();
    lru_map.clear();
    gpu_usage_bytes = 0;
    cpu_usage_bytes = 0;
    disk_usage_bytes = 0;
}
