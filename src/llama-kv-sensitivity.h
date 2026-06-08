#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

struct llama_model;
struct llama_context;

#define LLAMA_KV_SENS_MAX_LAYERS 512

enum llama_kv_sens_metric : uint32_t {
    LLAMA_KV_SENS_METRIC_L2     = 0,
    LLAMA_KV_SENS_METRIC_COSINE = 1,
    LLAMA_KV_SENS_METRIC_KL     = 2,
};

struct llama_kv_layer_sensitivity {
    float evict_10;
    float evict_25;
    float evict_50;
    float evict_75;
    float evict_90;

    float k_quant_8;
    float k_quant_4;
    float k_quant_2;

    float v_quant_8;
    float v_quant_4;
    float v_quant_2;
};

struct llama_kv_sensitivity {
    uint32_t n_layer = 0;

    llama_kv_layer_sensitivity layers[LLAMA_KV_SENS_MAX_LAYERS];

    llama_kv_sens_metric metric = LLAMA_KV_SENS_METRIC_L2;

    bool is_calibrated = false;

    float total_evict_cost(uint32_t il, float keep_ratio) const;
    float total_quant_cost(uint32_t il, int k_bits, int v_bits) const;
};

struct llama_kv_sens_calib_params {
    const char * calibration_prompt = nullptr;
    uint32_t     n_tokens_max      = 1024;
    float        learning_rate      = 0.01f;
    bool         verbose            = false;
    llama_kv_sens_metric metric     = LLAMA_KV_SENS_METRIC_L2;
};

bool llama_kv_sensitivity_calibrate(
        llama_kv_sensitivity          & sens,
        const llama_model             & model,
        llama_context                 * ctx,
        const llama_kv_sens_calib_params & params);

bool llama_kv_sensitivity_save(
        const llama_kv_sensitivity & sens,
        const char               * path);

bool llama_kv_sensitivity_load(
        llama_kv_sensitivity & sens,
        const char          * path);

void llama_kv_sensitivity_init_default(
        llama_kv_sensitivity & sens,
        uint32_t               n_layer);

float llama_kv_sens_interpolate_keep(
        float keep_ratio,
        float s10, float s25, float s50, float s75, float s90);

float llama_kv_sens_interpolate_bits(
        int bits,
        float s8, float s4, float s2);

struct llama_kv_layer_config {
    float keep_ratio;
    int   k_bits;
    int   v_bits;
};

struct llama_kv_budget_solver_params {
    size_t   memory_budget_bytes;
    uint32_t kv_size;
    uint32_t n_stream;

    const uint32_t * n_embd_k_gqa_per_layer;
    const uint32_t * n_embd_v_gqa_per_layer;
    uint32_t         n_layers;

    bool verbose;
};

bool llama_kv_budget_solve(
        const llama_kv_sensitivity         & sens,
        const llama_kv_budget_solver_params & params,
        llama_kv_layer_config               * out_configs);

size_t llama_kv_config_memory_bytes(
        const llama_kv_layer_config & config,
        uint32_t                      n_embd_k_gqa,
        uint32_t                      n_embd_v_gqa,
        uint32_t                      kv_size,
        uint32_t                      n_stream);

