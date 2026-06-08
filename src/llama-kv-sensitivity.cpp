#include "llama-kv-sensitivity.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-hparams.h"
#include "llama-batch.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

static constexpr uint32_t LLAMA_KV_SENS_MAGIC   = 0x4B565345;
static constexpr uint32_t LLAMA_KV_SENS_VERSION = 1;

static float compute_attn_entropy(const float * scores, size_t n) {
    double sum_exp = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_exp += exp((double) scores[i]);
    }

    double entropy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double p = exp((double) scores[i]) / sum_exp;
        if (p > 1e-12) {
            entropy -= p * log(p);
        }
    }
    return (float) entropy;
}

static float estimate_evict_sensitivity(
        const float * attn_scores,
        size_t        n_scores,
        float         evict_ratio) {
    if (n_scores == 0) {
        return 0.0f;
    }

    const size_t n_keep = std::max((size_t) 1, (size_t)(n_scores * (1.0f - evict_ratio)));

    std::vector<float> sorted_scores(attn_scores, attn_scores + n_scores);
    std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<float>());

    double total_mass = 0.0;
    double kept_mass   = 0.0;
    for (size_t i = 0; i < n_scores; ++i) {
        const float w = expf(sorted_scores[i]);
        total_mass += (double) w;
        if (i < n_keep) {
            kept_mass += (double) w;
        }
    }

    if (total_mass < 1e-12) {
        return 1.0f;
    }

    const float retained = (float)(kept_mass / total_mass);
    return 1.0f - retained;
}

static float estimate_quant_sensitivity(
        const float * attn_scores,
        size_t        n_scores,
        int           bits) {
    if (n_scores == 0 || bits <= 0) {
        return 0.0f;
    }

    const float n_levels = (float)(1 << bits);

    float max_score = -1e30f;
    float min_score =  1e30f;
    for (size_t i = 0; i < n_scores; ++i) {
        max_score = std::max(max_score, attn_scores[i]);
        min_score = std::min(min_score, attn_scores[i]);
    }

    const float range = max_score - min_score;
    if (range < 1e-12f) {
        return 0.0f;
    }

    const float step = range / n_levels;

    double total_error = 0.0;
    for (size_t i = 0; i < n_scores; ++i) {
        const float quantized = min_score + roundf((attn_scores[i] - min_score) / step) * step;
        const float err = attn_scores[i] - quantized;
        total_error += (double)(err * err);
    }

    return (float) sqrt(total_error / (double) n_scores) / range;
}

float llama_kv_sens_interpolate_keep(
        float keep_ratio,
        float s10, float s25, float s50, float s75, float s90) {

    struct pivot { float ratio; float sens; };
    const pivot pivots[] = {
        { 0.10f, s10 },
        { 0.25f, s25 },
        { 0.50f, s50 },
        { 0.75f, s75 },
        { 0.90f, s90 },
    };

    if (keep_ratio <= pivots[0].ratio) {
        return pivots[0].sens;
    }
    if (keep_ratio >= pivots[4].ratio) {
        return pivots[4].sens;
    }

    for (int i = 0; i < 4; ++i) {
        if (keep_ratio >= pivots[i].ratio && keep_ratio <= pivots[i + 1].ratio) {
            const float t = (keep_ratio - pivots[i].ratio) / (pivots[i + 1].ratio - pivots[i].ratio);
            return pivots[i].sens + t * (pivots[i + 1].sens - pivots[i].sens);
        }
    }

    return s50;
}

float llama_kv_sens_interpolate_bits(
        int bits,
        float s8, float s4, float s2) {

    if (bits >= 8) { return s8; }
    if (bits <= 2) { return s2; }
    if (bits >= 4) {
        const float t = (float)(bits - 4) / 4.0f;
        return s4 + t * (s8 - s4);
    }

    const float t = (float)(bits - 2) / 2.0f;
    return s2 + t * (s4 - s2);
}

float llama_kv_sensitivity::total_evict_cost(uint32_t il, float keep_ratio) const {
    if (il >= n_layer) {
        return 0.0f;
    }
    const auto & L = layers[il];
    return llama_kv_sens_interpolate_keep(
            keep_ratio,
            L.evict_10, L.evict_25, L.evict_50, L.evict_75, L.evict_90);
}

float llama_kv_sensitivity::total_quant_cost(uint32_t il, int k_bits, int v_bits) const {
    if (il >= n_layer) {
        return 0.0f;
    }
    const auto & L = layers[il];
    const float k_cost = llama_kv_sens_interpolate_bits(k_bits, L.k_quant_8, L.k_quant_4, L.k_quant_2);
    const float v_cost = llama_kv_sens_interpolate_bits(v_bits, L.v_quant_8, L.v_quant_4, L.v_quant_2);
    return 0.5f * (k_cost + v_cost);
}

void llama_kv_sensitivity_init_default(
        llama_kv_sensitivity & sens,
        uint32_t               n_layer) {
    sens = {};
    sens.n_layer = std::min(n_layer, (uint32_t) LLAMA_KV_SENS_MAX_LAYERS);
    sens.metric  = LLAMA_KV_SENS_METRIC_L2;

    for (uint32_t il = 0; il < sens.n_layer; ++il) {
        auto & L = sens.layers[il];
        L.evict_10  = 0.01f;
        L.evict_25  = 0.02f;
        L.evict_50  = 0.05f;
        L.evict_75  = 0.12f;
        L.evict_90  = 0.30f;
        L.k_quant_8 = 0.005f;
        L.k_quant_4 = 0.02f;
        L.k_quant_2 = 0.08f;
        L.v_quant_8 = 0.005f;
        L.v_quant_4 = 0.02f;
        L.v_quant_2 = 0.08f;
    }

    sens.is_calibrated = false;
}

struct llama_kv_sens_attn_accum {
    std::vector<float> ref_output;
    std::vector<float> attn_scores;
    bool               has_data = false;
};

static bool kv_sens_run_forward_pass(
        llama_context            * ctx,
        const llama_model        & model,
        const std::vector<llama_token> & tokens,
        std::vector<llama_kv_sens_attn_accum> & layer_accums) {

    if (tokens.empty()) {
        return false;
    }

    const auto & hparams = model.hparams;
    const uint32_t n_layer = hparams.n_layer();

    layer_accums.resize(n_layer);

    llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens.data()),
            (int32_t) tokens.size());

    const int ret = llama_decode(ctx, batch);

    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: llama_decode failed with code %d\n", __func__, ret);
        return false;
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        layer_accums[il].has_data = true;
    }

    return true;
}

static void kv_sens_compute_layer_sensitivity(
        llama_kv_layer_sensitivity       & out,
        const llama_kv_sens_attn_accum   & accum,
        uint32_t                           /* il */) {

    if (!accum.has_data || accum.attn_scores.empty()) {
        out.evict_10  = 0.01f;
        out.evict_25  = 0.02f;
        out.evict_50  = 0.05f;
        out.evict_75  = 0.12f;
        out.evict_90  = 0.30f;
        out.k_quant_8 = 0.005f;
        out.k_quant_4 = 0.02f;
        out.k_quant_2 = 0.08f;
        out.v_quant_8 = 0.005f;
        out.v_quant_4 = 0.02f;
        out.v_quant_2 = 0.08f;
        return;
    }

    const float * scores = accum.attn_scores.data();
    const size_t  n      = accum.attn_scores.size();

    out.evict_10 = estimate_evict_sensitivity(scores, n, 0.10f);
    out.evict_25 = estimate_evict_sensitivity(scores, n, 0.25f);
    out.evict_50 = estimate_evict_sensitivity(scores, n, 0.50f);
    out.evict_75 = estimate_evict_sensitivity(scores, n, 0.75f);
    out.evict_90 = estimate_evict_sensitivity(scores, n, 0.90f);

    out.k_quant_8 = estimate_quant_sensitivity(scores, n, 8);
    out.k_quant_4 = estimate_quant_sensitivity(scores, n, 4);
    out.k_quant_2 = estimate_quant_sensitivity(scores, n, 2);

    out.v_quant_8 = out.k_quant_8 * 0.8f;
    out.v_quant_4 = out.k_quant_4 * 0.8f;
    out.v_quant_2 = out.k_quant_2 * 0.8f;

    const float entropy = compute_attn_entropy(scores, n);
    const float max_entropy = logf((float) n);
    const float uniformity = (max_entropy > 0.0f) ? (entropy / max_entropy) : 0.0f;

    if (uniformity > 0.8f) {
        out.evict_10 *= 1.5f;
        out.evict_25 *= 1.4f;
        out.evict_50 *= 1.3f;
        out.evict_75 *= 1.2f;
        out.evict_90 *= 1.1f;
    } else if (uniformity < 0.3f) {
        out.evict_10 *= 0.6f;
        out.evict_25 *= 0.7f;
        out.evict_50 *= 0.8f;
        out.evict_75 *= 0.9f;
        out.evict_90 *= 0.95f;
    }
}

bool llama_kv_sensitivity_calibrate(
        llama_kv_sensitivity             & sens,
        const llama_model                & model,
        llama_context                    * ctx,
        const llama_kv_sens_calib_params & params) {

    const auto & hparams = model.hparams;
    const uint32_t n_layer = hparams.n_layer();

    sens.n_layer = std::min(n_layer, (uint32_t) LLAMA_KV_SENS_MAX_LAYERS);
    sens.metric  = params.metric;

    LLAMA_LOG_INFO("%s: calibrating KV sensitivity for %u layers\n", __func__, sens.n_layer);

    std::vector<llama_token> calib_tokens;

    if (params.calibration_prompt && params.calibration_prompt[0] != '\0') {
        const llama_vocab * vocab = llama_model_get_vocab(&model);
        const int32_t text_len = (int32_t) strlen(params.calibration_prompt);
        const int32_t n_prompt_tokens = llama_tokenize(
                vocab,
                params.calibration_prompt,
                text_len,
                nullptr, 0, false, true);

        if (n_prompt_tokens < 0) {
            LLAMA_LOG_ERROR("%s: tokenization failed\n", __func__);
            return false;
        }

        const int32_t n_calib = std::min((int32_t) n_prompt_tokens, (int32_t) params.n_tokens_max);
        calib_tokens.resize(n_calib);
        llama_tokenize(
                vocab,
                params.calibration_prompt,
                text_len,
                calib_tokens.data(), n_calib, false, true);
    } else {
        const llama_vocab * vocab = llama_model_get_vocab(&model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);

        calib_tokens.resize(std::min((uint32_t) n_vocab, params.n_tokens_max));
        for (size_t i = 0; i < calib_tokens.size(); ++i) {
            calib_tokens[i] = (llama_token)(i % n_vocab);
        }
    }

    if (calib_tokens.empty()) {
        LLAMA_LOG_WARN("%s: no calibration tokens available, using defaults\n", __func__);
        llama_kv_sensitivity_init_default(sens, n_layer);
        return true;
    }

    LLAMA_LOG_INFO("%s: using %zu calibration tokens\n", __func__, calib_tokens.size());

    std::vector<llama_kv_sens_attn_accum> layer_accums;
    if (!kv_sens_run_forward_pass(ctx, model, calib_tokens, layer_accums)) {
        LLAMA_LOG_WARN("%s: forward pass failed, using defaults\n", __func__);
        llama_kv_sensitivity_init_default(sens, n_layer);
        return true;
    }

    uint32_t n_full_attn = 0;
    for (uint32_t il = 0; il < sens.n_layer; ++il) {
        if (!hparams.is_recr(il)) {
            ++n_full_attn;
        }
    }

    LLAMA_LOG_INFO("%s: found %u full attention layers out of %u total\n",
            __func__, n_full_attn, sens.n_layer);

    for (uint32_t il = 0; il < sens.n_layer; ++il) {
        if (hparams.is_recr(il)) {
            auto & L = sens.layers[il];
            L = {};
            continue;
        }

        kv_sens_compute_layer_sensitivity(sens.layers[il], layer_accums[il], il);

        if (params.verbose) {
            const auto & L = sens.layers[il];
            LLAMA_LOG_INFO(
                "%s: layer %3u | evict[10/25/50/75/90] = [%.4f/%.4f/%.4f/%.4f/%.4f] | "
                "k_quant[8/4/2] = [%.4f/%.4f/%.4f] | v_quant[8/4/2] = [%.4f/%.4f/%.4f]\n",
                __func__, il,
                L.evict_10, L.evict_25, L.evict_50, L.evict_75, L.evict_90,
                L.k_quant_8, L.k_quant_4, L.k_quant_2,
                L.v_quant_8, L.v_quant_4, L.v_quant_2);
        }
    }

    sens.is_calibrated = true;

    LLAMA_LOG_INFO("%s: calibration complete\n", __func__);

    return true;
}

bool llama_kv_sensitivity_save(
        const llama_kv_sensitivity & sens,
        const char               * path) {

    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        LLAMA_LOG_ERROR("%s: failed to open %s for writing\n", __func__, path);
        return false;
    }

    fout.write(reinterpret_cast<const char *>(&LLAMA_KV_SENS_MAGIC),   sizeof(LLAMA_KV_SENS_MAGIC));
    fout.write(reinterpret_cast<const char *>(&LLAMA_KV_SENS_VERSION), sizeof(LLAMA_KV_SENS_VERSION));
    fout.write(reinterpret_cast<const char *>(&sens.n_layer),          sizeof(sens.n_layer));
    fout.write(reinterpret_cast<const char *>(&sens.metric),           sizeof(sens.metric));

    const uint8_t calibrated = sens.is_calibrated ? 1 : 0;
    fout.write(reinterpret_cast<const char *>(&calibrated), sizeof(calibrated));

    for (uint32_t il = 0; il < sens.n_layer; ++il) {
        fout.write(reinterpret_cast<const char *>(&sens.layers[il]), sizeof(llama_kv_layer_sensitivity));
    }

    if (!fout) {
        LLAMA_LOG_ERROR("%s: write error\n", __func__);
        return false;
    }

    LLAMA_LOG_INFO("%s: saved sensitivity data for %u layers to %s\n", __func__, sens.n_layer, path);

    return true;
}

bool llama_kv_sensitivity_load(
        llama_kv_sensitivity & sens,
        const char          * path) {

    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        LLAMA_LOG_ERROR("%s: failed to open %s for reading\n", __func__, path);
        return false;
    }

    uint32_t magic   = 0;
    uint32_t version = 0;

    fin.read(reinterpret_cast<char *>(&magic),   sizeof(magic));
    fin.read(reinterpret_cast<char *>(&version), sizeof(version));

    if (magic != LLAMA_KV_SENS_MAGIC) {
        LLAMA_LOG_ERROR("%s: invalid magic: 0x%08x (expected 0x%08x)\n",
                __func__, magic, LLAMA_KV_SENS_MAGIC);
        return false;
    }

    if (version != LLAMA_KV_SENS_VERSION) {
        LLAMA_LOG_ERROR("%s: unsupported version: %u (expected %u)\n",
                __func__, version, LLAMA_KV_SENS_VERSION);
        return false;
    }

    fin.read(reinterpret_cast<char *>(&sens.n_layer), sizeof(sens.n_layer));
    fin.read(reinterpret_cast<char *>(&sens.metric),  sizeof(sens.metric));

    uint8_t calibrated = 0;
    fin.read(reinterpret_cast<char *>(&calibrated), sizeof(calibrated));
    sens.is_calibrated = (calibrated != 0);

    if (sens.n_layer > LLAMA_KV_SENS_MAX_LAYERS) {
        LLAMA_LOG_ERROR("%s: n_layer %u exceeds maximum %d\n",
                __func__, sens.n_layer, LLAMA_KV_SENS_MAX_LAYERS);
        return false;
    }

    for (uint32_t il = 0; il < sens.n_layer; ++il) {
        fin.read(reinterpret_cast<char *>(&sens.layers[il]), sizeof(llama_kv_layer_sensitivity));
    }

    if (!fin) {
        LLAMA_LOG_ERROR("%s: read error\n", __func__);
        return false;
    }

    LLAMA_LOG_INFO("%s: loaded sensitivity data for %u layers from %s\n",
            __func__, sens.n_layer, path);

    return true;
}

static constexpr float KEEP_RATIOS[] = { 0.10f, 0.25f, 0.50f, 0.75f, 0.90f, 1.00f };
static constexpr int   QUANT_BITS[]  = { 2, 4, 8, 16 };
static constexpr int   N_KEEP_RATIOS = sizeof(KEEP_RATIOS) / sizeof(KEEP_RATIOS[0]);
static constexpr int   N_QUANT_BITS  = sizeof(QUANT_BITS)  / sizeof(QUANT_BITS[0]);

size_t llama_kv_config_memory_bytes(
        const llama_kv_layer_config & config,
        uint32_t                      n_embd_k_gqa,
        uint32_t                      n_embd_v_gqa,
        uint32_t                      kv_size,
        uint32_t                      n_stream) {

    const uint32_t effective_kv_size = (uint32_t)(kv_size * config.keep_ratio);
    const size_t k_bytes = (size_t) n_embd_k_gqa * effective_kv_size * n_stream * (config.k_bits / 8);
    const size_t v_bytes = (size_t) n_embd_v_gqa * effective_kv_size * n_stream * (config.v_bits / 8);

    return k_bytes + v_bytes;
}

static int find_keep_ratio_index(float keep_ratio) {
    for (int i = 0; i < N_KEEP_RATIOS; ++i) {
        if (fabsf(KEEP_RATIOS[i] - keep_ratio) < 1e-6f) {
            return i;
        }
    }
    return 0;
}

static int find_quant_bits_index(int bits) {
    for (int i = 0; i < N_QUANT_BITS; ++i) {
        if (QUANT_BITS[i] == bits) {
            return i;
        }
    }
    return 0;
}

struct kv_config_candidate {
    uint32_t layer_idx;
    float    new_keep_ratio;
    int      new_k_bits;
    int      new_v_bits;
    float    delta_sensitivity;
    size_t   delta_memory;
    float    score;
};

static kv_config_candidate find_best_upgrade(
        const llama_kv_sensitivity          & sens,
        const llama_kv_budget_solver_params & params,
        const llama_kv_layer_config         * current_configs,
        const size_t                        * current_memory) {

    kv_config_candidate best = {};
    best.score = -1.0f;

    for (uint32_t il = 0; il < params.n_layers; ++il) {
        if (il >= sens.n_layer) {
            continue;
        }

        const auto & cur = current_configs[il];
        const int cur_keep_idx = find_keep_ratio_index(cur.keep_ratio);
        const int cur_k_idx    = find_quant_bits_index(cur.k_bits);
        const int cur_v_idx    = find_quant_bits_index(cur.v_bits);

        const uint32_t n_embd_k = params.n_embd_k_gqa_per_layer ? params.n_embd_k_gqa_per_layer[il] : 0;
        const uint32_t n_embd_v = params.n_embd_v_gqa_per_layer ? params.n_embd_v_gqa_per_layer[il] : 0;

        if (n_embd_k == 0 || n_embd_v == 0) {
            continue;
        }

        const float cur_sens = sens.total_evict_cost(il, cur.keep_ratio)
                             + sens.total_quant_cost(il, cur.k_bits, cur.v_bits);

        auto try_upgrade = [&](float new_keep, int new_k, int new_v) {
            if (new_keep <= cur.keep_ratio && new_k <= cur.k_bits && new_v <= cur.v_bits) {
                return;
            }

            llama_kv_layer_config new_config = { new_keep, new_k, new_v };
            const size_t new_mem = llama_kv_config_memory_bytes(
                    new_config, n_embd_k, n_embd_v, params.kv_size, params.n_stream);

            if (new_mem <= current_memory[il]) {
                return;
            }

            const float new_sens = sens.total_evict_cost(il, new_keep)
                                 + sens.total_quant_cost(il, new_k, new_v);

            const float delta_sens = new_sens - cur_sens;
            const size_t delta_mem = new_mem - current_memory[il];

            if (delta_mem == 0) {
                return;
            }

            const float score = -delta_sens / (float) delta_mem;

            if (score > best.score) {
                best.layer_idx       = il;
                best.new_keep_ratio  = new_keep;
                best.new_k_bits      = new_k;
                best.new_v_bits      = new_v;
                best.delta_sensitivity = delta_sens;
                best.delta_memory    = delta_mem;
                best.score           = score;
            }
        };

        if (cur_keep_idx + 1 < N_KEEP_RATIOS) {
            try_upgrade(KEEP_RATIOS[cur_keep_idx + 1], cur.k_bits, cur.v_bits);
        }

        if (cur_k_idx + 1 < N_QUANT_BITS) {
            try_upgrade(cur.keep_ratio, QUANT_BITS[cur_k_idx + 1], cur.v_bits);
        }

        if (cur_v_idx + 1 < N_QUANT_BITS) {
            try_upgrade(cur.keep_ratio, cur.k_bits, QUANT_BITS[cur_v_idx + 1]);
        }
    }

    return best;
}

bool llama_kv_budget_solve(
        const llama_kv_sensitivity         & sens,
        const llama_kv_budget_solver_params & params,
        llama_kv_layer_config               * out_configs) {

    if (!out_configs || params.n_layers == 0) {
        return false;
    }

    if (params.verbose) {
        LLAMA_LOG_INFO("%s: solving KV budget for %u layers, budget = %.2f MB\n",
                __func__, params.n_layers, params.memory_budget_bytes / (1024.0f * 1024.0f));
    }

    for (uint32_t il = 0; il < params.n_layers; ++il) {
        out_configs[il] = { KEEP_RATIOS[0], QUANT_BITS[0], QUANT_BITS[0] };
    }

    std::vector<size_t> current_memory(params.n_layers, 0);
    size_t total_memory = 0;

    for (uint32_t il = 0; il < params.n_layers; ++il) {
        if (il < sens.n_layer && params.n_embd_k_gqa_per_layer && params.n_embd_v_gqa_per_layer) {
            current_memory[il] = llama_kv_config_memory_bytes(
                    out_configs[il],
                    params.n_embd_k_gqa_per_layer[il],
                    params.n_embd_v_gqa_per_layer[il],
                    params.kv_size,
                    params.n_stream);
            total_memory += current_memory[il];
        }
    }

    if (params.verbose) {
        LLAMA_LOG_INFO("%s: initial memory = %.2f MB\n",
                __func__, total_memory / (1024.0f * 1024.0f));
    }

    int iteration = 0;
    while (total_memory < params.memory_budget_bytes) {
        auto best = find_best_upgrade(sens, params, out_configs, current_memory.data());

        if (best.score < 0.0f) {
            if (params.verbose) {
                LLAMA_LOG_INFO("%s: no more upgrades available\n", __func__);
            }
            break;
        }

        if (total_memory + best.delta_memory > params.memory_budget_bytes) {
            if (params.verbose) {
                LLAMA_LOG_INFO("%s: budget exhausted (would exceed by %.2f MB)\n",
                        __func__, (total_memory + best.delta_memory - params.memory_budget_bytes) / (1024.0f * 1024.0f));
            }
            break;
        }

        out_configs[best.layer_idx].keep_ratio = best.new_keep_ratio;
        out_configs[best.layer_idx].k_bits     = best.new_k_bits;
        out_configs[best.layer_idx].v_bits     = best.new_v_bits;

        current_memory[best.layer_idx] += best.delta_memory;
        total_memory += best.delta_memory;

        if (params.verbose && (iteration % 10 == 0)) {
            LLAMA_LOG_INFO("%s: iteration %d: upgraded layer %u to (keep=%.2f, k=%d, v=%d), "
                          "memory = %.2f MB, Δsens = %.4f\n",
                    __func__, iteration, best.layer_idx,
                    best.new_keep_ratio, best.new_k_bits, best.new_v_bits,
                    total_memory / (1024.0f * 1024.0f), best.delta_sensitivity);
        }

        ++iteration;

        if (iteration > 10000) {
            LLAMA_LOG_WARN("%s: exceeded maximum iterations (10000)\n", __func__);
            break;
        }
    }

    if (params.verbose) {
        LLAMA_LOG_INFO("%s: final memory = %.2f MB (%.1f%% of budget)\n",
                __func__, total_memory / (1024.0f * 1024.0f),
                100.0f * total_memory / params.memory_budget_bytes);

        for (uint32_t il = 0; il < params.n_layers; ++il) {
            const auto & cfg = out_configs[il];
            LLAMA_LOG_INFO("%s: layer %3u: keep=%.2f, k=%2d, v=%2d, mem=%.2f MB\n",
                    __func__, il, cfg.keep_ratio, cfg.k_bits, cfg.v_bits,
                    current_memory[il] / (1024.0f * 1024.0f));
        }
    }

    return true;
}
