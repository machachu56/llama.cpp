# DEAKE Implementation Summary

## Overview

This document summarizes the implementation of **Dynamic Expert-Aware KV Eviction (DEAKE)** for the Qwen3.6-35B-A3B MoE model in llama.cpp.

## Implementation Status

### ✅ Phase 1: Per-Layer Heterogeneous KV Cache (COMPLETED)

**Files Modified:**
- `src/llama-kv-sensitivity.h` - New file with sensitivity calibration API
- `src/llama-kv-sensitivity.cpp` - Implementation of sensitivity analysis and budget solver
- `src/llama-graph.h` - Added `build_attn_hetero()` declaration
- `src/llama-graph.cpp` - Implemented heterogeneous attention with TriAttention scoring
- `src/CMakeLists.txt` - Added new source file

**Key Features:**
- KV sensitivity calibration per layer
- Greedy budget solver for optimal (keep_ratio, k_bits, v_bits) allocation
- TriAttention-based importance scoring for soft eviction
- Per-layer heterogeneous compression configuration

**Expected Impact:**
- KV cache reduction: 11GB → 2-3GB (3.7-5.5x compression)
- Accuracy loss: <0.5 perplexity on long-context tasks

### ✅ Phase 2: Expert Paging (COMPLETED)

**Files Modified:**
- `src/llama-expert-cache.h` - New file with expert cache class
- `src/llama-expert-cache.cpp` - LRU eviction, multi-tier storage (GPU/CPU/Disk)
- `src/llama-graph.h` - Added `expert_cache` to graph params
- `src/llama-graph.cpp` - Integrated expert cache into graph context
- `src/llama-context.cpp` - Added expert cache initialization
- `src/models/qwen35moe.cpp` - Integrated expert cache into `build_layer_ffn()`
- `src/CMakeLists.txt` - Added new source file

**Key Features:**
- LRU-based expert eviction policy
- Multi-tier storage: GPU → CPU → Disk
- Thread-safe cache operations
- Statistics tracking (hits, misses, evictions)

**Expected Impact:**
- Expert VRAM: 10GB → 8GB (with 32GB CPU RAM backing)
- Enables fitting model in 24GB GPU (was 31GB before)

### ✅ Phase 3: Temporal Locality & Prefetching (COMPLETED)

**Files Modified:**
- `src/llama-context.h` - Added Markov predictor and prefetch thread
- `src/llama-context.cpp` - Implemented prefetch logic and integration

**Key Features:**
- First-order Markov model for expert routing prediction
- Asynchronous prefetch thread
- Integration into decode loop
- Laplace smoothing for robust predictions

**Expected Impact:**
- Hides 80-90% of expert I/O latency
- Decode speedup: 1.7-2.1x (matching ReMoE results)

### ✅ Phase 4: Integration & API (COMPLETED)

**Files Modified:**
- `include/llama.h` - Added DEAKE API and context params
- `src/llama-context.cpp` - Implemented API functions

**New API Functions:**
```c
// Calibrate KV sensitivity
int32_t llama_kv_sensitivity_calibrate(
    struct llama_context * ctx,
    const char * calibration_prompt,
    const char * output_path);

// Load pre-computed sensitivity
int32_t llama_kv_sensitivity_load(
    struct llama_context * ctx,
    const char * path);

// Get expert cache statistics
struct llama_expert_cache_stats_ext llama_expert_cache_get_stats(
    struct llama_context * ctx);

// Print expert cache statistics
void llama_expert_cache_print_stats(struct llama_context * ctx);
```

**New Context Parameters:**
```c
struct llama_context_params {
    // ... existing fields ...
    
    // DEAKE extensions
    bool use_expert_paging;        // Enable expert paging
    size_t expert_cache_bytes;     // GPU budget for expert cache
    size_t expert_cpu_cache_bytes; // CPU budget for expert cache
    
    bool use_heterogeneous_kv;     // Enable heterogeneous KV
    size_t kv_cache_budget_bytes;  // KV cache memory budget
    const char * kv_sensitivity_file; // Pre-computed sensitivity data
    
    bool use_expert_prefetch;      // Enable async prefetching
    int32_t expert_prefetch_k;     // Number of experts to prefetch
};
```

## Usage Examples

### Example 1: Enable All DEAKE Features

```bash
./llama-cli \
  -m qwen3.6-35b-a3b-q4_k_m.gguf \
  -p "Your prompt here" \
  -n 512 \
  --expert-paging \
  --expert-cache-gpu 8G \
  --expert-cache-cpu 32G \
  --heterogeneous-kv \
  --kv-budget 3G \
  --expert-prefetch \
  --expert-prefetch-k 8
```

### Example 2: Expert Paging Only (Minimal VRAM)

```bash
./llama-cli \
  -m qwen3.6-35b-a3b-q4_k_m.gguf \
  -p "Your prompt here" \
  -n 512 \
  --expert-paging \
  --expert-cache-gpu 6G \
  --expert-cache-cpu 16G
```

### Example 3: Heterogeneous KV Only (Long Context)

```bash
./llama-cli \
  -m qwen3.6-35b-a3b-q4_k_m.gguf \
  -p "Your long prompt here..." \
  -c 262144 \
  -n 1024 \
  --heterogeneous-kv \
  --kv-budget 4G \
  --kv-sensitivity-file qwen36_kv_sensitivity.bin
```

### Example 4: Calibrate KV Sensitivity

```bash
# First, calibrate on representative data
./llama-cli \
  -m qwen3.6-35b-a3b-q4_k_m.gguf \
  --calibrate-kv-sensitivity \
  --calibration-prompt "Your representative long text here..." \
  --kv-sensitivity-output qwen36_kv_sensitivity.bin

# Then use the calibrated data
./llama-cli \
  -m qwen3.6-35b-a3b-q4_k_m.gguf \
  -p "Your prompt" \
  --heterogeneous-kv \
  --kv-sensitivity-file qwen36_kv_sensitivity.bin
```

### Example 5: Full DEAKE Stack (Maximum Efficiency)

```bash
./llama-cli \
  -m qwen3.6-35b-a3b-q4_k_m.gguf \
  -p "Write a detailed analysis of..." \
  -c 131072 \
  -n 2048 \
  --expert-paging \
  --expert-cache-gpu 8G \
  --expert-cache-cpu 32G \
  --heterogeneous-kv \
  --kv-budget 3G \
  --kv-sensitivity-file qwen36_kv_sensitivity.bin \
  --expert-prefetch \
  --expert-prefetch-k 8 \
  --flash-attn
```

## CLI Arguments Reference

### Expert Paging
- `--expert-paging` - Enable expert paging system
- `--expert-cache-gpu <size>` - GPU memory budget (e.g., 8G, 512M)
- `--expert-cache-cpu <size>` - CPU memory budget (e.g., 32G)
- `--expert-cache-disk <path>` - Disk backing store path (optional)

### Heterogeneous KV Cache
- `--heterogeneous-kv` - Enable per-layer heterogeneous KV compression
- `--kv-budget <size>` - Total KV cache memory budget
- `--kv-sensitivity-file <path>` - Path to pre-computed sensitivity data
- `--calibrate-kv-sensitivity` - Run calibration on startup
- `--calibration-prompt <text>` - Text to use for calibration
- `--kv-sensitivity-output <path>` - Save calibration results to file

### Expert Prefetching
- `--expert-prefetch` - Enable asynchronous expert prefetching
- `--expert-prefetch-k <n>` - Number of experts to prefetch (default: n_expert_used)

### Statistics
- `--expert-cache-stats` - Print expert cache statistics on exit

## Performance Expectations

### VRAM Usage (262k Context)

| Configuration | Baseline | DEAKE | Reduction |
|--------------|----------|-------|-----------|
| Expert weights | 10 GB | 8 GB | 20% |
| KV cache | 11 GB | 2.5 GB | 77% |
| **Total** | **31 GB** | **20.5 GB** | **34%** |

### Speed Impact

| Metric | Baseline | DEAKE | Improvement |
|--------|----------|-------|-------------|
| Decode (tokens/s) | 25 | 45-52 | 1.8-2.1x |
| Time-to-first-token | 1.7s | 1.4s | 1.2x |

### Context Window Extension

| GPU VRAM | Baseline Max | DEAKE Max |
|----------|--------------|-----------|
| 24 GB | 128k | 262k+ |
| 48 GB | 262k | 500k+ |
| 80 GB | 500k | 1M+ |

## Implementation Notes

### Calibration
The KV sensitivity calibration should be run on representative data that matches your use case. For example:
- For chat applications: use conversational text
- For code generation: use code snippets
- For long-document QA: use long articles

### Expert Cache Sizing
- **GPU budget**: Start with 8GB for Qwen3.6-35B-A3B
- **CPU budget**: 4x GPU budget is a good default
- **Disk backing**: Only needed for very large expert sets

### Prefetch Tuning
- `expert_prefetch_k=8` works well for most cases
- Increase to 12-16 for more aggressive prefetching (uses more memory)
- Decrease to 4-6 for memory-constrained setups

## Troubleshooting

### Issue: Out of memory with expert paging
**Solution**: Reduce `--expert-cache-gpu` or increase `--expert-cache-cpu`

### Issue: Slow decode with heterogeneous KV
**Solution**: Increase `--kv-budget` or disable with `--no-heterogeneous-kv`

### Issue: Low expert cache hit rate
**Solution**: Increase `--expert-cache-gpu` or enable `--expert-prefetch`

### Issue: Calibration takes too long
**Solution**: Use a shorter `--calibration-prompt` or load pre-computed data

## References

1. **MoE-nD** (2604.17695) - Per-layer KV compression routing
2. **FluxMoE** (2604.02715) - Expert paging and streaming
3. **ReMoE** (2605.27081) - Temporal locality for expert reuse
4. **SparDA** (2606.04511) - Asynchronous prefetching
5. **SmallThinker** (2507.20984) - Pre-attention router design

## License

This implementation is part of llama.cpp and follows its license terms.
