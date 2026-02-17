# PatchXFormer Phase 1 Ablation Study Plan

## Overview
This plan defines 5 progressive variants that build from baseline to full model, allowing us to measure the contribution of each enhancement component.

---

## Variant Definitions

### Variant 0: Baseline (No Enhancements)
**Purpose**: Establish baseline performance with standard components

**Components**:
- ✅ Standard `PatchEmbedding` (from Embed.py)
- ✅ Standard `EncoderLayer` (from Transformer_EncDec.py)
- ✅ Standard `FullAttention` only (no frequency enhancement)
- ✅ Standard `LayerNorm` (no adaptive normalization)
- ✅ Simple Linear Prediction Head (no residual connections)
- ❌ No global token
- ❌ No frequency-enhanced attention
- ❌ No adaptive normalization
- ❌ No cross-attention with exogenous features
- ❌ No residual connections in head

**Expected Performance**: Lowest baseline to compare against

---

### Variant 1: +Enhanced Patch Embedding
**Purpose**: Test impact of enhanced patch embedding with global token

**Components**:
- ✅ `EnhancedPatchEmbedding` (with global token, positional embedding, layer norm)
- ✅ Standard `EncoderLayer`
- ✅ Standard `FullAttention` only
- ✅ Standard `LayerNorm`
- ✅ Simple Linear Prediction Head
- ❌ No frequency-enhanced attention
- ❌ No adaptive normalization
- ❌ No cross-attention
- ❌ No residual connections in head

**Enhancement Added**: Enhanced patch embedding with global token

---

### Variant 2: +Frequency-Enhanced Attention
**Purpose**: Test impact of frequency domain enhancement in attention

**Components**:
- ✅ `EnhancedPatchEmbedding`
- ✅ `EnhancedHybridEncoderLayer` with `FrequencyEnhancedAttention`
- ✅ Standard `FullAttention` (dual attention: frequency + standard)
- ✅ Standard `LayerNorm` (not adaptive yet)
- ✅ Simple Linear Prediction Head
- ❌ No adaptive normalization
- ❌ No cross-attention
- ❌ No residual connections in head

**Enhancement Added**: Frequency-enhanced attention mechanism

---

### Variant 3: +Adaptive Normalization
**Purpose**: Test impact of adaptive normalization vs standard LayerNorm

**Components**:
- ✅ `EnhancedPatchEmbedding`
- ✅ `EnhancedHybridEncoderLayer` with frequency attention
- ✅ Dual attention (frequency + standard)
- ✅ `AdaptiveNormalization` (replaces LayerNorm in encoder)
- ✅ Simple Linear Prediction Head
- ❌ No cross-attention
- ❌ No residual connections in head

**Enhancement Added**: Adaptive normalization with learnable parameters

---

### Variant 4: +Cross-Attention
**Purpose**: Test impact of cross-attention with exogenous features

**Components**:
- ✅ `EnhancedPatchEmbedding`
- ✅ `EnhancedHybridEncoderLayer` with frequency attention
- ✅ Dual attention (frequency + standard)
- ✅ `AdaptiveNormalization`
- ✅ Cross-attention with exogenous features (x_mark_enc)
- ✅ Simple Linear Prediction Head
- ❌ No residual connections in head

**Enhancement Added**: Cross-attention mechanism for exogenous features

---

### Variant 5: Full Model (+Residual Prediction Head)
**Purpose**: Complete model with all enhancements

**Components**:
- ✅ `EnhancedPatchEmbedding`
- ✅ `EnhancedHybridEncoderLayer` with frequency attention
- ✅ Dual attention (frequency + standard)
- ✅ `AdaptiveNormalization`
- ✅ Cross-attention with exogenous features
- ✅ `EnhancedPredictionHead` with residual connections

**Enhancement Added**: Residual connections in prediction head

**This is the FULL MODEL** - all enhancements enabled

---

## Implementation Strategy

### Configuration Flags
Add to model configs:
```python
# Ablation study flags
use_enhanced_patch_embedding = True/False
use_frequency_attention = True/False
use_adaptive_norm = True/False
use_cross_attention = True/False
use_residual_head = True/False
```

### Variant Mapping

| Variant | Enhanced Patch | Frequency Attn | Adaptive Norm | Cross Attn | Residual Head |
|---------|---------------|----------------|---------------|------------|---------------|
| 0 (Baseline) | ❌ | ❌ | ❌ | ❌ | ❌ |
| 1 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2 | ✅ | ✅ | ❌ | ❌ | ❌ |
| 3 | ✅ | ✅ | ✅ | ❌ | ❌ |
| 4 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 5 (Full) | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Experimental Protocol

### Training Configuration
- Use same hyperparameters for all variants (from PatchXFormer.sh)
- Same random seed for reproducibility
- Same train/val/test splits
- Same number of epochs

### Evaluation Metrics
For each variant, measure:
1. **MSE** (Mean Squared Error)
2. **MAE** (Mean Absolute Error)
3. **RMSE** (Root Mean Squared Error)
4. **MAPE** (Mean Absolute Percentage Error)
5. **MSPE** (Mean Squared Percentage Error)
6. **DTW** (Dynamic Time Warping) - if enabled

### Comparison Method
- Compare each variant against Variant 0 (baseline)
- Calculate improvement percentage: `(baseline_metric - variant_metric) / baseline_metric * 100`
- Track cumulative improvements across variants

---

## Expected Outcomes

### Research Questions Answered:
1. **Q1**: Does enhanced patch embedding improve performance?
   - Compare Variant 0 vs Variant 1

2. **Q2**: Does frequency-enhanced attention help?
   - Compare Variant 1 vs Variant 2

3. **Q3**: Does adaptive normalization outperform LayerNorm?
   - Compare Variant 2 vs Variant 3

4. **Q4**: Does cross-attention with exogenous features help?
   - Compare Variant 3 vs Variant 4

5. **Q5**: Do residual connections in prediction head help?
   - Compare Variant 4 vs Variant 5 (Full Model)

### Success Criteria:
- Each variant should show measurable improvement or at least maintain performance
- Full model (Variant 5) should show best overall performance
- If any variant degrades performance, that component may need refinement

---

## Implementation Checklist

### Code Changes Required:
- [ ] Add configuration flags to model initialization
- [ ] Create conditional logic in `EnhancedPatchEmbedding` to fallback to `PatchEmbedding`
- [ ] Create conditional logic in `EnhancedHybridEncoderLayer` to disable frequency attention
- [ ] Create conditional logic to switch between `AdaptiveNormalization` and `LayerNorm`
- [ ] Create conditional logic to enable/disable cross-attention
- [ ] Create conditional logic in prediction head to use simple vs enhanced head
- [ ] Update model `__init__` to accept ablation flags
- [ ] Create variant naming convention for checkpoints/logs

### Experiment Script Changes:
- [ ] Update PatchXFormer.sh to run all 5 variants sequentially
- [ ] Add variant identifier to model_id for tracking
- [ ] Ensure separate checkpoint directories per variant
- [ ] Add summary script to compare all variants

### Documentation:
- [ ] Document variant configurations
- [ ] Create results comparison table template
- [ ] Prepare visualization script for ablation results

---

## File Structure for Results

```
results/
├── variant_0_baseline/
│   ├── metrics.npy
│   ├── pred.npy
│   └── true.npy
├── variant_1_enhanced_patch/
│   ├── metrics.npy
│   ├── pred.npy
│   └── true.npy
├── variant_2_frequency_attention/
│   ├── metrics.npy
│   ├── pred.npy
│   └── true.npy
├── variant_3_adaptive_norm/
│   ├── metrics.npy
│   ├── pred.npy
│   └── true.npy
├── variant_4_cross_attention/
│   ├── metrics.npy
│   ├── pred.npy
│   └── true.npy
└── variant_5_full_model/
    ├── metrics.npy
    ├── pred.npy
    └── true.npy
```

---

## Timeline Estimate

- **Variant 0 (Baseline)**: 1-2 days (implementation + training)
- **Variant 1**: 1 day (implementation + training)
- **Variant 2**: 1 day (implementation + training)
- **Variant 3**: 1 day (implementation + training)
- **Variant 4**: 1 day (implementation + training)
- **Variant 5 (Full)**: 1 day (verification + training)
- **Analysis & Reporting**: 1-2 days

**Total**: ~7-10 days for complete Phase 1 ablation study

---

## Notes

1. **Progressive Enhancement**: Each variant builds on the previous one, making it easy to isolate component contributions
2. **Full Model Access**: Variant 5 is the full model - can be used for final experiments
3. **Reproducibility**: All variants use same random seed and data splits
4. **Fair Comparison**: Same hyperparameters ensure fair comparison
5. **Extensibility**: Easy to add more variants later if needed

---

## Next Steps After Phase 1

If Phase 1 shows significant improvements from specific components:
- **Phase 2**: Sub-component ablation (e.g., test global token separately)
- **Phase 3**: Hyperparameter tuning for best-performing variant
- **Phase 4**: Cross-dataset validation

---

**End of Plan**

