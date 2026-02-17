# PatchXFormer Variant 0: Baseline Model

## Overview
This is the baseline model with **NO enhancements** - uses only standard transformer components.

## Files Created
1. **Model**: `models/PatchXFormer_Variant0_Baseline.py`
2. **Shell Script**: `PatchXFormer_Variant0_Baseline.sh`
3. **Updated Files**:
   - `models/__init__.py` - Added variant model import
   - `exp/exp_basic.py` - Registered variant model in model_dict

## Model Components

### ✅ What This Variant Uses (Standard Components):
- **Standard PatchEmbedding** (from `layers/Embed.py`)
  - No global token
  - Standard initialization
  - Standard positional embedding
  - Standard dropout
  
- **Standard EncoderLayer** (from `layers/Transformer_EncDec.py`)
  - Standard self-attention only
  - Standard LayerNorm
  - Standard feed-forward network
  
- **Standard FullAttention** (from `layers/SelfAttention_Family.py`)
  - No frequency enhancement
  - Standard attention mechanism
  
- **Standard LayerNorm**
  - No adaptive normalization
  
- **Simple Linear Prediction Head**
  - No residual connections
  - Direct linear projection

### ❌ What This Variant Does NOT Use:
- ❌ Enhanced patch embedding with global token
- ❌ Frequency-enhanced attention
- ❌ Adaptive normalization
- ❌ Cross-attention with exogenous features
- ❌ Residual connections in prediction head

## Usage

### Running the Baseline Model

```bash
bash PatchXFormer_Variant0_Baseline.sh
```

### Model Name
Use `--model PatchXFormer_Variant0_Baseline` in your run.py command.

### Model ID Format
Results will be saved with model_id: `variant0_baseline_solar_<dataset>96_<pred_len>`

## Expected Performance
This variant establishes the **baseline performance** against which all other variants will be compared.

## Next Steps
After running Variant 0, proceed to:
- Variant 1: +Enhanced Patch Embedding
- Variant 2: +Frequency-Enhanced Attention
- Variant 3: +Adaptive Normalization
- Variant 4: +Cross-Attention
- Variant 5: Full Model (+Residual Head)

## Notes
- All hyperparameters are the same as the full model for fair comparison
- Same random seed (2021) for reproducibility
- Results will be saved in `./results/variant0_baseline_*/`

