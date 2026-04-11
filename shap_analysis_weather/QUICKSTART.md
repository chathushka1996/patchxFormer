# Quick Start Guide for SHAP Analysis

## Prerequisites

Before running SHAP analysis, ensure you have:

1. ✅ Trained PatchXFormer model checkpoint
2. ✅ Python 3.8+ installed
3. ✅ Required packages installed
4. ✅ Dataset available at `dataset/sl_piliyandala/`

## Installation Steps

### Step 1: Install Dependencies

```bash
cd shap_analysis_weather
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python check_setup.py
```

This will check:
- Python version
- Required packages
- Dataset availability
- Model checkpoints
- GPU availability

## Running SHAP Analysis

### Option 1: Automatic (Easiest)

Simply run:

```bash
python shap_analysis_weather/run_shap_analysis.py
```

This will:
- Automatically find your trained model
- Run SHAP analysis with default settings
- Save all results to `shap_analysis_weather/results/`

### Option 2: Manual (More Control)

If you want to specify exact parameters:

```bash
python shap_analysis_weather/shap_patchxformer_analysis.py \
    --checkpoint_path YOUR_MODEL_PATH/checkpoint.pth \
    --pred_len 96 \
    --seq_len 96 \
    --d_model 256 \
    --d_ff 512 \
    --e_layers 2 \
    --n_heads 8 \
    --num_samples 500 \
    --background_samples 100 \
    --output_dir shap_analysis_weather/results
```

### Option 3: Simple/Reliable (If Option 1 fails)

Use KernelExplainer (slower but more reliable):

```bash
python shap_analysis_weather/shap_simple_kernel.py \
    --checkpoint_path YOUR_MODEL_PATH/checkpoint.pth \
    --pred_len 96 \
    --seq_len 96 \
    --d_model 256
```

### Option 4: Interactive Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook shap_analysis_weather/interactive_shap_analysis.ipynb
```

## Expected Runtime

- **With GPU**: 10-20 minutes
- **With CPU**: 20-40 minutes

Factors affecting speed:
- Number of samples analyzed
- Model complexity
- Hardware specifications

## Output Files

After running, you'll find:

```
shap_analysis_weather/results/
├── plots/
│   ├── shap_summary_beeswarm.png          # Main summary plot
│   ├── shap_feature_importance_bar.png    # Feature rankings
│   ├── shap_dependence_plots.png          # Individual feature impacts
│   ├── shap_temporal_contribution.png     # Time-series analysis
│   ├── shap_temporal_heatmap.png          # Temporal heatmap
│   ├── shap_horizon_importance.png        # Prediction horizon analysis
│   └── shap_feature_interactions.png      # Feature correlations
├── csv_reports/
│   ├── feature_importance_weather.csv     # Rankings table
│   ├── feature_shap_correlations.csv      # Correlation matrix
│   └── shap_values_detailed.csv           # Raw SHAP values
└── COMPREHENSIVE_SHAP_REPORT.txt          # Full text report
```

## Using Results in Your Thesis

### Essential Figures

1. **shap_summary_beeswarm.png** 
   - Use in: Results section
   - Shows: Overall feature importance and value distribution

2. **shap_feature_importance_bar.png**
   - Use in: Results section
   - Shows: Clear ranking of features

3. **shap_dependence_plots.png**
   - Use in: Discussion section
   - Shows: How each feature affects predictions

### Essential Tables

1. **feature_importance_weather.csv**
   - Use in: Results section
   - Include as LaTeX table

### Text Report

**COMPREHENSIVE_SHAP_REPORT.txt** contains:
- Feature importance rankings
- Statistical analysis
- Key findings
- Comparison with reference paper
- Recommendations

Copy relevant sections to your thesis.

## Common Parameters

Adjust these based on your trained model:

### For pred_len=96 (24 hours)
```bash
--pred_len 96 --seq_len 96 --d_model 256 --d_ff 512 --e_layers 2 --n_heads 8
```

### For pred_len=192 (48 hours)
```bash
--pred_len 192 --seq_len 96 --d_model 384 --d_ff 768 --e_layers 2 --n_heads 8
```

### For pred_len=336 (3.5 days)
```bash
--pred_len 336 --seq_len 96 --d_model 512 --d_ff 1024 --e_layers 3 --n_heads 8
```

### For pred_len=720 (7.5 days)
```bash
--pred_len 720 --seq_len 96 --d_model 512 --d_ff 1024 --e_layers 3 --n_heads 8
```

## Troubleshooting

### Error: "Out of Memory"

**Solution 1**: Reduce samples
```bash
--num_samples 200 --background_samples 50
```

**Solution 2**: Use CPU
```bash
--use_gpu False
```

**Solution 3**: Use simple version
```bash
python shap_analysis_weather/shap_simple_kernel.py ...
```

### Error: "Checkpoint not found"

**Solution**: Specify full path
```bash
--checkpoint_path ./drive/MyDrive/msc-val/model_log/YOUR_EXPERIMENT/checkpoint.pth
```

Find your checkpoint:
```bash
find . -name "checkpoint.pth"
```

### Error: "SHAP explainer failed"

The script automatically falls back to KernelExplainer if GradientExplainer fails.

If still failing, use:
```bash
python shap_analysis_weather/shap_simple_kernel.py ...
```

### Slow Execution

SHAP analysis is computationally intensive. To speed up:

1. Reduce samples: `--num_samples 200`
2. Use GPU: Ensure CUDA is available
3. Use simple version: `shap_simple_kernel.py`

## Checking Results

### Quick Visual Check

Open these files to verify:
1. `results/plots/shap_summary_beeswarm.png` - Should show scatter plot
2. `results/plots/shap_feature_importance_bar.png` - Should show bar chart
3. `results/COMPREHENSIVE_SHAP_REPORT.txt` - Should have detailed text

### Data Validation

Check CSV files:
```python
import pandas as pd

# Load importance
df = pd.read_csv('results/csv_reports/feature_importance_weather.csv')
print(df)

# Should show 7 weather features ranked by importance
```

## Next Steps

After running SHAP analysis:

1. **Review comprehensive report**: `COMPREHENSIVE_SHAP_REPORT.txt`
2. **Check key figures**: Plots directory
3. **Analyze feature importance**: CSV reports
4. **Compare with reference paper**: PMC11695015
5. **Write thesis section**: Use findings and figures

## Support

### Documentation
- Main README: `shap_analysis_weather/README.md`
- SHAP docs: https://shap.readthedocs.io/

### Code Comments
All Python files have detailed comments explaining:
- What each function does
- Why specific approaches were chosen
- How to interpret results

### Example Commands

See `run_shap_analysis.py` for working examples.

## FAQ

**Q: How long should SHAP analysis take?**
A: 10-40 minutes depending on hardware and settings.

**Q: Can I run on CPU?**
A: Yes, but it will be slower. Use `--use_gpu False`.

**Q: How many samples should I use?**
A: 500 is recommended. Minimum 200, maximum limited by memory.

**Q: What if my model has different architecture?**
A: Adjust `--d_model`, `--n_heads`, `--e_layers` to match your trained model.

**Q: Can I analyze multiple prediction horizons?**
A: Yes, run the script multiple times with different `--pred_len` values.

**Q: How do I cite SHAP in my thesis?**
A: 
```
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to 
interpreting model predictions. In Advances in neural information 
processing systems (pp. 4765-4774).
```

## Complete Example

Here's a complete workflow:

```bash
# 1. Check setup
python shap_analysis_weather/check_setup.py

# 2. Install dependencies (if needed)
pip install -r shap_analysis_weather/requirements.txt

# 3. Run analysis (automatic)
python shap_analysis_weather/run_shap_analysis.py

# 4. Check results
ls -l shap_analysis_weather/results/plots/
cat shap_analysis_weather/results/COMPREHENSIVE_SHAP_REPORT.txt

# 5. (Optional) Interactive exploration
jupyter notebook shap_analysis_weather/interactive_shap_analysis.ipynb
```

---

**Ready to start?** Run:
```bash
python shap_analysis_weather/run_shap_analysis.py
```

Good luck with your thesis! 🎓
