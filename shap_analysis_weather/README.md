# SHAP Analysis for PatchXFormer Solar Power Forecasting

## Overview

This directory contains comprehensive SHAP (SHapley Additive exPlanations) analysis tools for interpreting the PatchXFormer transformer-based solar power forecasting model. The analysis focuses specifically on **weather features only**, excluding time-based features (day of year, time of day) as requested.

## What is SHAP?

SHAP (SHapley Additive exPlanations) is a unified framework for interpreting machine learning model predictions. It assigns each feature an importance value based on cooperative game theory (Shapley values). SHAP provides:

- **Local Interpretability**: Explains individual predictions
- **Global Interpretability**: Shows overall feature importance
- **Consistency**: Features that contribute more receive higher importance
- **Theoretical Guarantees**: Based on solid mathematical foundations

### Why SHAP for Transformers?

For transformer-based time series models like PatchXFormer:

1. **Handles Complexity**: Transformers use attention mechanisms that create complex feature interactions
2. **Temporal Understanding**: SHAP can analyze how features contribute across time steps
3. **Model-Agnostic**: Works regardless of internal model architecture
4. **Gradient-Based**: GradientExplainer leverages backpropagation for efficiency

## SHAP Explainer Types

### 1. GradientExplainer (Recommended for PatchXFormer)

**Best for**: Deep learning models with differentiable operations

**Advantages**:
- Fast computation using gradients
- Works well with transformers
- Handles attention mechanisms effectively
- Provides accurate approximations

**How it works**: Computes SHAP values by integrating gradients along the path from a baseline to the input.

### 2. DeepExplainer

**Best for**: PyTorch/TensorFlow models

**Advantages**:
- Fast for neural networks
- Uses DeepLIFT algorithm
- Good for CNNs and RNNs

**How it works**: Propagates Shapley values through the network layers.

### 3. KernelExplainer

**Best for**: Any model (model-agnostic)

**Disadvantages**:
- Very slow for large models
- Computationally expensive

**How it works**: Uses weighted linear regression to estimate SHAP values.

## Weather Features Analyzed

The analysis focuses on these meteorological variables:

1. **temp** - Temperature (°C)
2. **dew** - Dew point (°C)
3. **humidity** - Relative humidity (%)
4. **winddir** - Wind direction (degrees)
5. **windspeed** - Wind speed (m/s)
6. **pressure** - Atmospheric pressure (hPa)
7. **cloudcover** - Cloud cover (%)

**Note**: Time features (dayofyear, timeofday) are excluded from the analysis as requested.

## Directory Structure

```
shap_analysis_weather/
├── shap_patchxformer_analysis.py    # Main SHAP analysis script
├── run_shap_analysis.py              # Helper script to run analysis
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── results/                          # Output directory
    ├── plots/                        # Visualization outputs
    │   ├── shap_summary_beeswarm.png/pdf
    │   ├── shap_feature_importance_bar.png/pdf
    │   ├── shap_dependence_plots.png/pdf
    │   ├── shap_temporal_contribution.png/pdf
    │   ├── shap_temporal_heatmap.png/pdf
    │   ├── shap_horizon_importance.png/pdf
    │   └── shap_feature_interactions.png/pdf
    ├── csv_reports/                  # CSV data files
    │   ├── feature_importance_weather.csv
    │   ├── feature_shap_correlations.csv
    │   └── shap_values_detailed.csv
    └── COMPREHENSIVE_SHAP_REPORT.txt # Detailed text report
```

## Installation

### Required Packages

```bash
pip install -r requirements.txt
```

Key dependencies:
- `shap>=0.42.0` - SHAP library
- `torch>=1.10.0` - PyTorch
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization
- `scikit-learn>=1.0.0` - Machine learning utilities
- `tqdm>=4.62.0` - Progress bars

## Usage

### Method 1: Quick Start (Recommended)

Run the helper script that automatically finds your trained model:

```bash
python shap_analysis_weather/run_shap_analysis.py
```

### Method 2: Manual Execution

If you want more control over parameters:

```bash
python shap_analysis_weather/shap_patchxformer_analysis.py \
    --checkpoint_path PATH_TO_YOUR_MODEL/checkpoint.pth \
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

### Parameters Explanation

- `--checkpoint_path`: Path to trained model checkpoint (required)
- `--pred_len`: Prediction horizon (96 = 24 hours for 15-min intervals)
- `--seq_len`: Input sequence length (96 = 24 hours)
- `--num_samples`: Number of samples to analyze (500 recommended)
- `--background_samples`: Background dataset size for SHAP (100 recommended)
- `--output_dir`: Where to save results

## Output Files

### 1. Visualizations (plots/)

#### shap_summary_beeswarm.png
- **Purpose**: Shows distribution of SHAP values for each feature
- **Interpretation**: 
  - Horizontal position = SHAP value (impact on prediction)
  - Color = Feature value (red=high, blue=low)
  - Vertical position = Feature (ranked by importance)

#### shap_feature_importance_bar.png
- **Purpose**: Bar chart of mean absolute SHAP values
- **Interpretation**: Taller bars = more important features

#### shap_dependence_plots.png
- **Purpose**: Shows how each feature's value affects predictions
- **Interpretation**: 
  - X-axis = Feature value
  - Y-axis = SHAP value (impact)
  - Trend line shows relationship
  - Non-linear curves indicate complex relationships

#### shap_temporal_contribution.png
- **Purpose**: Shows feature importance across input time steps
- **Interpretation**: 
  - X-axis = Time step in sequence
  - Y-axis = Mean absolute SHAP value
  - Lines show each feature's temporal contribution

#### shap_temporal_heatmap.png
- **Purpose**: Heatmap of feature contributions over time
- **Interpretation**: 
  - Columns = Time steps
  - Rows = Features
  - Color intensity = Importance

#### shap_horizon_importance.png
- **Purpose**: Feature importance for different prediction horizons
- **Interpretation**: Shows which features matter for short/long-term forecasts

#### shap_feature_interactions.png
- **Purpose**: Correlation matrix of SHAP values
- **Interpretation**: 
  - High correlation = features interact in affecting predictions
  - Useful for understanding feature dependencies

### 2. CSV Reports (csv_reports/)

#### feature_importance_weather.csv
Columns:
- `Feature`: Weather feature name
- `Mean_Abs_SHAP`: Mean absolute SHAP value
- `Importance_Rank`: Ranking (1 = most important)

#### feature_shap_correlations.csv
- Correlation matrix of SHAP values between features
- Shows which features have related impacts

#### shap_values_detailed.csv
- Raw SHAP values and feature values for all analyzed samples
- Useful for custom analysis

### 3. Comprehensive Report

`COMPREHENSIVE_SHAP_REPORT.txt` includes:
- Model configuration details
- Feature importance rankings
- Detailed analysis of each weather feature
- Key findings for thesis
- Comparison with reference paper (PMC11695015)
- Recommendations for model improvement
- Limitations and considerations

## Interpreting SHAP Values

### Positive SHAP Values
- Feature contributes to **INCREASING** predicted solar power
- Example: High temperature → Higher SHAP → More solar power predicted

### Negative SHAP Values
- Feature contributes to **DECREASING** predicted solar power
- Example: High cloud cover → Lower SHAP → Less solar power predicted

### Magnitude
- **Large |SHAP|**: Feature strongly influences prediction
- **Small |SHAP|**: Feature weakly influences prediction

### Feature Value Color Coding
- **Red/Pink**: High feature value
- **Blue**: Low feature value
- Shows if high/low values of feature increase/decrease predictions

## Theoretical Background

### Shapley Values

SHAP is based on Shapley values from cooperative game theory:

```
φᵢ = Σ_{S⊆N\{i}} |S|!(|N|-|S|-1)!/|N|! [f(S∪{i}) - f(S)]
```

Where:
- φᵢ = SHAP value for feature i
- N = Set of all features
- S = Subset of features
- f(S) = Model prediction using feature subset S

### Properties

1. **Efficiency**: Sum of SHAP values = prediction - baseline
2. **Symmetry**: Equal features get equal values
3. **Dummy**: Irrelevant features get zero value
4. **Additivity**: Values combine linearly for ensembles

## For Your Thesis

### What to Include

1. **Introduction to SHAP**
   - Why interpretability matters for solar forecasting
   - Brief explanation of Shapley values
   - Why SHAP is suitable for transformers

2. **Methodology**
   - SHAP explainer type used (GradientExplainer)
   - Number of samples analyzed
   - Features analyzed (weather only)

3. **Results**
   - Include key plots (summary, dependence, temporal)
   - Feature importance rankings table
   - Top 3-5 most influential features

4. **Discussion**
   - Interpret findings (e.g., "Temperature is the most important feature...")
   - Compare with reference paper findings
   - Physical interpretation (why certain features matter)

5. **Limitations**
   - Computational constraints
   - Sample size used
   - SHAP assumptions

### Key Figures for Thesis

**Essential**:
1. SHAP summary beeswarm plot
2. Feature importance bar chart
3. Top 3 feature dependence plots

**Recommended**:
4. Temporal contribution plot
5. Feature interaction heatmap

**Optional**:
6. Temporal heatmap
7. Prediction horizon analysis

## Computational Requirements

- **RAM**: 16 GB recommended (minimum 8 GB)
- **GPU**: CUDA-capable GPU recommended (can run on CPU)
- **Time**: 10-30 minutes depending on:
  - Number of samples
  - Model complexity
  - Hardware (GPU vs CPU)

## Troubleshooting

### Out of Memory Error
- Reduce `--num_samples` (try 200-300)
- Reduce `--background_samples` (try 50)
- Use smaller batch sizes
- Use CPU instead of GPU

### SHAP Explainer Fails
- Script automatically falls back to KernelExplainer
- May be slower but more reliable
- Check PyTorch version compatibility

### No Checkpoint Found
- Train model first using PatchXFormer.sh
- Specify checkpoint path manually
- Check checkpoint directory paths

## References

### Academic Papers

1. **Lundberg & Lee (2017)**
   "A Unified Approach to Interpreting Model Predictions"
   *NeurIPS 2017*
   - Original SHAP paper

2. **Nguyen et al. (2025)**
   "Solar energy prediction through machine learning models: A comparative analysis"
   *PLOS ONE, PMC11695015*
   - Referenced for comparison

3. **Vaswani et al. (2017)**
   "Attention Is All You Need"
   *NeurIPS 2017*
   - Transformer architecture

### Documentation

- SHAP Documentation: https://shap.readthedocs.io/
- SHAP GitHub: https://github.com/slundberg/shap
- PyTorch Documentation: https://pytorch.org/docs/

## Contact & Support

For questions about:
- SHAP implementation: Refer to SHAP documentation
- PatchXFormer model: Check main repository
- This analysis: Review code comments in `shap_patchxformer_analysis.py`

## License

This code is provided for research purposes. Please cite appropriately if used in publications.

## Acknowledgments

- SHAP library by Scott Lundberg
- PatchXFormer architecture inspiration
- Solar power dataset from Sri Lanka

---

**Last Updated**: April 11, 2026
**Version**: 1.0
**Author**: AI-Assisted Research for Master's Thesis
