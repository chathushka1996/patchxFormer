# Feature Importance Analysis for Weather Parameters

This guide explains how to analyze the importance of each weather parameter in predicting Solar Power Output.

## Quick Start (Simple Analysis)

The simplest way to analyze feature importance is using the standalone script that doesn't require a trained model:

```bash
python analyze_feature_importance_simple.py --root_path "./dataset/sl_piliyandala"
```

### Required Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## What the Analysis Provides

The script performs three types of analysis:

### 1. **Correlation Analysis**
- Calculates Pearson correlation coefficient between each weather parameter and Solar Power Output
- Shows statistical significance (p-values)
- Identifies which parameters have strong linear relationships with the target

### 2. **Mutual Information Analysis**
- Measures non-linear relationships between features and target
- Captures dependencies that correlation might miss
- Useful for identifying complex patterns

### 3. **Combined Ranking**
- Normalizes both correlation and mutual information scores
- Provides an overall importance ranking
- Helps identify the most critical weather parameters

## Output Files

The script generates:

1. **feature_importance_correlation.png** - Bar chart showing correlation scores
2. **feature_importance_mutual_info.png** - Bar chart showing mutual information scores
3. **feature_importance_combined.png** - Combined visualization of all metrics
4. **feature_importance_correlation_matrix.png** - Heatmap of correlations between all features
5. **feature_importance_summary.csv** - Detailed results in CSV format

## Advanced Analysis (Requires Trained Model)

For more advanced analysis using permutation importance (which requires a trained model), use:

```bash
python analyze_feature_importance.py \
    --root_path "./dataset/sl_piliyandala" \
    --model_path "./path/to/trained/model.pth" \
    --seq_len 96 \
    --pred_len 96
```

### Permutation Importance
- Shuffles each feature and measures performance degradation
- More accurate for model-specific importance
- Requires a trained model checkpoint

## Interpreting Results

### Correlation Scores
- **High positive correlation (>0.5)**: Strong positive relationship
- **High negative correlation (<-0.5)**: Strong negative relationship
- **Low correlation (<0.3)**: Weak relationship

### Mutual Information Scores
- **Higher scores**: More information shared with target
- **Lower scores**: Less predictive power

### Average Score
- Combines both metrics for overall importance
- **Rank 1**: Most important feature
- **Rank N**: Least important feature

## Example Usage

```python
from analyze_feature_importance_simple import analyze_feature_importance

# Analyze your dataset
results = analyze_feature_importance(
    root_path='./dataset/sl_piliyandala',
    target='Solar Power Output'
)

# Access results
for param, scores in results.items():
    print(f"{param}: {scores['average_score']:.4f}")
```

## Weather Parameters Analyzed

Based on your dataset, the following 7 weather parameters are analyzed:

1. **temp** - Temperature
2. **dew** - Dew point
3. **humidity** - Humidity
4. **winddir** - Wind direction
5. **windspeed** - Wind speed
6. **pressure** - Atmospheric pressure
7. **cloudcover** - Cloud cover

## Tips

1. **Start with simple analysis**: Use `analyze_feature_importance_simple.py` first
2. **Check statistical significance**: Focus on features with p-value < 0.05
3. **Look at combined scores**: Average score gives the best overall picture
4. **Consider domain knowledge**: Some features might be important despite low scores
5. **Use for feature selection**: Remove low-importance features to reduce model complexity

## Troubleshooting

### ModuleNotFoundError
Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### File not found
Make sure the dataset path is correct and contains `train.csv`

### Memory issues
If dataset is very large, consider sampling:
```python
train_df = pd.read_csv(train_path).sample(n=10000)  # Use sample
```

## Next Steps

After identifying important features:

1. **Feature Engineering**: Create new features from important ones
2. **Model Training**: Focus on top-ranked features
3. **Ablation Study**: Train models with/without specific features
4. **Domain Validation**: Verify results make physical sense

