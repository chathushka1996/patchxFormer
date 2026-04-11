# Thesis Section Template: Model Interpretability using SHAP

This document provides a template for including SHAP analysis in your thesis.

---

## CHAPTER/SECTION: Model Interpretability

### Section 1: Introduction to Model Interpretability

**Why Interpretability Matters in Solar Power Forecasting**

While deep learning models like transformers achieve high predictive accuracy, their "black-box" nature limits understanding of decision-making processes. For solar power forecasting applications, interpretability is crucial for:

1. **Trust and Adoption**: Stakeholders need to understand why the model makes certain predictions
2. **Physical Validation**: Ensuring predictions align with known meteorological relationships
3. **Debugging**: Identifying when models rely on spurious correlations
4. **Improvement**: Understanding which features to prioritize for data collection
5. **Regulatory Compliance**: Explainable AI requirements in energy sector applications

---

### Section 2: SHAP Methodology

#### 2.1 Theoretical Background

SHAP (SHapley Additive exPlanations) [Lundberg & Lee, 2017] provides a unified framework for interpreting machine learning model predictions based on cooperative game theory. The method assigns each feature an importance value (Shapley value) for a particular prediction, satisfying key theoretical properties:

**Mathematical Formulation:**

The SHAP value φᵢ for feature i is defined as:

```
φᵢ = Σ_{S⊆F\{i}} [|S|!(|F|-|S|-1)! / |F|!] × [f_S∪{i}(x_S∪{i}) - f_S(x_S)]
```

Where:
- F is the set of all features
- S is a subset of features excluding feature i
- f_S(x_S) is the model prediction using only features in subset S
- The summation is over all possible subsets S

**Key Properties:**

1. **Efficiency**: Σφᵢ = f(x) - E[f(X)]
   - The sum of all SHAP values equals the difference between the prediction and the expected value

2. **Symmetry**: If features i and j contribute equally, φᵢ = φⱼ
   - Equal contributions receive equal values

3. **Dummy**: If feature i has no impact, φᵢ = 0
   - Irrelevant features receive zero importance

4. **Additivity**: For ensemble models, SHAP values combine linearly
   - φᵢ(f + g) = φᵢ(f) + φᵢ(g)

#### 2.2 SHAP for Transformer Models

For transformer-based architectures like PatchXFormer, we employ **GradientExplainer**, which:

1. Leverages backpropagation gradients efficiently computed by transformers
2. Handles attention mechanisms and complex feature interactions
3. Provides accurate approximations with lower computational cost than kernel methods

**Rationale for GradientExplainer:**
- Transformer models are differentiable → gradient-based methods are efficient
- Self-attention creates complex feature interactions → need sophisticated explainer
- Time series requires temporal understanding → gradients capture temporal dependencies

#### 2.3 Implementation Details

**Analysis Configuration:**
- **Background Dataset**: 100 samples from test set
  - Represents typical model operating conditions
  - Used to establish baseline predictions

- **Explanation Dataset**: 200 samples from test set
  - Diverse range of weather conditions
  - Representative of different times of day/year

- **Features Analyzed**: 7 weather variables
  - Temperature (°C)
  - Dew Point (°C)
  - Humidity (%)
  - Wind Direction (degrees)
  - Wind Speed (m/s)
  - Pressure (hPa)
  - Cloud Cover (%)

- **Excluded Features**: Time features (day of year, time of day)
  - Focus on meteorological influences only
  - Time effects captured through temporal encoding in model

**Computational Setup:**
- Hardware: [SPECIFY: GPU/CPU, RAM]
- Analysis Duration: [SPECIFY: e.g., ~20 minutes]
- SHAP Library Version: 0.42.0
- PyTorch Version: [SPECIFY YOUR VERSION]

---

### Section 3: Results

#### 3.1 Global Feature Importance

**Table X: Weather Feature Importance Rankings**

| Rank | Feature       | Mean \|SHAP\| | Interpretation |
|------|---------------|---------------|----------------|
| 1    | [FEATURE_1]   | [VALUE]       | Most influential |
| 2    | [FEATURE_2]   | [VALUE]       | Second most important |
| 3    | [FEATURE_3]   | [VALUE]       | Third most important |
| 4    | [FEATURE_4]   | [VALUE]       | Moderate importance |
| 5    | [FEATURE_5]   | [VALUE]       | Lower importance |
| 6    | [FEATURE_6]   | [VALUE]       | Minimal impact |
| 7    | [FEATURE_7]   | [VALUE]       | Least influential |

*Note: Higher Mean |SHAP| indicates greater impact on model predictions.*

**Key Findings:**

[REPLACE WITH YOUR RESULTS]
Example:
> The SHAP analysis revealed that **temperature** (Mean |SHAP| = 0.XXX) is the most influential weather variable for solar power predictions, followed by **humidity** (Mean |SHAP| = 0.XXX) and **cloud cover** (Mean |SHAP| = 0.XXX). This aligns with physical understanding of solar energy generation, where temperature affects panel efficiency, humidity influences atmospheric transparency, and cloud cover directly blocks solar radiation.

#### 3.2 Feature Contribution Patterns

**Figure X: SHAP Summary Plot**

[INSERT: shap_summary_beeswarm.png]

*Caption: SHAP summary plot showing the distribution of feature contributions for 200 test samples. Each point represents a sample, with horizontal position indicating SHAP value (impact on prediction), color representing feature value (red=high, blue=low), and vertical position showing the feature. Features are ordered by importance.*

**Interpretation:**

[EXAMPLE - REPLACE WITH YOUR RESULTS]
> The beeswarm plot reveals several key patterns:
> 
> 1. **Temperature**: Shows strong positive correlation - higher temperatures (red) generally have positive SHAP values, increasing predicted solar power. This reflects that solar panels operate more efficiently in moderate-to-warm conditions.
>
> 2. **Humidity**: Exhibits negative correlation - higher humidity (red) tends to have negative SHAP values, decreasing predicted solar power. High humidity scatters incoming solar radiation, reducing energy capture.
>
> 3. **Cloud Cover**: Demonstrates clear negative impact - increased cloud cover consistently reduces predicted solar power, as expected from physical principles.

#### 3.3 Feature Dependence Analysis

**Figure Y: SHAP Dependence Plots**

[INSERT: shap_dependence_plots.png OR individual plots for top 3 features]

*Caption: SHAP dependence plots for the three most important weather features. Each plot shows how feature values (x-axis) affect model predictions (y-axis, SHAP value). The trend line indicates the average relationship, while scatter shows variation across samples.*

**Detailed Analysis:**

[EXAMPLE - REPLACE WITH YOUR RESULTS]

**Temperature Dependence:**
> The temperature dependence plot reveals a non-linear relationship with solar power predictions. SHAP values increase linearly for temperatures between 15-30°C, plateau at 30-35°C, and decline slightly above 35°C. This U-shaped pattern aligns with known solar panel physics: moderate temperatures optimize efficiency, but excessive heat (>35°C) reduces photovoltaic conversion efficiency due to increased electrical resistance.

**Humidity Dependence:**
> Humidity shows a monotonic negative relationship with predictions. SHAP values decrease steadily from low (<40%) to high (>80%) humidity. The relationship is approximately linear, suggesting consistent atmospheric scattering effects across the humidity range. High variance at extreme humidity levels indicates interactions with other meteorological variables.

**Cloud Cover Dependence:**
> Cloud cover exhibits the strongest and most consistent negative impact. SHAP values sharply decrease from 0-50% cloud cover, then stabilize at high coverage (>75%). This suggests a saturation effect: once cloud coverage exceeds ~50%, additional clouds have diminishing marginal impact as solar radiation is already substantially blocked.

#### 3.4 Temporal Feature Importance

**Figure Z: Temporal Contribution Analysis**

[INSERT: shap_temporal_contribution.png]

*Caption: Evolution of feature importance across the 96-timestep input sequence. Lines show mean absolute SHAP values at each time step, revealing which features contribute most at different points in the historical window.*

**Findings:**

[EXAMPLE - REPLACE WITH YOUR RESULTS]
> Temporal analysis reveals that:
>
> 1. **Recent timesteps dominate**: Features from the most recent 24 time steps (last quarter of sequence) have 2-3× higher SHAP values than earlier timesteps, indicating the model primarily relies on recent weather conditions.
>
> 2. **Temperature exhibits long-term influence**: Unlike other features, temperature maintains relatively consistent importance across the entire sequence, suggesting thermal inertia effects.
>
> 3. **Cloud cover is most important recently**: Cloud cover importance increases sharply in the final 12 timesteps, implying immediate cloud conditions strongly predict imminent solar power.

#### 3.5 Feature Interactions

**Figure W: Feature Interaction Heatmap**

[INSERT: shap_feature_interactions.png]

*Caption: Correlation matrix of SHAP values between features. High correlations indicate features that jointly influence predictions.*

**Analysis:**

[EXAMPLE - REPLACE WITH YOUR RESULTS]
> The interaction analysis reveals:
>
> - **Humidity-Temperature**: Strong negative correlation (r = -0.XX), indicating these features work antagonistically in predictions
> - **Cloud Cover-Humidity**: Moderate positive correlation (r = 0.XX), suggesting correlated weather phenomena
> - **Temperature-Pressure**: Weak correlation (r = 0.XX), implying largely independent contributions

---

### Section 4: Discussion

#### 4.1 Comparison with Reference Study

[COMPARE WITH PMC11695015 PAPER]

Example:
> Our SHAP analysis of the PatchXFormer model reveals results partially consistent with Nguyen et al. (2025), who analyzed solar energy predictions using CatBoost. Both studies identified **temperature** and **humidity** as the most influential weather variables. However, notable differences exist:
>
> **Similarities:**
> 1. Temperature ranked as most important feature (PatchXFormer: Rank 1, CatBoost: Rank 1)
> 2. Humidity ranked highly (PatchXFormer: Rank 2, CatBoost: Rank 2)
> 3. Negative correlation between humidity and solar power
>
> **Differences:**
> 1. **Cloud cover importance**: PatchXFormer assigned higher importance to cloud cover (Rank 3 vs Rank 5 in CatBoost study)
> 2. **Wind speed**: Lower importance in PatchXFormer (Rank 5) compared to CatBoost (Rank 3)
> 3. **Non-linear relationships**: PatchXFormer captured more complex non-linear dependencies, particularly for temperature
>
> These differences likely stem from:
> - **Model architecture**: Transformers capture long-range temporal dependencies better than gradient boosting
> - **Dataset differences**: Sri Lanka vs. Northern Hemisphere locations have different climate patterns
> - **Temporal resolution**: Our 15-minute intervals vs. hourly data in reference study

#### 4.2 Physical Interpretation

**Consistency with Solar Physics:**

[RELATE FINDINGS TO PHYSICAL PRINCIPLES]

Example:
> The SHAP analysis results align well with established solar energy physics:
>
> 1. **Temperature Effects**: The observed U-shaped temperature relationship matches the Shockley-Queisser limit for photovoltaic efficiency. Solar cells exhibit maximum efficiency at ~25°C, with degradation at higher temperatures due to increased dark current and reduced open-circuit voltage.
>
> 2. **Humidity Impact**: The negative humidity-power relationship reflects atmospheric scattering principles. Water vapor molecules scatter incoming shortwave radiation (Rayleigh scattering), reducing direct normal irradiance (DNI) reaching solar panels.
>
> 3. **Cloud Cover**: The strong negative impact and saturation effect match radiative transfer theory. Optically thick clouds block >90% of direct solar radiation, explaining the diminishing marginal effect beyond 50% coverage.
>
> 4. **Pressure**: Lower importance aligns with expectations - pressure primarily affects atmospheric density, which has secondary effects on solar radiation compared to direct attenuators like clouds and humidity.

#### 4.3 Model Insights

**What SHAP Reveals About PatchXFormer:**

[INSIGHTS INTO YOUR MODEL'S BEHAVIOR]

Example:
> The SHAP analysis provides several insights into PatchXFormer's predictive strategy:
>
> 1. **Attention to Recent History**: The temporal analysis confirms that the self-attention mechanism effectively identifies and weights recent weather conditions more heavily, consistent with the physical reality that near-term conditions are most predictive.
>
> 2. **Non-linear Feature Encoding**: The complex dependence plots demonstrate that the transformer successfully learned non-linear meteorological relationships without explicit feature engineering.
>
> 3. **Feature Interactions**: The interaction heatmap suggests the model implicitly models correlated weather phenomena (e.g., humidity-temperature anticorrelation) through its attention mechanism.
>
> 4. **Robustness**: The consistent feature rankings across different samples indicate the model has learned generalizable patterns rather than overfitting to specific instances.

#### 4.4 Limitations of SHAP Analysis

**Important Considerations:**

1. **Independence Assumption**: SHAP assumes feature independence when marginalizing, but weather variables are inherently correlated (e.g., temperature-humidity)

2. **Sampling**: Analysis based on 200 samples from test set; conclusions may not fully generalize

3. **Computational Approximations**: GradientExplainer uses approximations; exact Shapley values are computationally intractable

4. **Time Feature Exclusion**: Excluding temporal features may conflate their effects with weather variables

5. **Model-Specific**: Results specific to PatchXFormer; different architectures may learn different feature dependencies

---

### Section 5: Implications

#### 5.1 For Model Development

Based on SHAP insights:

1. **Data Collection Priorities**: Focus on high-quality measurements of temperature, humidity, and cloud cover
2. **Feature Engineering**: Invest in better cloud detection systems and humidity sensors
3. **Model Refinement**: Consider explicit temperature non-linearity modeling
4. **Ensemble Opportunities**: Combine with physical models for high-temperature scenarios

#### 5.2 For Operational Deployment

1. **Sensor Reliability**: Ensure backup sensors for critical features (temperature, humidity, cloud cover)
2. **Real-time Monitoring**: Alert operators when critical features reach extreme values
3. **Uncertainty Quantification**: Higher prediction uncertainty expected when low-importance features dominate
4. **Calibration**: Periodic recalibration focusing on high-SHAP features

#### 5.3 For Future Research

1. **Multi-site Analysis**: Compare feature importance across different geographical locations
2. **Seasonal Variation**: Investigate if feature importance changes seasonally
3. **Extreme Weather**: Special analysis for typhoons, heatwaves, monsoons
4. **Causal Analysis**: Beyond correlation, establish causal relationships using interventional SHAP

---

### Section 6: Conclusion

[SUMMARIZE KEY POINTS]

Example:
> This chapter applied SHAP analysis to interpret the PatchXFormer solar power forecasting model, revealing that temperature, humidity, and cloud cover are the most influential meteorological variables. The analysis demonstrated that the transformer architecture successfully learns physically meaningful relationships, including non-linear temperature effects and humidity-based atmospheric scattering. These findings validate the model's decision-making process and provide actionable insights for operational deployment and future research directions.
>
> Compared to previous work using gradient boosting (Nguyen et al., 2025), the transformer-based approach captures more complex temporal dependencies and non-linear relationships, while maintaining physically interpretable feature importance rankings. The SHAP framework proved effective for explaining transformer predictions in time series forecasting contexts, addressing the "black-box" critique of deep learning models in critical energy infrastructure applications.

---

## References to Include

```
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting 
model predictions. Advances in neural information processing systems, 30.

Nguyen, H. N., Tran, Q. T., Ngo, C. T., Nguyen, D. D., & Tran, V. Q. (2025). 
Solar energy prediction through machine learning models: A comparative analysis 
of regressor algorithms. PLOS ONE, 20(1), e0315955.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., 
... & Polosukhin, I. (2017). Attention is all you need. Advances in neural 
information processing systems, 30.

[Add other relevant references from your thesis]
```

---

## Figures Checklist

Make sure to include these figures in your thesis:

- [ ] SHAP summary beeswarm plot (most important)
- [ ] Feature importance bar chart
- [ ] Top 3 feature dependence plots
- [ ] Temporal contribution plot
- [ ] Feature interaction heatmap (optional but recommended)

## Tables Checklist

- [ ] Feature importance rankings table
- [ ] Comparison with reference paper
- [ ] Statistical summary of SHAP values
- [ ] Model configuration parameters

---

**Note**: Replace all [BRACKETS] and example text with your actual results from the SHAP analysis. The structure and theoretical content can be used as-is, but numerical results and interpretations must come from your specific analysis outputs.
