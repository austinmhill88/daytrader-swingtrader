# ML Training Methodology

## Overview
This document defines the cross-validation methodology, embargo periods, walk-forward schedule, stability tests, and drift detection thresholds for machine learning model training.

## Core Principles

1. **Prevent Overfitting**: Use rigorous validation to ensure models generalize to unseen data
2. **Avoid Leakage**: Prevent information from the future leaking into training data
3. **Test Robustness**: Ensure models perform consistently across different time periods and regimes
4. **Monitor Drift**: Detect when feature distributions or relationships change over time

---

## Cross-Validation Methodology

### Walk-Forward Validation (Primary Method)

**Purpose**: Mimic real-world model retraining and testing schedule

**Parameters**:
- **Train Window**: 252 trading days (~1 year)
- **Test Window**: 63 trading days (~3 months)
- **Step Size**: 21 trading days (~1 month)
- **Embargo**: 1% of train window (~2-3 days)

**Process**:
1. Train model on window [Day 1 to Day 252]
2. Apply embargo [Day 253-254] (do not train or test on these days)
3. Test model on window [Day 255 to Day 317]
4. Step forward by 21 days
5. Train model on window [Day 22 to Day 273]
6. Apply embargo [Day 274-275]
7. Test model on window [Day 276 to Day 338]
8. Repeat until end of data
9. Aggregate all out-of-sample results

**Advantages**:
- Realistic simulation of production retraining
- Prevents look-ahead bias
- Tests robustness across multiple regimes

**Disadvantages**:
- Computationally expensive (many models to train)
- Requires sufficient historical data (recommend ≥3 years)

---

### Purged K-Fold Cross-Validation (Secondary Method)

**Purpose**: Maximize data usage while preventing leakage in time-series data

**Parameters**:
- **K**: 5 or 10 folds
- **Purge**: Remove samples within ±holding_period days of test set
- **Embargo**: Additional 1% buffer after purge

**Purging Logic**:
```python
def purge_training_data(train_indices, test_indices, holding_period):
    """
    Remove training samples that are too close in time to test samples.
    
    If a label uses data from time T to T+h (holding period), then:
    - Remove all training samples in [T-h, T+h] to prevent leakage
    """
    purged_train = []
    test_times = [timestamps[i] for i in test_indices]
    min_test_time = min(test_times)
    max_test_time = max(test_times)
    
    for idx in train_indices:
        sample_time = timestamps[idx]
        # Purge if sample is within holding_period of any test sample
        if sample_time < (min_test_time - holding_period) or \
           sample_time > (max_test_time + holding_period):
            purged_train.append(idx)
    
    # Additional embargo: remove samples immediately after test set
    embargo_size = int(0.01 * len(train_indices))
    purged_train = [idx for idx in purged_train if idx < min(test_indices) - embargo_size]
    
    return purged_train
```

**Example** (5-fold, holding period = 5 days):
```
Fold 1 Test: [Day 1-60]
  Purge: Remove train samples in [Day -4 to Day 65]
  Embargo: Remove train samples in [Day 60 to Day 61]
  Train: [Day 66-300]

Fold 2 Test: [Day 61-120]
  Purge: Remove train samples in [Day 56 to Day 125]
  Train: [Day 1-55] + [Day 126-300]

...
```

**Advantages**:
- Efficient use of limited data
- More CV folds = more robust estimates

**Disadvantages**:
- More complex to implement
- Requires careful handling of time-series dependencies

**Recommendation**: Use purged CV for model selection (hyperparameter tuning), then validate with walk-forward for final promotion decision.

---

## Embargo Periods

### Why Embargo?

In time-series data, adjacent samples are correlated. Even after splitting train/test, information can leak if samples are too close in time.

**Example**:
- Train on [Day 1-100]
- Test on [Day 101]
- If Day 100 and Day 101 have high autocorrelation, model may "cheat" by memorizing Day 100

**Solution**: Add an embargo period (gap) between train and test.

### Embargo Size

**Default**: 1% of training window
- Train window = 252 days → embargo = 2-3 days
- Train window = 500 days → embargo = 5 days

**Holding Period Based**:
- If strategy holding period = H days, embargo = H days
- Example: Swing strategy with 5-day holding → embargo = 5 days

**Recommendation**: Use the larger of 1% or holding_period.

---

## Walk-Forward Retraining Schedule

### Frequency

| Strategy Type | Retraining Frequency | Rationale |
|---------------|----------------------|-----------|
| **Intraday** | Weekly | Faster market dynamics, need fresh models |
| **Swing** | Monthly | Slower dynamics, less frequent retraining |
| **Long-term** | Quarterly | Very slow dynamics |

### Trigger-Based Retraining (Advanced)

Instead of fixed schedule, retrain when:
1. **Performance Degradation**: Live Sharpe < 0.5 for 20 days
2. **Feature Drift**: KS statistic > 0.2 or PSI > 0.3
3. **Regime Change**: Detected shift in volatility or trend regime
4. **Model Age**: Model > 90 days old (fallback)

**Benefits**:
- Retrain only when necessary
- Save computational resources

**Trade-offs**:
- More complex logic
- Requires robust drift detection

---

## Stability Tests

### 1. SHAP Importance Stability Across Folds

**Purpose**: Ensure feature importance is consistent, not random

**Method**:
1. For each CV fold, compute SHAP values for all features
2. Rank features by mean absolute SHAP value
3. Compute coefficient of variation (CV) for top 10 features across folds

**Formula**:
```python
for feature in top_10_features:
    shap_values_across_folds = [fold1_shap, fold2_shap, ..., foldK_shap]
    cv = std(shap_values_across_folds) / mean(shap_values_across_folds)

avg_cv = mean([cv for all top 10 features])
```

**Threshold**: `avg_cv < 0.3`

**Interpretation**:
- Low CV → Feature importance is stable (good)
- High CV → Importance varies wildly (overfitting, bad)

---

### 2. Feature Drift Detection

**Purpose**: Detect when feature distributions change between train and test

**Methods**:

#### A. Kolmogorov-Smirnov (KS) Test
- Compares cumulative distributions of train vs test
- Returns KS statistic (0 to 1) and p-value
- **Threshold**: KS < 0.1 (distributions are similar)

```python
from scipy.stats import ks_2samp

for feature in features:
    ks_stat, p_value = ks_2samp(train[feature], test[feature])
    if ks_stat > 0.1:
        print(f"Warning: Feature {feature} has drifted (KS={ks_stat:.3f})")
```

#### B. Population Stability Index (PSI)
- Measures shift in distribution by bucketing values
- **Formula**: PSI = Σ (p_test - p_train) × ln(p_test / p_train)
- **Threshold**: PSI < 0.2 (stable), 0.2-0.25 (moderate drift), > 0.25 (high drift)

```python
def calculate_psi(train_data, test_data, bins=10):
    train_hist, bin_edges = np.histogram(train_data, bins=bins, density=True)
    test_hist, _ = np.histogram(test_data, bins=bin_edges, density=True)
    
    # Add small epsilon to avoid log(0)
    train_hist = train_hist + 1e-6
    test_hist = test_hist + 1e-6
    
    psi = np.sum((test_hist - train_hist) * np.log(test_hist / train_hist))
    return psi
```

**Recommendation**: Use both KS and PSI. If either breaches threshold, investigate.

---

### 3. Performance Consistency Across Windows

**Purpose**: Ensure model performs well across multiple time periods, not just one lucky window

**Method**:
- In walk-forward validation, count windows with positive returns
- Require at least 70% of windows to be profitable

```python
num_positive_windows = sum([1 for w in windows if w['return'] > 0])
consistency_rate = num_positive_windows / len(windows)

# Threshold: consistency_rate > 0.70
```

**Interpretation**:
- High consistency → Robust model
- Low consistency → Model works only in specific regimes (overfitting)

---

### 4. Monotonic Constraints (Domain Knowledge)

**Purpose**: Enforce logical relationships between features and target

**Examples**:
- Higher volatility → Lower position size (monotonic decrease)
- Higher liquidity → Larger position size (monotonic increase)

**Implementation** (for tree-based models):
```python
import lightgbm as lgb

# Define monotonic constraints
# +1: increasing, -1: decreasing, 0: no constraint
monotone_constraints = [0, 1, -1, 0, 0, ...]  # One per feature

model = lgb.LGBMClassifier(monotone_constraints=monotone_constraints)
```

**Benefits**:
- Prevents nonsensical patterns (e.g., "higher volatility = larger size")
- Improves interpretability
- Reduces overfitting

---

## Drift Detection Thresholds

| Metric | Threshold | Action if Breached |
|--------|-----------|---------------------|
| **KS Statistic** | < 0.1 | Acceptable drift |
|  | 0.1 - 0.2 | Moderate drift, monitor closely |
|  | > 0.2 | High drift, retrain immediately |
| **PSI** | < 0.2 | Stable |
|  | 0.2 - 0.25 | Moderate drift |
|  | > 0.25 | High drift, retrain immediately |
| **SHAP Stability CV** | < 0.3 | Stable importance |
|  | > 0.3 | Unstable, review features |
| **Performance Consistency** | > 70% positive windows | Robust |
|  | 50% - 70% | Marginal |
|  | < 50% | Reject model |

---

## Meta-Labeling (Optional Advanced Technique)

**Purpose**: Use a secondary ML model to filter signals from a primary model or rule-based strategy

**Process**:
1. **Primary Model/Strategy**: Generates buy/sell signals
2. **Meta-Labeling Model**: Predicts whether to take the signal (yes/no)
   - Input features: Same as primary model + signal confidence, market regime, etc.
   - Target: 1 if trade was profitable, 0 if trade was unprofitable

**Benefits**:
- Improves precision (filters out bad signals)
- Reduces false positives
- Can incorporate additional context (regime, risk)

**Trade-offs**:
- More complex
- Requires careful validation (avoid overfitting on meta-labels)

**Recommendation**: Implement after primary models are stable and performing well.

---

## Hyperparameter Tuning

### Grid Search vs Bayesian Optimization

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| **Grid Search** | Simple, exhaustive | Slow for large grids | Small hyperparameter space (<5 params) |
| **Random Search** | Faster than grid | Not exhaustive | Medium space (5-10 params) |
| **Bayesian Optimization** | Efficient, smart | Complex setup | Large space (>10 params) |

**Recommendation**: Start with random search (fast), then use Bayesian optimization for final tuning.

### Hyperparameter Search Example (LightGBM)
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

search = RandomizedSearchCV(
    lgb.LGBMClassifier(),
    param_distributions=param_dist,
    n_iter=50,  # Try 50 random combinations
    cv=purged_cv,  # Use purged CV
    scoring='roc_auc',
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
```

---

## Random Seed Management

**Purpose**: Ensure reproducibility of model training

**Best Practices**:
1. Set seeds for all random libraries:
   ```python
   import random
   import numpy as np
   import lightgbm as lgb
   
   SEED = 42
   random.seed(SEED)
   np.random.seed(SEED)
   lgb.LGBMClassifier(random_state=SEED)
   ```

2. Store seed in model registry for each trained model

3. If using GPU, set GPU seed as well:
   ```python
   import torch
   torch.manual_seed(SEED)
   torch.cuda.manual_seed_all(SEED)
   ```

---

## Implementation Checklist

- [ ] Implement walk-forward validation framework
- [ ] Implement purged k-fold cross-validation
- [ ] Implement embargo logic (1% or holding_period)
- [ ] Compute SHAP values for each CV fold
- [ ] Implement SHAP stability check (CV < 0.3)
- [ ] Implement feature drift detection (KS and PSI)
- [ ] Implement performance consistency check (70% positive windows)
- [ ] Add monotonic constraints to tree models (where applicable)
- [ ] Implement hyperparameter tuning (random or Bayesian)
- [ ] Set and track random seeds for reproducibility
- [ ] Create retraining schedule (weekly/monthly)
- [ ] Implement trigger-based retraining (drift detection)
- [ ] Log all validation metrics to model registry
- [ ] Create Grafana dashboard for drift monitoring

---

## Configuration File Example

```yaml
# config/training_config.yaml
ml_training:
  validation:
    primary_method: "walk_forward"
    secondary_method: "purged_cv"  # For hyperparameter tuning
    
    walk_forward:
      train_window_days: 252
      test_window_days: 63
      step_days: 21
      embargo_pct: 0.01  # 1% of train window
      min_train_samples: 1000
    
    purged_cv:
      n_folds: 5
      holding_period_days: 5  # Strategy-specific
      embargo_pct: 0.01
  
  stability_tests:
    shap_stability:
      enabled: true
      top_n_features: 10
      max_cv: 0.3
    
    feature_drift:
      enabled: true
      methods: ["ks", "psi"]
      ks_threshold: 0.1
      psi_threshold: 0.2
    
    performance_consistency:
      enabled: true
      min_positive_windows_pct: 0.70
  
  hyperparameter_tuning:
    method: "random"  # grid, random, bayesian
    n_iter: 50
    cv_folds: 5
    scoring: "roc_auc"
    
    lightgbm:
      n_estimators: [100, 200, 500]
      max_depth: [3, 5, 7, 10]
      learning_rate: [0.01, 0.05, 0.1]
      num_leaves: [31, 63, 127]
      min_child_samples: [20, 50, 100]
  
  retraining:
    schedule:
      intraday_strategies: "weekly"
      swing_strategies: "monthly"
    
    triggers:
      performance_degradation:
        enabled: true
        sharpe_threshold: 0.5
        lookback_days: 20
      
      feature_drift:
        enabled: true
        ks_threshold: 0.2
        psi_threshold: 0.3
      
      model_age:
        enabled: true
        max_age_days: 90
  
  reproducibility:
    random_seed: 42
    track_seeds: true
    track_library_versions: true
```

---

## Monitoring and Alerts

### Training Job Logs
```json
{
  "job_id": "train_20240115_123045",
  "model_type": "lightgbm",
  "strategy": "intraday_mean_reversion",
  "train_window": "2023-01-01 to 2024-01-01",
  "test_window": "2024-01-01 to 2024-04-01",
  "validation_method": "walk_forward",
  "n_train_samples": 25000,
  "n_test_samples": 6000,
  "oos_sharpe": 1.35,
  "oos_win_rate": 0.52,
  "shap_stability_cv": 0.25,
  "feature_drift_ks_max": 0.08,
  "performance_consistency": 0.75,
  "promotion_decision": "approved"
}
```

### Drift Alerts
```
[FEATURE DRIFT] 2024-01-15
Feature: realized_volatility
KS Statistic: 0.23 (threshold: 0.1)
PSI: 0.31 (threshold: 0.2)
Action: Model retrain triggered
```

---

## References

- **Advances in Financial Machine Learning** by Marcos Lopez de Prado
  - Chapter 7: Cross-Validation in Finance
  - Chapter 8: Feature Importance
  - Chapter 11: The Dangers of Backtesting

- **Purged K-Fold CV**: López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.

- **SHAP Values**: Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NeurIPS.

- **PSI**: Yurdakul, B. (2018). "Statistical properties of population stability index." Western Michigan University.

- **Feature Drift**: Webb, G. I., et al. (2016). "Characterizing concept drift." Data Mining and Knowledge Discovery, 30(4), 964-994.
