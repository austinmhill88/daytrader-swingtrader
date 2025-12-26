# Model Governance and Promotion Gates

## Overview
This document defines the model registry, validation methodology, promotion gates, and audit requirements to prevent overfitting and ensure production-ready models.

## Model Registry

### Purpose
A centralized repository to track all trained models, their metadata, performance metrics, validation results, and promotion status.

### Registry Implementation
**Primary Choice**: MLflow Tracking Server

**Alternative**: Local filesystem-based registry (for simpler deployments)

### Registry Fields

For each registered model, store:

#### Model Metadata
- **Model ID**: Unique identifier (UUID)
- **Model Type**: lightgbm, xgboost, random_forest
- **Strategy**: intraday_mean_reversion, swing_trend_following
- **Created At**: Training timestamp (UTC)
- **Created By**: System or user who triggered training
- **Git Commit Hash**: Code version used for training
- **Config Hash**: Hash of configuration file

#### Training Data
- **Dataset Hash**: SHA-256 of training data snapshot
- **Dataset Path**: Location of immutable snapshot
- **Date Range**: Start and end dates of training data
- **Symbol Universe**: List of symbols used
- **Row Count**: Number of training samples
- **Feature Version**: Hash of feature engineering code

#### Model Artifacts
- **Model File**: Path to serialized model (.pkl, .joblib, .txt)
- **Feature List**: Ordered list of features used
- **Hyperparameters**: Complete hyperparameter dictionary
- **Feature Importance**: Top 20 features by importance
- **SHAP Values**: Summary statistics (optional, large files)

#### Validation Methodology
- **Validation Method**: purged_cv, walk_forward, simple_split
- **Train Window Days**: Training period length
- **Test Window Days**: Testing period length
- **Embargo Days**: Embargo period to prevent leakage
- **Cross-Validation Folds**: Number of CV folds (if using purged CV)
- **Random Seed**: For reproducibility

#### Performance Metrics

##### In-Sample (Training Set)
- Accuracy, Precision, Recall, F1 Score, AUC-ROC (for classification)
- RMSE, MAE, R² (for regression)
- Sharpe Ratio, Sortino Ratio (if backtested on training set)

##### Out-of-Sample (Test Set)
- Same metrics as in-sample
- **Key Requirement**: OOS performance must meet promotion gates

##### Backtest Metrics (if strategy backtested)
- Total Return (%)
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown (%)
- Calmar Ratio
- Win Rate (%)
- Average Win / Average Loss
- Total Trades
- Profit Factor
- Turnover (% of capital per period)
- Average Holding Period (bars or days)

#### Anti-Overfitting Diagnostics

1. **Probability of Backtest Overfitting (PBO)**
   - Metric: 0 to 1 (lower is better)
   - Interpretation: Probability that performance is due to overfitting
   - Threshold: PBO < 0.5 (prefer < 0.3)

2. **Deflated Sharpe Ratio (DSR)**
   - Adjusts Sharpe for multiple testing and non-normality
   - Interpretation: Statistically significant Sharpe after adjustments
   - Threshold: DSR > 1.0 (conservative), > 0.5 (moderate)

3. **White's Reality Check**
   - Tests if strategy outperforms benchmark after data snooping
   - Interpretation: P-value for statistical significance
   - Threshold: P-value < 0.05 (strategy beats benchmark)

4. **SHAP Stability Across Folds**
   - Measure: Variance of SHAP importance across CV folds
   - Interpretation: Consistent feature importance indicates robustness
   - Threshold: Coefficient of variation < 0.3 for top 10 features

5. **Feature Drift Detection**
   - Measure: KS statistic or PSI between train and test distributions
   - Interpretation: Significant drift may indicate regime change
   - Threshold: KS < 0.1 or PSI < 0.2

#### Promotion Status
- **Status**: pending, approved, rejected, retired
- **Promoted At**: Timestamp of promotion (if approved)
- **Promoted By**: Automated system or manual approver
- **Rejection Reason**: If rejected, which gate(s) failed
- **Production Deployment Date**: When model went live (if applicable)
- **Retired Date**: When model was retired from production

#### Audit Trail
- **Logs**: All operations (register, promote, reject, retire)
- **Decisions**: Automated gate decisions and manual overrides
- **Performance Tracking**: Live performance vs backtested metrics

---

## Validation Methodology

### Walk-Forward Validation (Primary)

**Purpose**: Simulate realistic out-of-sample performance over rolling windows.

**Process**:
1. **Training Window**: 252 days (1 trading year)
2. **Testing Window**: 63 days (3 months)
3. **Step Size**: 21 days (1 month)
4. **Embargo**: 1% of training data (~2-3 days) between train and test

**Example Timeline**:
```
Window 1: Train [Day 1 - 252], Embargo [253-254], Test [255-317]
Window 2: Train [Day 22 - 273], Embargo [274-275], Test [276-338]
Window 3: Train [Day 43 - 294], Embargo [295-296], Test [297-359]
...
Aggregate all out-of-sample results
```

**Advantages**:
- Mimics real-world model retraining schedule
- Prevents look-ahead bias
- Tests robustness across different market regimes

**Output**:
- Aggregate OOS Sharpe, Sortino, Max DD, Win Rate
- Performance by regime (bull, bear, sideways)
- Consistency of performance across windows

---

### Purged K-Fold Cross-Validation (Secondary)

**Purpose**: Maximize data usage while preventing leakage in time-series data.

**Process**:
1. **Purging**: Remove samples close in time to test set from training set
2. **Embargo**: Additional buffer period after purge
3. **K-Folds**: Typically 5 or 10 folds
4. **Combinatorial Purged CV (CPCV)**: Evaluate all combinations of test folds

**Purge Logic**:
- If a sample's label uses data from time T to T+h (holding period), remove all training samples in range [T-h, T+h] to prevent leakage.

**Example** (3-fold, holding period = 5 days):
```
Fold 1 Test: [Day 1-100]
  Purge: [Day -5 to Day 105]
  Train: [Day 106-300]

Fold 2 Test: [Day 101-200]
  Purge: [Day 96 to Day 205]
  Train: [Day 1-95] + [Day 206-300]

Fold 3 Test: [Day 201-300]
  Purge: [Day 196 to Day 300]
  Train: [Day 1-195]
```

**Advantages**:
- More efficient use of limited data
- Rigorous leakage prevention

**Disadvantages**:
- More complex than walk-forward
- May not capture sequential regime changes as well

**Recommended Use**:
- Use alongside walk-forward for model selection
- Walk-forward for final promotion decision

---

## Promotion Gates

Models must pass **all** of the following gates to be promoted to production:

### Gate 1: Out-of-Sample Performance Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Sortino Ratio** | > 1.0 | Focus on downside risk; Sharpe alone can be gamed with high-frequency noise |
| **Max Drawdown** | < 10% | Limit worst-case loss; adjust based on strategy type |
| **Sharpe Ratio** | > 1.0 | Traditional risk-adjusted return; used as secondary check |
| **Win Rate** | > 45% | Avoid strategies with too many small losses |
| **Total Trades** | > 100 | Ensure sufficient sample size for statistical significance |
| **Profit Factor** | > 1.3 | Gross profit / Gross loss > 1.3 |
| **Calmar Ratio** | > 1.5 | Return / Max DD > 1.5 (for swing strategies) |

**Notes**:
- Thresholds may vary by strategy type (intraday vs swing)
- Adjust based on market conditions (e.g., relax in low-volatility regimes)
- Monitor distribution of metrics over time and recalibrate gates

---

### Gate 2: Anti-Overfitting Diagnostics

| Diagnostic | Threshold | Rationale |
|------------|-----------|-----------|
| **PBO** | < 0.5 | Probability of overfitting < 50%; prefer < 0.3 for high confidence |
| **Deflated Sharpe Ratio** | > 0.5 | Statistically significant after accounting for multiple testing |
| **White's Reality Check** | P-value < 0.05 | Strategy outperforms benchmark with 95% confidence |

**Implementation**:
- Compute PBO using combinatorial symmetric cross-validation (CSCV)
- Calculate DSR using formula from Lopez de Prado (2018)
- Run White's Reality Check against buy-and-hold benchmark (SPY)

---

### Gate 3: Stability and Consistency

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| **SHAP Stability** | CV < 0.3 | Coefficient of variation of top-10 feature importance across folds |
| **Performance Stability** | Positive OOS return in > 70% of windows | Avoid strategies that only work in specific regimes |
| **Feature Drift** | KS < 0.1 or PSI < 0.2 | Ensure training and test feature distributions are similar |

**SHAP Stability Calculation**:
```python
# For each of top 10 features, compute CV across K folds
cv = std(shap_importance) / mean(shap_importance)
# Average CV across all top 10 features must be < 0.3
```

**Performance Stability**:
- In walk-forward validation, count windows with positive return
- Require at least 70% of windows to be profitable

---

### Gate 4: Minimum Sample Size

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Total OOS Trades** | > 100 | Ensure statistical significance |
| **OOS Days** | > 63 | At least one full walk-forward test window |
| **Symbols Traded** | > 10 | Diversification across instruments |

**Notes**:
- For intraday strategies, require higher trade counts (> 200)
- For swing strategies, lower counts acceptable if holding periods are longer

---

### Gate 5: Risk-Adjusted Metrics

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Tail Risk** | 95th percentile loss < 2% per trade | Avoid tail events |
| **Exposure Utilization** | < 80% of max allowed | Leave buffer for unexpected opportunities |
| **Turnover** | < 500% annual for swing, < 2000% for intraday | Control transaction costs |

---

## Promotion Workflow

### Step 1: Model Training
1. User or scheduler triggers training job
2. ML Trainer loads data snapshot, trains model, runs validation
3. Model artifacts and metrics logged to registry with status = "pending"

### Step 2: Automated Gate Evaluation
1. System evaluates all promotion gates
2. Results logged to registry
3. Status updated:
   - All gates pass → status = "approved"
   - Any gate fails → status = "rejected" with reasons

### Step 3: Manual Review (Optional)
1. If automated approval, notify admins (Slack/email)
2. Admins can review and override decision (e.g., reject false positive)
3. Manual override logged to audit trail

### Step 4: Production Deployment
1. Approved models copied to production directory
2. Strategy configuration updated to use new model
3. Old model marked as "retired" (but kept for audit)
4. Deployment logged with timestamp and deployer

### Step 5: Live Monitoring
1. Track live performance (Sharpe, DD, win rate)
2. Compare live vs backtested metrics
3. Trigger alerts if live performance degrades significantly
4. Automatic demotion if live performance fails gates for N consecutive days

---

## Model Demotion

### Triggers
- **Live Sharpe < 0.5** for 20 consecutive trading days
- **Live Max DD > 15%** (exceeds backtest threshold + buffer)
- **Significant feature drift**: KS > 0.2 or PSI > 0.3
- **Manual demotion**: Admin decision

### Demotion Workflow
1. Pause strategy using the model
2. Alert admins
3. Mark model as "demoted" in registry
4. Revert to previous approved model (if available)
5. Investigate root cause (regime change, data quality, overfitting)
6. Retrain and re-validate before re-promotion

---

## Model Retraining Schedule

### Frequency
- **Intraday Strategies**: Retrain weekly (market conditions change faster)
- **Swing Strategies**: Retrain monthly (longer holding periods, slower drift)
- **On-Demand**: Retrain if performance degrades or market regime changes

### Retraining Process
1. Fetch latest data snapshot
2. Retrain model with same hyperparameters (or re-optimize if needed)
3. Run validation and gates
4. If approved, replace production model
5. Log transition and archive old model

---

## Audit and Compliance

### Audit Requirements
- **Model Lineage**: Full trace from data snapshot to production deployment
- **Decision Logs**: All promotion, rejection, demotion decisions with timestamps
- **Performance Tracking**: Compare backtested vs live metrics
- **Reproducibility**: Ability to reproduce any historical model from snapshot + code

### Compliance Considerations
- **SEC/FINRA**: No specific rules for algo trading models, but must maintain audit trails
- **Internal Policy**: Document all model changes; review quarterly
- **Risk Committee**: Monthly review of model performance and gate breaches

### Audit Reports (Quarterly)
- Number of models trained, approved, rejected
- Common rejection reasons (which gates failed most often)
- Live vs backtested performance comparison
- Feature drift incidents
- Demotion events and resolutions

---

## Implementation Checklist

- [ ] Set up MLflow tracking server (or local registry)
- [ ] Implement model registration in `ml_trainer.py`
- [ ] Implement walk-forward validation in `backtest/walk_forward.py`
- [ ] Implement purged CV (optional, for robustness)
- [ ] Compute anti-overfitting diagnostics (PBO, DSR, White's)
- [ ] Implement SHAP stability checks
- [ ] Implement feature drift detection (KS, PSI)
- [ ] Implement promotion gate evaluation logic
- [ ] Set up automated promotion workflow
- [ ] Set up live performance monitoring
- [ ] Implement demotion triggers and workflow
- [ ] Create Slack/email alerts for promotions and demotions
- [ ] Document manual override procedures
- [ ] Create quarterly audit report template

---

## References

- **Advances in Financial Machine Learning** by Marcos Lopez de Prado
  - Chapter 7: Cross-Validation in Finance
  - Chapter 11: The Dangers of Backtesting
  - Chapter 14: Backtesting through Cross-Validation

- **Deflated Sharpe Ratio**: Bailey, D. H., & López de Prado, M. (2014). "The Sharpe ratio efficient frontier." Journal of Risk, 15(2), 3-44.

- **Probability of Backtest Overfitting**: Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). "Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance." Notices of the AMS, 61(5), 458-471.

- **White's Reality Check**: White, H. (2000). "A reality check for data snooping." Econometrica, 68(5), 1097-1126.

- **SHAP (SHapley Additive exPlanations)**: Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NeurIPS.

- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
