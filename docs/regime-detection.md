# Regime Detection and Market Regime Sizing

## Overview
This document defines the regime detection methodology, regime states, transition logic, and how each regime impacts strategy activation and position sizing.

## Purpose of Regime Detection

Market conditions change over time, and strategies that work well in one regime may perform poorly in another. Regime detection allows the system to:
1. **Adapt position sizing** based on market conditions
2. **Enable/disable strategies** that are regime-dependent
3. **Adjust risk limits** dynamically
4. **Improve risk-adjusted returns** by avoiding unfavorable periods

---

## Regime Definitions

### Volatility Regimes

| Regime | Realized Volatility (Annualized) | VIX Proxy | Description |
|--------|----------------------------------|-----------|-------------|
| **Low Volatility** | < 10% | VIX < 15 | Calm markets, low risk, mean-reversion favored |
| **Medium Volatility** | 10% - 20% | VIX 15-25 | Normal conditions, balanced strategies |
| **High Volatility** | > 20% | VIX > 25 | Elevated risk, trend-following may dominate, reduce size |

**Calculation**:
- Use 20-day rolling realized volatility of SPY returns
- Annualize: RV = std(returns) × sqrt(252)
- Alternatively, use VIX index if available

---

### Trend Regimes (Market Direction)

| Regime | SPY/QQQ Trend | Slope of 50-day SMA | Description |
|--------|---------------|---------------------|-------------|
| **Bull Market** | Price > SMA(50) | Positive slope | Uptrend, favor longs, momentum strategies |
| **Bear Market** | Price < SMA(50) | Negative slope | Downtrend, favor shorts, defensive positioning |
| **Sideways/Choppy** | Price oscillates around SMA(50) | Flat slope | Range-bound, mean-reversion strategies |

**Calculation**:
- Track SPY and QQQ daily closes
- Compute 50-day SMA
- Slope = (SMA_today - SMA_20_days_ago) / 20
- Regime = Bull if price > SMA and slope > 0.001, Bear if price < SMA and slope < -0.001, else Sideways

---

### Breadth Regimes (Market Internal Strength)

| Regime | Advance/Decline Ratio | New Highs / New Lows | Description |
|--------|------------------------|----------------------|-------------|
| **Strong Breadth** | > 1.5 (7 of last 10 days) | New Highs >> New Lows | Broad participation, healthy market |
| **Weak Breadth** | < 0.67 (7 of last 10 days) | New Lows >> New Highs | Narrow leadership, fragile market |
| **Neutral Breadth** | Between 0.67 and 1.5 | Balanced | Mixed signals |

**Data Source**:
- NYSE or Nasdaq Advance/Decline line (if available from data provider)
- Alternatively, compute percentage of stocks in universe above their 50-day SMA

---

### Correlation Regimes (Diversification)

| Regime | Average Pairwise Correlation | Description |
|--------|------------------------------|-------------|
| **Low Correlation** | < 0.3 | High diversification, stock-picking works well |
| **Medium Correlation** | 0.3 - 0.6 | Normal conditions |
| **High Correlation** | > 0.6 | Risk-off, all stocks move together, harder to diversify |

**Calculation**:
- Compute 20-day rolling correlation matrix for universe (or top 100 stocks)
- Average of all pairwise correlations
- High correlation often occurs during market crashes (2008, 2020)

---

## Combined Regime State

Combine the above dimensions into a single regime state:

### Example States

1. **Risk-On (Bull + Low Vol + Strong Breadth + Low Correlation)**
   - Increase position sizes
   - Enable aggressive strategies (momentum, breakout)
   - Expand universe to include more symbols

2. **Risk-Off (Bear + High Vol + Weak Breadth + High Correlation)**
   - Reduce position sizes by 50%
   - Disable or limit short strategies (short squeezes more likely)
   - Focus on defensive symbols (low beta, high quality)
   - Tighten stop losses

3. **Choppy/Range-Bound (Sideways + Medium Vol + Neutral Breadth)**
   - Favor mean-reversion strategies
   - Reduce trend-following exposure
   - Normal position sizes

4. **Crash/Crisis (Bear + Extreme Vol + Weak Breadth + High Correlation)**
   - Emergency mode: Reduce exposure to 20% of normal
   - Exit all positions or hold only defensive positions
   - Consider full stop trading until volatility subsides

---

## Regime Detection Methods

### Method 1: Threshold-Based (Simplest, Recommended for Start)

**Inputs**:
- Realized volatility (SPY, 20-day)
- Trend indicator (SPY price vs 50-day SMA)
- Breadth indicator (% of universe above 50-day SMA)
- Correlation (average pairwise, top 100 stocks, 20-day)

**Logic**:
```python
def detect_regime_threshold():
    vol_regime = "low" if realized_vol < 0.10 else "medium" if realized_vol < 0.20 else "high"
    trend_regime = "bull" if spy_price > spy_sma50 and sma_slope > 0.001 else \
                   "bear" if spy_price < spy_sma50 and sma_slope < -0.001 else "sideways"
    breadth_regime = "strong" if pct_above_sma > 0.60 else "weak" if pct_above_sma < 0.40 else "neutral"
    correlation_regime = "low" if avg_corr < 0.30 else "high" if avg_corr > 0.60 else "medium"
    
    # Combine into single state
    if vol_regime == "high" and trend_regime == "bear":
        return "risk_off"
    elif vol_regime == "low" and trend_regime == "bull" and breadth_regime == "strong":
        return "risk_on"
    elif trend_regime == "sideways":
        return "choppy"
    else:
        return "neutral"
```

**Advantages**:
- Simple to implement and understand
- Transparent, easy to backtest

**Disadvantages**:
- Can be noisy, frequent regime switches
- Thresholds are somewhat arbitrary

---

### Method 2: Hidden Markov Model (HMM) (Advanced)

**Purpose**: Probabilistic model that infers latent market regimes from observed indicators

**Inputs**:
- Daily returns of SPY, QQQ
- Realized volatility
- Breadth indicators
- Volume

**States**:
- State 1: Bull/Low Vol
- State 2: Bear/High Vol
- State 3: Sideways/Medium Vol

**Advantages**:
- Smooths regime transitions
- Provides regime probabilities (not just binary classification)
- Can capture complex patterns

**Disadvantages**:
- More complex to implement (requires `hmmlearn` library)
- Requires sufficient historical data for training
- Can be overfit if not careful

**Implementation**:
```python
from hmmlearn import hmm

# Train HMM on historical data
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
features = np.column_stack([returns, volatility, breadth])
model.fit(features)

# Predict regime
regime_probs = model.predict_proba(latest_features)
regime = model.predict(latest_features)
```

---

### Method 3: Markov Switching Regression (Academic)

**Purpose**: Econometric model that allows parameters to switch between regimes

**Advantages**:
- Rigorous statistical framework
- Used in academic research

**Disadvantages**:
- Very complex, requires `statsmodels` or `PyMarkovSwitching`
- Computationally intensive
- Likely overkill for this use case

**Recommendation**: Use Method 1 (threshold-based) initially, upgrade to Method 2 (HMM) if needed.

---

## Regime Transition Logic

### Avoiding Whipsaw

**Problem**: Regime detection can oscillate rapidly, causing frequent strategy changes.

**Solutions**:
1. **Hysteresis**: Require different thresholds for entering vs exiting a regime
   - Example: Enter "high vol" at 20%, but don't exit until vol < 15%
   
2. **Confirmation Period**: Require regime to persist for N days before acting
   - Example: Only switch to "risk off" if regime persists for 3 consecutive days
   
3. **Smoothing**: Use exponential moving average of regime indicators
   - Example: Use 5-day EMA of realized volatility instead of daily value

**Recommended Approach**:
```python
# Track regime history
regime_history = deque(maxlen=5)

def get_confirmed_regime():
    current_regime = detect_regime_threshold()
    regime_history.append(current_regime)
    
    # Require 3 of last 5 days to confirm regime change
    if regime_history.count(current_regime) >= 3:
        return current_regime
    else:
        return previous_confirmed_regime  # Stay in previous regime
```

---

## Sizing Rules by Regime

### Position Sizing Adjustments

| Regime | Position Size Multiplier | Max Gross Exposure | Max Positions | Notes |
|--------|--------------------------|---------------------|---------------|-------|
| **Risk-On** | 1.0x | 100% | 30 | Normal sizing |
| **Neutral** | 0.8x | 80% | 25 | Slightly conservative |
| **Choppy** | 0.7x | 70% | 20 | Reduce exposure, favor mean-reversion |
| **Risk-Off** | 0.5x | 50% | 15 | Defensive, tight stops |
| **Crash** | 0.2x or 0x | 20% or 0% | 5 or 0 | Emergency mode, exit most/all positions |

**Implementation**:
```python
def calculate_position_size(signal, regime):
    base_size = calculate_base_size(signal)  # Based on Kelly, fixed %, or ATR
    
    regime_multipliers = {
        "risk_on": 1.0,
        "neutral": 0.8,
        "choppy": 0.7,
        "risk_off": 0.5,
        "crash": 0.2
    }
    
    multiplier = regime_multipliers.get(regime, 0.8)  # Default to 0.8
    return base_size * multiplier
```

---

### Strategy Activation by Regime

Some strategies work better in certain regimes:

| Strategy | Optimal Regimes | Disable in Regimes |
|----------|-----------------|---------------------|
| **Intraday Mean Reversion** | Low/Medium Vol, Choppy | High Vol, Crash |
| **Swing Trend Following** | Bull, Risk-On | Choppy, High Correlation |
| **Breakout/Momentum** | Bull, Strong Breadth | Bear, Weak Breadth |
| **Short Strategies** | Bear, Weak Breadth | Risk-Off (short squeeze risk) |
| **Market Neutral** | All regimes (less regime-dependent) | Crash (correlation breakdown) |

**Configuration**:
```yaml
strategies:
  intraday_mean_reversion:
    enabled_regimes: ["risk_on", "neutral", "choppy"]
    disabled_regimes: ["risk_off", "crash"]
  
  swing_trend_following:
    enabled_regimes: ["risk_on", "bull"]
    disabled_regimes: ["choppy", "crash"]
```

**Implementation**:
```python
def should_generate_signal(strategy, regime):
    if regime in strategy.disabled_regimes:
        return False
    if regime not in strategy.enabled_regimes:
        return False  # Only generate signals in optimal regimes
    return True
```

---

## Risk Limit Adjustments by Regime

### Daily Drawdown Limits

| Regime | Daily Max Drawdown | Rationale |
|--------|---------------------|-----------|
| Risk-On | 2.0% | Normal limit |
| Neutral | 1.5% | Tighter control |
| Risk-Off | 1.0% | Very tight, preserve capital |
| Crash | 0.5% or halt | Emergency mode |

### Stop Loss Adjustments

| Regime | Stop Loss Multiplier | Example (2x ATR base) |
|--------|----------------------|-----------------------|
| Risk-On | 1.0x | 2.0 ATR |
| Neutral | 0.9x | 1.8 ATR |
| Risk-Off | 0.7x | 1.4 ATR (tighter stops) |

**Rationale**: In high-volatility regimes, tighten stops to reduce tail risk.

---

## Regime Features for ML Models

If using ML models, regime information can be input features:

### Regime Indicators as Features
1. **Realized Volatility** (continuous)
2. **VIX or VIX Proxy** (continuous)
3. **Trend Strength** (e.g., price distance from SMA, continuous)
4. **Breadth** (% above SMA, continuous)
5. **Correlation** (average pairwise correlation, continuous)
6. **Regime Label** (categorical: one-hot encoded)

### Meta-Labeling with Regime
- Train separate models for each regime (e.g., "Bull Model", "Bear Model")
- Or, include regime features in a single model and let it learn regime-dependent patterns

**Example**:
```python
features = [
    "zscore", "rsi", "atr", "volume_zscore",  # Standard features
    "realized_vol", "trend_strength", "breadth", "correlation",  # Regime features
    "regime_risk_on", "regime_risk_off", "regime_choppy"  # One-hot encoded regime
]
```

---

## Logging and Monitoring

### Daily Regime Log
Log regime state at market close each day:
```json
{
  "date": "2024-01-15",
  "regime": "risk_on",
  "realized_vol": 0.12,
  "trend": "bull",
  "breadth": 0.65,
  "correlation": 0.28,
  "position_size_multiplier": 1.0,
  "active_strategies": ["intraday_mean_reversion", "swing_trend_following"]
}
```

### Regime Transition Alerts
Alert operators when regime changes:
```
[REGIME CHANGE] 2024-01-15 16:00 ET
Old: Neutral
New: Risk-Off
Reason: Realized vol increased to 22%, SPY broke below SMA(50)
Action: Reduced position sizes to 50%, disabled breakout strategy
```

---

## Backtesting Regime-Aware Strategies

### Historical Regime Labeling
1. Compute regime indicators for full historical period
2. Label each day with regime state
3. Run backtest with regime-aware sizing and strategy activation

### Analyze Performance by Regime
```python
results_by_regime = {
    "risk_on": {"sharpe": 1.8, "win_rate": 0.55, "max_dd": 5%},
    "risk_off": {"sharpe": 0.3, "win_rate": 0.45, "max_dd": 8%},
    "choppy": {"sharpe": 1.2, "win_rate": 0.52, "max_dd": 6%},
}
```

**Goal**: Ensure strategy performs well in all regimes, or at least doesn't blow up in unfavorable regimes.

---

## Implementation Checklist

- [ ] Define regime indicators (volatility, trend, breadth, correlation)
- [ ] Implement threshold-based regime detection
- [ ] Add hysteresis and confirmation logic to avoid whipsaw
- [ ] Create regime state machine (state transitions)
- [ ] Implement position size adjustments by regime
- [ ] Implement strategy activation/deactivation by regime
- [ ] Adjust risk limits (daily DD, stop losses) by regime
- [ ] Log regime state daily
- [ ] Send alerts on regime transitions
- [ ] Backtest regime-aware strategies
- [ ] Analyze performance by regime
- [ ] (Optional) Implement HMM-based regime detection for smoothness
- [ ] Create Grafana dashboard panel for current regime

---

## Configuration File Example

```yaml
# config/regime_config.yaml
regime:
  enabled: true
  method: "threshold"  # threshold, hmm, markov_switching
  
  indicators:
    realized_volatility:
      symbol: "SPY"
      window: 20  # days
    trend:
      symbol: "SPY"
      sma_window: 50
      slope_window: 20
    breadth:
      method: "pct_above_sma"  # or advance_decline if available
      sma_window: 50
      min_symbols: 100
    correlation:
      method: "pairwise"
      window: 20
      top_n_symbols: 100
  
  thresholds:
    volatility:
      low: 0.10
      high: 0.20
    trend:
      bull_slope: 0.001
      bear_slope: -0.001
    breadth:
      strong: 0.60
      weak: 0.40
    correlation:
      low: 0.30
      high: 0.60
  
  regime_states:
    risk_on:
      position_multiplier: 1.0
      max_gross_exposure: 1.0
      max_positions: 30
      enabled_strategies: ["intraday_mean_reversion", "swing_trend_following"]
    
    risk_off:
      position_multiplier: 0.5
      max_gross_exposure: 0.5
      max_positions: 15
      enabled_strategies: ["swing_trend_following"]
      disabled_strategies: ["breakout"]
    
    choppy:
      position_multiplier: 0.7
      max_gross_exposure: 0.7
      max_positions: 20
      enabled_strategies: ["intraday_mean_reversion"]
    
    crash:
      position_multiplier: 0.0
      max_gross_exposure: 0.0
      max_positions: 0
      enabled_strategies: []
  
  confirmation:
    require_consecutive_days: 3
    use_hysteresis: true
```

---

## References

- **Regime Switching Models**: Hamilton, J. D. (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." Econometrica, 357-384.
- **Market Regimes and Trading**: Kritzman, M., & Li, Y. (2010). "Skulls, financial turbulence, and risk management." Financial Analysts Journal, 66(5), 30-41.
- **VIX and Volatility Regimes**: Whaley, R. E. (2009). "Understanding the VIX." The Journal of Portfolio Management, 35(3), 98-105.
- **Breadth Indicators**: Zweig, M. (1986). "Winning on Wall Street." Warner Books. (Advance/Decline line)
- **HMM for Regime Detection**: Ang, A., & Timmermann, A. (2012). "Regime changes and financial markets." Annual Review of Financial Economics, 4(1), 313-337.
