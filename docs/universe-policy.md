# Universe Construction Policy

## Overview
This document defines the criteria for stock universe selection, tier classification, refresh cadence, and data sources for liquidity, shortability, and earnings information.

## Purpose

A well-constructed universe ensures:
1. **Liquidity**: Sufficient volume to enter/exit positions without excessive slippage
2. **Tradeability**: Stocks that are shortable (if strategy requires shorting)
3. **Quality**: Avoid illiquid, manipulated, or problematic symbols
4. **Stability**: Consistent universe to enable apples-to-apples backtesting
5. **Adaptability**: Dynamic refresh to adapt to changing market conditions

---

## Universe Selection Criteria

### Primary Filters

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Average Dollar Volume (ADV)** | > $1,000,000 per day | Ensure sufficient liquidity for execution |
| **Price Range** | $5 - $500 | Avoid penny stocks (unreliable) and very high-priced stocks (capital inefficiency) |
| **Average Spread** | < 50 bps (basis points) | Control transaction costs |
| **Market Cap** | > $100M (optional) | Avoid micro-caps (higher manipulation risk) |
| **Listing Status** | Primary-listed on NYSE, Nasdaq, AMEX | Avoid OTC stocks |
| **Corporate Status** | Not bankrupt, not in delisting process | Avoid distressed situations |

### ADV Calculation
```python
ADV = mean(Close × Volume) over last 60 trading days
```

**Notes**:
- Use 60-day lookback to capture recent liquidity trends
- Recompute daily (or weekly) to keep universe fresh
- Remove symbols that fall below ADV threshold for 5 consecutive days

---

### Spread Calculation
```python
Spread (bps) = ((Ask - Bid) / Mid) × 10,000
Avg_Spread = mean(Spread) over last 20 trading days
```

**Data Source**:
- Real-time: Use quote data from Alpaca WebSocket
- Historical: Use daily bid/ask from data provider (if available)
- Proxy (if bid/ask unavailable): Assume spread = 0.1% for liquid stocks, 0.5% for less liquid

---

### Volatility Bucketing (Optional)

Group symbols by realized volatility for regime-aware selection:

| Bucket | Annualized Volatility | Strategy Suitability |
|--------|------------------------|----------------------|
| **Low Vol** | < 15% | Mean reversion, pairs trading |
| **Medium Vol** | 15% - 30% | Balanced strategies |
| **High Vol** | > 30% | Momentum, breakout (use smaller sizes) |

**Calculation**:
```python
Returns = log(Close / Close.shift(1))
Realized_Vol = std(Returns) × sqrt(252)  # Annualized
```

---

## Universe Tiers

### Tier 1: Core Universe (Top 200)
**Purpose**: Highest quality, most liquid stocks for primary trading

**Selection**:
- Rank all stocks by ADV
- Take top 200 that meet all primary filters
- Include major indices (SPY, QQQ, IWM) for market exposure

**Characteristics**:
- ADV > $10M per day (typically)
- Spread < 10 bps (typically)
- Large/mega-cap stocks

**Use Cases**:
- All strategies can trade Tier 1
- Default universe for live trading

---

### Tier 2: Extended Universe (Top 1000)
**Purpose**: Broader opportunity set for diversification

**Selection**:
- Rank all stocks by ADV
- Take top 1000 that meet all primary filters
- Includes Tier 1 (Core 200) plus additional 800 symbols

**Characteristics**:
- ADV > $1M per day
- Spread < 50 bps
- Large, mid, and some small-cap stocks

**Use Cases**:
- Swing strategies (longer holding periods tolerate wider spreads)
- Alpha generation (more opportunities)
- Use with smaller position sizes due to lower liquidity

---

### Tier 3: Research Universe (Top 2000+)
**Purpose**: Backtesting and research only (not for live trading initially)

**Selection**:
- Relax ADV to $500K per day
- Include smaller-cap stocks

**Use Cases**:
- Strategy development
- Academic research
- Expansion candidates (promote to Tier 2 if liquidity improves)

---

## Exclusion Criteria

### Hard Exclusions (Never Trade)
1. **Penny Stocks**: Price < $5 (manipulable, unreliable quotes)
2. **Illiquid Stocks**: ADV < $1M (slippage too high)
3. **ETFs/ETNs**: Focus on equities only (or separate universe for ETFs)
4. **Preferred Shares**: Different risk profile
5. **Warrants, Rights, Units**: Complex instruments
6. **Foreign ADRs** (optional): May have currency risk, lower liquidity
7. **Bankrupt/Delisted**: In bankruptcy proceedings or scheduled for delisting

### Soft Exclusions (Conditional)
1. **IPOs**: Exclude for first 30 days (unstable price action)
2. **Stocks in Earnings Blackout**: Exclude 2 days before and 1 day after earnings announcement
3. **Corporate Actions**: Exclude 5 days before/after mergers, spin-offs, major events
4. **Hard to Borrow**: Exclude if short borrow rate > 10% (for short strategies)
5. **Low Float**: Exclude if float < 10M shares (prone to squeezes)

---

## Shortability Validation

### Purpose
Ensure stocks are available to short before placing short orders.

### Data Sources
1. **Alpaca Shortability API** (primary):
   - Call `/v2/stocks/{symbol}/locate` to check availability
   - Returns: `easy_to_borrow` (ETB) or `hard_to_borrow` (HTB)
   
2. **Interactive Brokers API** (if using IB):
   - `reqScannerSubscription` with `shortable` filter
   
3. **Third-party Data** (e.g., S3 Partners, Fintel):
   - Borrow rates and availability (subscription required)

### Shortability Thresholds
- **Allow Short**: Easy to Borrow (ETB) or borrow rate < 1%
- **Conditional Short**: Borrow rate 1% - 5% (proceed with caution)
- **Disallow Short**: Borrow rate > 5% or Hard to Borrow (HTB)

### Daily Refresh
- Query shortability every day at pre-market
- Update universe flags: `shortable: true/false`
- Short strategies skip non-shortable symbols

---

## Earnings Blackout Detection

### Purpose
Avoid trading around earnings announcements (high volatility, unpredictable gaps).

### Data Sources
1. **Earnings Calendar API**:
   - Alpaca (if available)
   - Polygon.io (premium)
   - Yahoo Finance (free but less reliable)
   - Earnings Whispers (paid)

2. **Corporate Filings** (backup):
   - SEC EDGAR for 8-K filings (earnings releases)

### Blackout Windows
- **Conservative**: Exclude 5 days before and 2 days after earnings
- **Moderate**: Exclude 2 days before and 1 day after earnings
- **Aggressive**: Exclude earnings day only

**Recommendation**: Start with moderate, adjust based on strategy risk tolerance.

### Implementation
```python
def is_in_earnings_blackout(symbol, date):
    earnings_date = get_next_earnings_date(symbol)
    if earnings_date is None:
        return False  # No known earnings, assume tradeable
    days_to_earnings = (earnings_date - date).days
    return -2 <= days_to_earnings <= 1  # 2 days before, earnings day, 1 day after
```

---

## Universe Refresh Cadence

### Daily Refresh (Recommended)
- **Time**: Pre-market (8:30 AM ET)
- **Actions**:
  - Update shortability flags
  - Check for new earnings announcements
  - Flag corporate actions (splits, mergers, delistings)
  - Remove symbols that fell below thresholds

### Weekly Refresh (Liquidity and Tier Updates)
- **Time**: Sunday evening or Monday pre-market
- **Actions**:
  - Recompute ADV for all symbols (using last 60 days)
  - Recompute average spreads
  - Re-rank symbols and update tiers (Core 200, Extended 1000)
  - Add new IPOs (if > 30 days old)
  - Remove delisted symbols

### Monthly Refresh (Full Re-evaluation)
- **Time**: First Sunday of the month
- **Actions**:
  - Full universe rebuild from scratch
  - Review market cap thresholds
  - Calibrate spread and ADV thresholds based on market conditions
  - Analyze universe performance (hit rate, slippage, fill rate)

---

## Regime-Aware Universe Adjustments

### Bull Market / Risk-On
- Expand universe to Tier 2 (Extended 1000)
- Include more growth stocks, higher volatility names
- Relax spread threshold slightly (< 60 bps)

### Bear Market / Risk-Off
- Contract universe to Tier 1 (Core 200)
- Focus on liquid, defensive stocks
- Tighten spread threshold (< 30 bps)
- Prioritize shortable stocks

### High Volatility
- Reduce universe size (top 100 most liquid)
- Avoid high-beta stocks
- Focus on ETFs (SPY, QQQ) for exposure

---

## Data Sources Summary

| Data Type | Primary Source | Fallback | Refresh Frequency |
|-----------|----------------|----------|-------------------|
| **OHLCV (Historical)** | Alpaca | Polygon, Tiingo | Daily (EOD) |
| **OHLCV (Real-time)** | Alpaca WebSocket | N/A | Streaming |
| **Shortability** | Alpaca Locate API | IB, S3 Partners | Daily (pre-market) |
| **Earnings Calendar** | Polygon.io | Yahoo Finance | Daily (pre-market) |
| **Corporate Actions** | Alpaca | SEC EDGAR | Daily |
| **Market Cap / Fundamentals** | Alpaca | Financial Modeling Prep | Weekly |

---

## Universe Analytics

### Liquidity Scoring

Assign a liquidity score (0-100) to each symbol:

```python
def calculate_liquidity_score(symbol):
    adv = get_average_dollar_volume(symbol, days=60)
    spread = get_average_spread(symbol, days=20)
    volume_consistency = 1 - std(volumes) / mean(volumes)  # Lower volatility is better
    
    # Normalize each component to 0-100
    adv_score = min(100, (adv / 10_000_000) * 100)  # $10M ADV = 100
    spread_score = max(0, 100 - spread * 2)  # 0 bps = 100, 50 bps = 0
    consistency_score = volume_consistency * 100
    
    # Weighted average
    liquidity_score = (adv_score * 0.5 + spread_score * 0.3 + consistency_score * 0.2)
    return liquidity_score
```

**Use Cases**:
- Rank symbols within a tier
- Adjust position sizes (higher score = larger size)
- Filter for specific strategies (e.g., only trade score > 70)

---

### Volatility Analysis

Classify symbols by volatility and analyze strategy performance:

```python
results_by_volatility = {
    "low_vol": {"sharpe": 1.5, "win_rate": 0.52, "slippage_bps": 3},
    "medium_vol": {"sharpe": 1.2, "win_rate": 0.50, "slippage_bps": 5},
    "high_vol": {"sharpe": 0.8, "win_rate": 0.48, "slippage_bps": 10}
}
```

**Insight**: If high-vol stocks underperform, exclude them or use smaller sizes.

---

## Historical Universe Consistency (for Backtesting)

### Challenge
The universe changes over time (companies are added/removed). Backtests must use the universe that was available at each point in time to avoid look-ahead bias.

### Solution: Point-in-Time Universe
1. Store daily universe snapshots in database:
   ```sql
   CREATE TABLE universe_history (
       date DATE,
       symbol VARCHAR(10),
       tier INT,
       adv FLOAT,
       spread_bps FLOAT,
       shortable BOOLEAN
   );
   ```
2. During backtesting, query universe for each date:
   ```python
   universe_on_date = db.query("SELECT symbol FROM universe_history WHERE date = ?", date)
   ```

### Survivorship Bias
Avoid survivorship bias by including delisted symbols in historical universe.

**Example**: If you only use current S&P 500 members in a 20-year backtest, you'll overestimate performance (winners are over-represented).

**Solution**: Use historical index constituents at each point in time, or use a broad universe and allow natural additions/removals.

---

## Configuration File Example

```yaml
# config/universe_config.yaml
universe:
  refresh_cadence: "daily"  # daily, weekly, monthly
  refresh_time: "08:30"  # Pre-market
  
  filters:
    min_price: 5.0
    max_price: 500.0
    min_adv_usd: 1_000_000
    max_spread_bps: 50
    min_market_cap_usd: 100_000_000  # Optional
    exchanges: ["NYSE", "NASDAQ", "AMEX"]
  
  tiers:
    core:
      size: 200
      min_adv_usd: 10_000_000
      max_spread_bps: 10
    extended:
      size: 1000
      min_adv_usd: 1_000_000
      max_spread_bps: 50
    research:
      size: 2000
      min_adv_usd: 500_000
      max_spread_bps: 100
  
  shortability:
    enabled: true
    source: "alpaca"  # alpaca, ib, s3_partners
    max_borrow_rate_pct: 5.0
    check_frequency: "daily"
  
  earnings_blackout:
    enabled: true
    days_before: 2
    days_after: 1
    calendar_source: "polygon"  # polygon, yahoo, earnings_whispers
  
  exclusions:
    exclude_ipos_days: 30
    exclude_etfs: true
    exclude_preferred: true
    exclude_foreign_adrs: false
  
  analytics:
    liquidity_scoring: true
    volatility_bucketing: true
    performance_tracking: true
  
  regime_adjustments:
    risk_on:
      tier: "extended"  # Use Tier 2
      max_spread_bps: 60
    risk_off:
      tier: "core"  # Use Tier 1
      max_spread_bps: 30
```

---

## Implementation Checklist

- [ ] Implement ADV calculation (60-day rolling)
- [ ] Implement spread calculation (20-day rolling)
- [ ] Implement tier assignment logic (Core 200, Extended 1000)
- [ ] Integrate shortability API (Alpaca Locate)
- [ ] Integrate earnings calendar API (Polygon or Yahoo)
- [ ] Implement earnings blackout logic
- [ ] Implement daily universe refresh (pre-market)
- [ ] Implement weekly tier refresh
- [ ] Store universe snapshots for point-in-time backtesting
- [ ] Implement liquidity scoring
- [ ] Implement volatility bucketing
- [ ] Create regime-aware universe adjustments
- [ ] Log universe changes (additions/removals)
- [ ] Create Grafana dashboard panel for universe size and quality
- [ ] Backtest with historical universe (avoid look-ahead bias)

---

## Monitoring and Alerts

### Daily Universe Summary
Log at pre-market:
```json
{
  "date": "2024-01-15",
  "tier_1_count": 200,
  "tier_2_count": 1000,
  "new_additions": ["AAPL", "NVDA"],
  "removals": ["XYZ"],
  "shortable_count": 850,
  "earnings_blackout_count": 25
}
```

### Alerts
- **Significant Universe Contraction**: Alert if Core drops below 150 symbols
- **High Blackout Rate**: Alert if > 10% of universe in earnings blackout
- **Shortability Issues**: Alert if shortable count drops below 70%

---

## References

- **Liquidity Metrics**: Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and time-series effects." Journal of Financial Markets, 5(1), 31-56.
- **Bid-Ask Spread**: Roll, R. (1984). "A simple implicit measure of the effective bid-ask spread in an efficient market." The Journal of Finance, 39(4), 1127-1139.
- **Market Microstructure**: Hasbrouck, J. (2007). "Empirical Market Microstructure." Oxford University Press.
- **Survivorship Bias**: Brown, S. J., Goetzmann, W., Ibbotson, R. G., & Ross, S. A. (1992). "Survivorship bias in performance studies." The Review of Financial Studies, 5(4), 553-580.
