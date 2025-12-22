# Backtesting Assumptions and Cost Modeling

## Overview
This document defines the slippage models, cost assumptions, fill simulation logic, and limitations used in the backtesting framework to achieve realistic performance estimates.

## Core Principle
**"Backtest results should closely approximate paper trading and live trading performance."**

To achieve this, the backtester must model:
1. Slippage (spread, market impact, adverse selection)
2. Transaction costs (commissions, SEC fees, short borrow costs)
3. Partial fills and queue position
4. Latency and execution delays
5. Realistic order types and time-in-force

---

## Slippage Modeling

### Definition
**Slippage** = Actual execution price - Expected price at signal generation

Slippage arises from:
- **Bid-ask spread** (minimum cost for market orders)
- **Market impact** (moving the market with large orders)
- **Adverse selection** (price moves against you before fill)
- **Latency** (price changes during order transmission)

### Slippage Model (Default)

#### For Market Orders
```
Slippage = (Spread / 2) + Volatility_Adjustment + Market_Impact
```

**Components**:
1. **Spread Cost**: Half-spread (optimistic assumption of hitting the mid)
   - Spread (bps) = (Ask - Bid) / Mid × 10,000
   - Spread varies by:
     - Time of day (wider at open/close)
     - Volatility (wider in volatile periods)
     - Liquidity (wider for less liquid stocks)

2. **Volatility Adjustment**:
   - During high volatility, price moves faster than you can execute
   - Adjustment = k × ATR × (Current_Vol / Avg_Vol)
   - Typical k = 0.1 to 0.3 (calibrate vs paper trading)

3. **Market Impact**:
   - Larger orders move the price against you
   - Simplified model: Impact = α × (Order_Size / ADV)^β
   - Typical α = 0.001, β = 0.5 (calibrate per universe tier)
   - More sophisticated: Use market impact curves from academic research

#### For Limit Orders
```
Fill_Probability = f(Limit_Price, Current_Price, Volatility, Queue_Position)
Slippage = Max(0, Limit_Price - Expected_Price)  # Positive if you get price improvement
```

**Fill Probability Model**:
- If Limit Price is aggressive (at or through the market), assume high fill rate (> 90%)
- If Limit Price is passive (away from market), model queue position and probability

**Simplified Approach** (used in initial implementation):
- Limit orders with offset < 5 bps: 80% fill rate
- Limit orders with offset 5-10 bps: 60% fill rate
- Limit orders with offset > 10 bps: 40% fill rate

**Advanced Approach** (future enhancement):
- Model order book dynamics
- Estimate queue position based on order size and book depth
- Simulate fills based on volume at price level

---

### Slippage by Time of Day

Market microstructure varies throughout the trading session:

| Time Period | Spread Multiplier | Volatility Multiplier | Notes |
|-------------|-------------------|-----------------------|-------|
| Market Open (9:30-9:35) | 2.0x | 3.0x | Wide spreads, high volatility, avoid or use smaller sizes |
| Morning (9:35-10:30) | 1.5x | 1.5x | Still elevated |
| Mid-Day (10:30-15:00) | 1.0x | 1.0x | Normal conditions (baseline) |
| Afternoon (15:00-15:55) | 1.2x | 1.2x | Volatility picks up |
| Market Close (15:55-16:00) | 2.5x | 2.0x | Wide spreads, volatile, avoid or use smaller sizes |

**Backtester Configuration**:
```python
slippage_config = {
    "base_slippage_bps": 5,  # Baseline slippage in basis points
    "time_multipliers": {
        "09:30-09:35": 2.0,
        "09:35-10:30": 1.5,
        "10:30-15:00": 1.0,
        "15:00-15:55": 1.2,
        "15:55-16:00": 2.5
    },
    "volatility_adjustment": True,
    "market_impact": True
}
```

---

### Slippage by Liquidity Tier

Different liquidity tiers have different slippage characteristics:

| Tier | ADV Range | Base Spread (bps) | Slippage Assumption |
|------|-----------|-------------------|---------------------|
| **Tier 1** (Mega-cap) | > $500M | 2-5 | Low slippage, tight spreads |
| **Tier 2** (Large-cap) | $100M - $500M | 5-10 | Moderate slippage |
| **Tier 3** (Mid-cap) | $10M - $100M | 10-20 | Higher slippage, wider spreads |
| **Tier 4** (Small-cap) | $1M - $10M | 20-50 | High slippage, use caution |

**Recommendation**:
- Focus on Tier 1 and Tier 2 for live trading
- Use Tier 3 only for swing strategies with longer holding periods
- Avoid Tier 4 in live trading unless you have specialized execution algorithms

---

## Transaction Costs

### Commission Costs
**Alpaca**: Zero commission for stocks (as of 2024)

**Other Brokers** (for reference):
- Interactive Brokers: $0.005 per share (min $1, max 0.5% of trade value)
- TD Ameritrade: $0 for stocks

**Backtester Assumption**: $0 commission (accurate for Alpaca)

### SEC Fees
- **Rate**: $27.80 per $1,000,000 of principal (sales only, updated 2024)
- **Calculation**: Total_Sell_Value × 0.0000278
- **Backtester**: Model SEC fees on all sell orders

### FINRA Trading Activity Fee (TAF)
- **Rate**: $0.000166 per share (sales only, max $8.30 per trade)
- **Calculation**: Min(Shares_Sold × 0.000166, 8.30)
- **Backtester**: Model TAF on all sell orders

### Short Borrow Costs
**Purpose**: Model cost of borrowing shares to short

**Cost Structure**:
- **Easy to Borrow (ETB)**: 0.5% - 1.5% annualized
- **Hard to Borrow (HTB)**: 5% - 30% annualized (or more for extreme cases)
- **Short Squeeze Candidates**: Can exceed 100% annualized

**Backtester Assumption** (conservative):
- Default: 1% annualized for all shorts (assume ETB stocks only)
- Optional: Query Alpaca's shortability API for real-time borrow rates
- Cost per short position: (Short_Value × Annual_Rate × Holding_Days) / 365

**Example**:
- Short $10,000 for 5 days at 1% annual rate
- Cost = $10,000 × 0.01 × 5 / 365 = $1.37

### Total Transaction Costs Formula
```python
def calculate_transaction_costs(trade):
    costs = 0
    
    # Commission (Alpaca = $0)
    costs += 0
    
    if trade.side == 'sell':
        # SEC Fee (sells only)
        costs += trade.value * 0.0000278
        
        # FINRA TAF (sells only)
        costs += min(trade.shares * 0.000166, 8.30)
    
    if trade.side == 'short':
        # Borrow cost (daily rate)
        annual_rate = 0.01  # 1% default
        daily_cost = trade.value * annual_rate / 365
        costs += daily_cost * trade.holding_days
    
    return costs
```

---

## Fill Simulation

### Market Orders
**Assumption**: Market orders always fill, but with slippage

**Execution Price**:
```python
if trade.side == 'buy':
    execution_price = signal_price × (1 + slippage_bps / 10000)
elif trade.side == 'sell' or trade.side == 'cover':
    execution_price = signal_price × (1 - slippage_bps / 10000)
elif trade.side == 'short':
    execution_price = signal_price × (1 - slippage_bps / 10000)
```

### Limit Orders
**Assumption**: Limit orders only fill if price reaches the limit, with probability based on order aggressiveness

**Fill Logic**:
1. Check if bar's high/low touched the limit price
2. If touched, apply fill probability (based on offset from mid)
3. If filled, execution price = limit price (or better)

**Example**:
```python
def simulate_limit_fill(limit_price, bar, fill_probability):
    if trade.side == 'buy':
        if bar.low <= limit_price:
            if random.random() < fill_probability:
                return True, min(limit_price, bar.close)  # Fill at limit or better
    elif trade.side == 'sell':
        if bar.high >= limit_price:
            if random.random() < fill_probability:
                return True, max(limit_price, bar.close)  # Fill at limit or better
    return False, None
```

### Partial Fills
**Assumption**: Large orders may not fill completely in one bar

**Partial Fill Model** (simplified):
- If Order_Size > 10% of Bar_Volume, assume partial fill
- Fill_Quantity = Bar_Volume × Fill_Percentage (e.g., 5% of bar volume)
- Remaining quantity carries over to next bar(s)

**Backtester Configuration**:
```python
partial_fill_config = {
    "enable": True,
    "volume_threshold": 0.10,  # Order > 10% of bar volume triggers partial fill
    "max_fill_pct_per_bar": 0.05  # Max 5% of bar volume per execution
}
```

**Trade-off**:
- More realistic for large orders
- Increases backtest complexity
- May not be necessary if position sizes are small relative to ADV

---

## Latency Modeling

### Signal-to-Order Latency
**Definition**: Time between signal generation and order placement

**Sources of Latency**:
1. **Signal Generation**: Feature computation, model inference (10-50 ms)
2. **Risk Checks**: Pre-trade validation (5-10 ms)
3. **Order Transmission**: API call to broker (50-200 ms)

**Total Latency**: 100-300 ms (typical)

**Backtester Assumption**:
- Signal generated at bar close (time T)
- Order placed at T + Latency
- Execution price = bar close + slippage (conservative)

**Advanced Approach**:
- Use next bar's open price as execution price (more realistic for minute bars)
- Model latency explicitly: execution price = price at T + latency

---

## Order Types and Time-in-Force

### Order Types Supported
1. **Market**: Execute immediately at best available price
2. **Limit**: Execute only at specified price or better
3. **Stop**: Convert to market order when stop price is reached
4. **Stop-Limit**: Convert to limit order when stop price is reached

### Time-in-Force
1. **Day**: Cancel at market close if not filled
2. **GTC** (Good-Till-Canceled): Remain open until filled or manually canceled
3. **IOC** (Immediate-or-Cancel): Fill immediately or cancel
4. **FOK** (Fill-or-Kill): Fill entire order immediately or cancel

**Backtester Configuration**:
```python
order_config = {
    "default_order_type": "limit",
    "default_tif": "day",
    "limit_offset_bps": 5,  # Offset from mid for limit orders
    "order_timeout_bars": 5  # Cancel after 5 bars if not filled (for limit orders)
}
```

---

## Corporate Actions

### Stock Splits
**Handling**: Adjust historical prices and shares before split date

**Example**:
- 2-for-1 split on Date X
- Prices before X divided by 2
- Shares before X multiplied by 2

**Backtester**: Use split-adjusted prices from data provider (Alpaca provides adjusted data)

### Dividends
**Handling**: Optional adjustment based on strategy

**Options**:
1. **Total Return**: Adjust prices for dividends (assume reinvestment)
2. **Price Return**: Do not adjust for dividends (actual cash flow)

**Recommendation**:
- Use total return adjustment for long-term backtests
- Use price return for short-term intraday strategies (dividends less relevant)

**Backtester Configuration**:
```python
corporate_actions_config = {
    "split_adjusted": True,
    "dividend_adjusted": True  # Set False for intraday strategies
}
```

### Mergers and Acquisitions
**Handling**: Mark symbol as delisted after acquisition date

**Backtester**: Close all positions at acquisition price (if available) or last traded price

---

## Limitations and Disclaimers

### Known Limitations
1. **Order Book Depth**: Backtester does not model full order book, only assumes bid/ask spread
2. **Market Impact**: Simplified linear model; real impact can be non-linear
3. **Liquidity Shocks**: Does not model flash crashes or sudden liquidity droughts
4. **News Events**: Does not model halts, circuit breakers, or news-driven gaps
5. **Overnight Gaps**: Assumes positions held overnight experience gap risk (not explicitly modeled in slippage)
6. **Short Availability**: Assumes all stocks are shortable; real-world may have restrictions

### Optimistic Assumptions (Backtester May Overestimate Performance)
- Market orders always fill (in reality, may fail in extreme volatility)
- Limit orders fill at limit price (in reality, may get worse fills or no fills)
- No broker outages or API failures
- No manual errors or operational risk

### Pessimistic Assumptions (Backtester May Underestimate Performance)
- Fixed slippage model (some trades may get price improvement)
- No modeling of limit order price improvement
- Conservative fill probabilities for limit orders

### Recommendation
**Test in Paper Trading**: Run the same strategy in paper trading for 2-4 weeks and compare results to backtest. Calibrate slippage parameters to match reality.

---

## Calibration Process

### Step 1: Run Backtest with Initial Assumptions
- Use default slippage (5 bps), transaction costs as documented

### Step 2: Run Paper Trading
- Execute the same strategy in paper trading for 2-4 weeks
- Log all orders: signal price, order price, execution price, slippage

### Step 3: Analyze Discrepancies
- Compute realized slippage: (Execution Price - Signal Price) / Signal Price
- Group by:
  - Time of day
  - Liquidity tier
  - Order type (market vs limit)
  - Volatility regime

### Step 4: Calibrate Slippage Model
- Adjust base_slippage_bps to match median realized slippage
- Adjust time_multipliers if certain periods have higher slippage
- Adjust volatility_adjustment and market_impact factors

### Step 5: Re-run Backtest
- Verify that updated backtest results align with paper trading performance

### Step 6: Iterate
- Repeat calibration quarterly or after significant market regime changes

---

## Configuration File Example

```yaml
# config/backtest_config.yaml
backtesting:
  slippage:
    base_bps: 5
    time_of_day_adjustments:
      "09:30-09:35": 2.0
      "09:35-10:30": 1.5
      "10:30-15:00": 1.0
      "15:00-15:55": 1.2
      "15:55-16:00": 2.5
    volatility_adjustment: true
    market_impact: true
    market_impact_alpha: 0.001
    market_impact_beta: 0.5
  
  transaction_costs:
    commission_per_share: 0.0  # Alpaca = $0
    sec_fee_rate: 0.0000278  # $27.80 per $1M
    finra_taf_per_share: 0.000166
    finra_taf_max_per_trade: 8.30
    short_borrow_rate_annual: 0.01  # 1% default
  
  fill_simulation:
    market_orders_always_fill: true
    limit_order_fill_probability:
      offset_0_5_bps: 0.80
      offset_5_10_bps: 0.60
      offset_10_plus_bps: 0.40
    partial_fills_enabled: false  # Enable for large order sizes
  
  latency:
    signal_to_order_ms: 150  # Median latency
    use_next_bar_open: false  # If true, execution at next bar open
  
  order_config:
    default_type: "limit"
    default_tif: "day"
    limit_offset_bps: 5
    order_timeout_bars: 5
  
  corporate_actions:
    split_adjusted: true
    dividend_adjusted: true
```

---

## Implementation Checklist

- [ ] Implement slippage model (spread + volatility + market impact)
- [ ] Add time-of-day slippage adjustments
- [ ] Add liquidity-tier slippage adjustments
- [ ] Implement transaction cost calculation (SEC fees, TAF, borrow costs)
- [ ] Implement limit order fill simulation with probability model
- [ ] Implement partial fill logic (optional)
- [ ] Add latency modeling (signal-to-order delay)
- [ ] Handle corporate actions (splits, dividends, delistings)
- [ ] Create calibration process and scripts
- [ ] Document assumptions in backtest reports
- [ ] Compare backtest vs paper trading performance
- [ ] Update slippage model based on calibration

---

## References

- **Market Microstructure in Practice** by Lehalle & Laruelle
- **Algorithmic and High-Frequency Trading** by Cartea, Jaimungal, & Penalva
- **Quantitative Trading** by Ernest Chan (Chapter on transaction costs)
- [Almgren-Chriss Market Impact Model](https://www.math.nyu.edu/faculty/chriss/optliq_f.pdf)
- [SEC Fee Schedule](https://www.sec.gov/rules)
- [FINRA TAF](https://www.finra.org/filing-reporting/taf)
