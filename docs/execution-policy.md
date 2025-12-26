# Execution Policy and Smart Order Management

## Overview
This document defines the execution engine policies including time-slicing logic, limit order offsets, bracket order management, session gates, throttling, and retry strategies.

## Core Principles

1. **Minimize Market Impact**: Split large orders to avoid moving the market
2. **Optimize Fill Quality**: Balance between fill rate and price improvement
3. **Manage Slippage**: Use limit orders with smart offsets
4. **Avoid Toxic Periods**: Reduce activity during high-volatility windows
5. **Respect Rate Limits**: Throttle orders to stay within broker limits

---

## Order Types and Default Configuration

### Default Order Type: Limit Orders

**Rationale**:
- Better price control than market orders
- Potential for price improvement
- Lower adverse selection

**Trade-off**:
- Risk of no fill (opportunity cost)
- Requires careful offset selection

**Configuration**:
```yaml
execution:
  default_order_type: "limit"
  limit_offset_bps: 5  # 5 basis points from mid-price
  order_timeout_seconds: 60  # Cancel if not filled within 60s
```

### When to Use Market Orders

Use market orders only for:
1. **Emergency Exits**: Kill-switch activated, must flatten immediately
2. **EOD Flatten**: Close of intraday positions at 15:55 ET
3. **Small Sizes**: Orders < $500 (slippage negligible)

**Risk**: Market orders can experience significant slippage during volatile periods.

---

## Limit Order Offset Strategy

### Static Offset (Default)

**Method**: Fixed basis point offset from mid-price

```python
if side == 'buy':
    limit_price = mid_price × (1 + offset_bps / 10000)
elif side == 'sell':
    limit_price = mid_price × (1 - offset_bps / 10000)
```

**Default**: 5 bps offset
- Aggressive enough for high fill rate (~80%)
- Conservative enough to avoid worst adverse selection

---

### Adaptive Offset (Advanced)

**Method**: Adjust offset based on spread and volatility

```python
def calculate_adaptive_offset(symbol, base_offset_bps=5):
    spread_bps = get_current_spread_bps(symbol)
    volatility_factor = get_current_volatility_factor(symbol)
    
    # Use larger of base offset or 50% of spread
    offset_bps = max(base_offset_bps, spread_bps * 0.5)
    
    # Increase offset during high volatility
    if volatility_factor > 1.5:  # 1.5x normal volatility
        offset_bps *= 1.5
    
    return offset_bps
```

**Benefits**:
- Better fill rates for wide-spread stocks
- Reduced adverse selection during volatility

**Trade-offs**:
- More complex
- Requires real-time spread and volatility data

---

### Time-of-Day Adjustments

Widen offsets during toxic periods:

| Time Period | Offset Multiplier | Rationale |
|-------------|-------------------|-----------|
| 09:30-09:35 | 2.0x | Market open, wide spreads |
| 09:35-10:00 | 1.5x | Elevated volatility |
| 10:00-15:00 | 1.0x | Normal conditions |
| 15:00-15:55 | 1.2x | Afternoon pickup |
| 15:55-16:00 | 2.0x | Market close, wide spreads |

**Example**:
```python
base_offset = 5 bps
current_time = get_current_time()
if "09:30" <= current_time < "09:35":
    offset = base_offset * 2.0  # 10 bps
```

---

## Time-Slicing for Large Orders

### When to Time-Slice

**Threshold**: Order size > $10,000 (configurable)

**Rationale**:
- Large orders can move the market (market impact)
- Splitting into smaller child orders reduces impact

### Time-Slicing Algorithm (TWAP)

**TWAP (Time-Weighted Average Price)**: Split order evenly over time

**Example**: Buy $50,000 of AAPL
- Split into 5 child orders of $10,000 each
- Execute one every 2 minutes over 10 minutes

```python
def time_slice_order(order_size, threshold=10000, slice_size=5000):
    if order_size < threshold:
        return [order_size]  # No slicing needed
    
    n_slices = math.ceil(order_size / slice_size)
    child_orders = [slice_size] * (n_slices - 1)
    child_orders.append(order_size - sum(child_orders))  # Remainder
    
    return child_orders

# Example: $50,000 order with $5,000 slices
child_orders = time_slice_order(50000, threshold=10000, slice_size=5000)
# Result: [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]

for i, child_size in enumerate(child_orders):
    place_order(child_size)
    if i < len(child_orders) - 1:
        time.sleep(120)  # Wait 2 minutes
```

### Advanced: VWAP-Targeted Execution

**VWAP (Volume-Weighted Average Price)**: Schedule child orders based on historical volume profile

**Example**: More volume typically in first and last hours
- Schedule 30% of order in first hour
- 40% mid-day
- 30% last hour

**Implementation**:
```python
def vwap_schedule(order_size, volume_profile):
    """
    volume_profile: List of percentages by time bucket
    Example: [0.20, 0.15, 0.15, 0.20, 0.30] for 5 time buckets
    """
    schedule = []
    for pct in volume_profile:
        schedule.append(order_size * pct)
    return schedule
```

**Recommendation**: Start with simple TWAP, upgrade to VWAP if needed.

---

## Bracket Orders (Entry + Stop + Take Profit)

### Purpose
Automatically manage risk with stop-loss and take-profit orders upon entry.

### Bracket Order Structure

**Entry Order**: Limit or market order to enter position
**Stop-Loss Order**: Stop order to exit if price moves against you
**Take-Profit Order**: Limit order to exit at profit target

**Example**:
```python
# Entry: Buy 100 shares of AAPL at $150
# Stop-Loss: Sell 100 shares if price drops to $147 (2% loss)
# Take-Profit: Sell 100 shares if price rises to $153 (2% gain)

bracket_order = {
    "symbol": "AAPL",
    "qty": 100,
    "side": "buy",
    "type": "limit",
    "limit_price": 150.00,
    "order_class": "bracket",
    "stop_loss": {
        "stop_price": 147.00,
        "limit_price": 146.50  # Optional limit on stop (stop-limit)
    },
    "take_profit": {
        "limit_price": 153.00
    }
}
```

### Stop-Loss Calculation

**ATR-Based** (Recommended):
```python
stop_loss_distance = ATR × multiplier  # e.g., 1.5x ATR
stop_price = entry_price - stop_loss_distance (for longs)
stop_price = entry_price + stop_loss_distance (for shorts)
```

**Percentage-Based**:
```python
stop_loss_pct = 2.0  # 2% stop
stop_price = entry_price × (1 - stop_loss_pct / 100)  # For longs
```

**Recommendation**: Use ATR-based stops (adapt to volatility).

### Take-Profit Calculation

**Risk-Reward Ratio**:
```python
risk = entry_price - stop_price
reward = entry_price + (risk × risk_reward_ratio)  # e.g., 2:1 ratio

take_profit_price = entry_price + (risk × 2)  # For longs
```

**Partial Profit-Taking** (Advanced):
- Exit 50% at 1:1 risk-reward
- Exit 50% at 2:1 risk-reward (or trailing stop)

---

## Trailing Stops (Dynamic Stop-Loss)

### Purpose
Lock in profits as price moves in your favor.

### Trailing Stop Logic

**Fixed Percentage Trailing**:
```python
trailing_stop_pct = 3.0  # 3% trailing stop

def update_trailing_stop(entry_price, current_price, side):
    if side == 'buy':
        high_water_mark = max(entry_price, current_price)
        trailing_stop = high_water_mark × (1 - trailing_stop_pct / 100)
    elif side == 'sell':  # Short position
        low_water_mark = min(entry_price, current_price)
        trailing_stop = low_water_mark × (1 + trailing_stop_pct / 100)
    return trailing_stop
```

**ATR-Based Trailing**:
```python
trailing_stop_distance = ATR × multiplier  # e.g., 3x ATR

trailing_stop = high_water_mark - trailing_stop_distance  # For longs
```

**Recommendation**: ATR-based trailing stops adapt to volatility.

---

## Session Gates (Toxic Period Avoidance)

### Avoid First 2-5 Minutes (09:30-09:35 ET)

**Rationale**:
- Wide bid-ask spreads
- High volatility
- Market orders from overnight news
- Increased slippage and adverse selection

**Policy**: Disable new entries during 09:30-09:35 ET
- Exception: EOD flatten or emergency exits

---

### Reduce Activity in Last 5 Minutes (15:55-16:00 ET)

**Rationale**:
- MOC (Market-on-Close) imbalance orders create volatility
- Wide spreads
- Unpredictable price action

**Policy**: Reduce position sizes by 50% or disable new entries

---

### EOD Flatten for Intraday Strategies

**Policy**: Close all intraday positions by 15:55 ET

**Implementation**:
```python
def flatten_intraday_positions(time_now):
    if time_now >= "15:55":
        intraday_positions = get_positions(strategy_type="intraday")
        for position in intraday_positions:
            place_order(
                symbol=position.symbol,
                qty=position.qty,
                side='sell' if position.side == 'long' else 'buy_to_cover',
                type='market',  # Use market order for guaranteed exit
                reason="EOD_FLATTEN"
            )
```

---

## Order Throttling and Rate Limits

### Alpaca Rate Limits

**REST API**:
- 200 requests per minute (general)
- 200 orders per minute

**WebSocket**:
- 1 connection per account
- Unlimited messages (subject to fair use)

### Throttling Strategy

**Limit Orders Per Second**:
```python
import time
from collections import deque

class OrderThrottler:
    def __init__(self, max_orders_per_minute=150):
        self.max_orders = max_orders_per_minute
        self.order_times = deque(maxlen=max_orders)
    
    def can_place_order(self):
        now = time.time()
        # Remove orders older than 60 seconds
        while self.order_times and now - self.order_times[0] > 60:
            self.order_times.popleft()
        
        # Check if under limit
        return len(self.order_times) < self.max_orders
    
    def record_order(self):
        self.order_times.append(time.time())

throttler = OrderThrottler(max_orders_per_minute=150)

def place_order_with_throttle(order):
    while not throttler.can_place_order():
        time.sleep(1)  # Wait 1 second
    
    place_order(order)
    throttler.record_order()
```

---

## Retry Logic and Error Handling

### Transient Errors (Retry)

Retry for these errors:
- **500 Internal Server Error**: Broker API issue
- **503 Service Unavailable**: Temporary outage
- **429 Too Many Requests**: Rate limit (back off)
- **Connection Timeout**: Network issue

### Exponential Backoff

```python
import time

def place_order_with_retry(order, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = api.submit_order(**order)
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Order failed (attempt {attempt+1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Order failed after {max_retries} attempts: {e}")
                raise
```

### Non-Retryable Errors (Fail Fast)

Do not retry for:
- **400 Bad Request**: Invalid order parameters (fix code)
- **401 Unauthorized**: Invalid API key
- **403 Forbidden**: Account restrictions (e.g., PDT, insufficient funds)
- **404 Not Found**: Symbol not found

**Action**: Log error, alert operator, do not retry.

---

## Fill Monitoring and Partial Fills

### Order Status Tracking

Poll order status every 1-5 seconds:
```python
def monitor_order(order_id, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        order_status = api.get_order(order_id)
        if order_status.status in ['filled', 'canceled', 'rejected']:
            return order_status
        time.sleep(1)
    
    # Timeout: Cancel order
    api.cancel_order(order_id)
    return api.get_order(order_id)
```

### Partial Fill Handling

**Threshold**: Accept partial fills > 80% of order size

```python
def handle_partial_fill(order):
    filled_qty = order.filled_qty
    requested_qty = order.qty
    fill_rate = filled_qty / requested_qty
    
    if fill_rate >= 0.80:
        logger.info(f"Accepting partial fill: {filled_qty}/{requested_qty}")
        return "accept"
    else:
        logger.warning(f"Low fill rate: {filled_qty}/{requested_qty}, canceling and retrying")
        api.cancel_order(order.id)
        return "retry"
```

---

## Configuration File Example

```yaml
# config/execution_config.yaml
execution:
  default_order_type: "limit"
  use_bracket_orders: true
  time_in_force: "day"
  
  limit_orders:
    base_offset_bps: 5
    adaptive_offset: true
    time_of_day_adjustments:
      "09:30-09:35": 2.0
      "09:35-10:00": 1.5
      "10:00-15:00": 1.0
      "15:00-15:55": 1.2
      "15:55-16:00": 2.0
    order_timeout_seconds: 60
  
  time_slicing:
    enabled: true
    size_threshold_usd: 10000
    child_order_size_usd: 5000
    interval_seconds: 120
    method: "twap"  # twap or vwap
  
  bracket_orders:
    stop_loss:
      method: "atr"  # atr or percentage
      atr_multiplier: 1.5
      percentage: 2.0
    take_profit:
      risk_reward_ratio: 2.0
    trailing_stop:
      enabled: true
      atr_multiplier: 3.0
      percentage: 3.0
  
  session_gates:
    avoid_first_minutes: 5  # 09:30-09:35
    avoid_last_minutes: 5   # 15:55-16:00
    flatten_eod_time: "15:55"
    flatten_eod_order_type: "market"
  
  throttling:
    max_orders_per_minute: 150
    backoff_on_rate_limit: true
  
  retry_policy:
    max_retries: 3
    backoff_strategy: "exponential"
    retryable_errors: [500, 503, 429, "timeout"]
  
  partial_fills:
    min_fill_threshold: 0.80
    retry_on_low_fill: true
  
  slippage_limits:
    max_slippage_bps: 20
    cancel_on_breach: true
```

---

## Monitoring and Alerts

### Execution Metrics (Track in real-time)
- **Fill Rate**: % of orders filled
- **Avg Fill Time**: Time from order placement to fill
- **Slippage**: Execution price vs expected price
- **Rejection Rate**: % of orders rejected
- **Partial Fill Rate**: % of orders partially filled

### Alerts
- **High Rejection Rate**: > 5% of orders rejected in last hour
- **High Slippage**: Average slippage > 15 bps in last hour
- **Low Fill Rate**: < 70% fill rate in last hour
- **Order Timeout**: > 10 orders timed out in last hour

---

## Implementation Checklist

- [ ] Implement limit order offset logic (static and adaptive)
- [ ] Implement time-of-day offset adjustments
- [ ] Implement time-slicing (TWAP)
- [ ] Implement bracket order submission
- [ ] Implement trailing stop logic
- [ ] Implement session gates (avoid first/last minutes)
- [ ] Implement EOD flatten (15:55 ET)
- [ ] Implement order throttling (rate limit compliance)
- [ ] Implement retry logic with exponential backoff
- [ ] Implement order status monitoring
- [ ] Implement partial fill handling
- [ ] Track execution metrics (fill rate, slippage, latency)
- [ ] Create Grafana dashboard for execution metrics
- [ ] Set up alerts for execution issues

---

## References

- **Algorithmic and High-Frequency Trading** by Cartea, Jaimungal, & Penalva
- **Optimal Trading Strategies** by Kissell
- **Market Microstructure in Practice** by Lehalle & Laruelle
- [Alpaca API Docs - Orders](https://alpaca.markets/docs/trading/orders/)
- [Alpaca API Docs - Rate Limits](https://alpaca.markets/docs/api-references/trading-api/#rate-limits)
