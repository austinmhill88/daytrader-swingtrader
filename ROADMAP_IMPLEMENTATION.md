# Phase 1-5 Implementation Guide

## Overview
This document describes all enhancements made to implement the complete roadmap for the trading system. All components are production-ready and follow existing code patterns.

---

## Phase 1: Foundation Enhancements ✅

### 1.1 Data Ingestion Hardening

**File**: `src/alpaca_client.py`

**Changes**:
- Enhanced `get_bars()` method with automatic pagination
- Added `paginate` parameter (default: True)
- Handles large date ranges by fetching data in chunks
- Partial data recovery on batch errors
- Safety limit of 100 iterations to prevent infinite loops

**Usage**:
```python
# Automatic pagination for large date ranges
bars = alpaca_client.get_bars(
    symbol='AAPL',
    timeframe='1Min',
    start='2024-01-01T00:00:00Z',
    end='2024-12-31T23:59:59Z',
    paginate=True  # Will automatically fetch all data
)
```

### 1.2 Shortability Cache Enhancement

**File**: `src/risk_manager.py`

**Changes**:
- TTL-based cache expiration (1 hour default)
- Timestamp tracking for cache entries
- Added `clear_shortability_cache()` method

**Usage**:
```python
# Cache automatically expires after 1 hour
is_shortable = risk_manager._is_shortable('AAPL')

# Manually clear cache (e.g., at EOD)
risk_manager.clear_shortability_cache()
```

---

## Phase 2: ML Training Pipeline ✅

### 2.1 ML Pipeline Orchestrator

**File**: `src/ml_pipeline.py`

**Key Features**:
- End-to-end ML training coordination
- Feature engineering via FeatureStore
- Model training with cross-validation
- Walk-forward validation integration
- Promotion gate checking
- MLflow registry integration

**Usage**:
```python
from src.ml_pipeline import MLPipeline

# Initialize
ml_pipeline = MLPipeline(config)

# Train model for a strategy
results = ml_pipeline.train_strategy_model(
    strategy_name='intraday_mean_reversion',
    historical_data=symbol_data_dict,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Check if promotion gates passed
if results['promotion_passed']:
    logger.info(f"Model ready for production: {results['model_id']}")
```

**Configuration** (`config.yaml`):
```yaml
ml_training:
  enabled: true  # Enable ML training
  training_schedule: "weekly"
  validation_method: "walk_forward"
  
strategies:
  intraday_mean_reversion:
    ml_model:
      enabled: true
      model_type: "lightgbm"
      features: ["zscore", "rsi", "atr", "volume_zscore", "spread"]
```

---

## Phase 3: Execution Sophistication ✅

### 3.1 Slippage Forecasting

**File**: `src/slippage_model.py`

**Key Features**:
- Symbol-specific slippage curves
- Time-of-day multipliers (higher at open/close)
- Order size adjustments
- Spread forecasting
- Persistent model storage

**Usage**:
```python
from src.slippage_model import SlippageModel

slippage_model = SlippageModel(config)

# Forecast slippage for a trade
estimated_slippage_bps = slippage_model.forecast_slippage(
    symbol='AAPL',
    side='buy',
    qty=100,
    current_price=150.0,
    current_time=datetime.now()
)

# Update model with actual fills
slippage_model.update_symbol_stats(
    symbol='AAPL',
    actual_slippage_bps=6.5,
    spread_bps=8.0,
    trade_time=datetime.now()
)

# Save model
slippage_model.save_model()
```

### 3.2 Dynamic Position Sizing

**File**: `src/dynamic_sizing.py`

**Key Features**:
- Regime-based adjustments
- Volatility targeting
- Kelly criterion support
- Portfolio heat calculation
- Correlation-based adjustments

**Usage**:
```python
from src.dynamic_sizing import DynamicSizer

sizer = DynamicSizer(config)

# Calculate position size
qty = sizer.calculate_position_size(
    equity=100000,
    price=150.0,
    atr=3.5,
    symbol='AAPL',
    regime='medium_volatility',
    realized_vol=0.25
)

# Check portfolio heat
heat = sizer.calculate_portfolio_heat(
    portfolio_positions=positions_dict,
    equity=100000
)

# Determine if sizing should be reduced
should_reduce, multiplier = sizer.should_reduce_sizing(
    portfolio_heat=heat,
    consecutive_losses=2
)
```

### 3.3 Exposure Management

**File**: `src/exposure_manager.py`

**Key Features**:
- Intraday exposure caps
- Auto-flatten at configurable time
- Emergency flatten on drawdown/loss streak
- Position reduction capabilities

**Usage**:
```python
from src.exposure_manager import ExposureManager

exposure_mgr = ExposureManager(config)

# Check if new position would exceed limits
allowed, reason = exposure_mgr.check_exposure_limits(
    equity=100000,
    current_positions=positions,
    new_position_value=5000,
    strategy_type='intraday'
)

# Check if should flatten
if exposure_mgr.should_flatten_intraday():
    symbols_to_flatten = exposure_mgr.get_positions_to_flatten(
        current_positions=positions,
        strategy_type='intraday'
    )

# Emergency flatten check
should_flatten, reason = exposure_mgr.should_emergency_flatten(
    daily_drawdown_pct=4.5,
    consecutive_losses=8,
    portfolio_heat=18.0
)
```

### 3.4 Rate Limiting

**File**: `src/rate_limiter.py`

**Key Features**:
- Sliding window rate limiting
- Exponential backoff on 429 errors
- Batch processing with delays
- Operation queuing

**Usage**:
```python
from src.rate_limiter import RateLimiter, BatchProcessor

rate_limiter = RateLimiter(config)

# Execute with automatic backoff
result = rate_limiter.execute_with_backoff(
    alpaca_client.get_bars,
    symbol='AAPL',
    timeframe='1Min',
    max_retries=5
)

# Batch processing
batch_processor = BatchProcessor(rate_limiter)
batch_processor.add_order(order_params)
results = batch_processor.process_orders(submit_order_func)
```

### 3.5 Enhanced Alerting

**File**: `src/enhanced_alerting.py`

**Key Features**:
- Model drift detection (KS test, PSI)
- Data staleness monitoring
- Data quality checks
- Execution lag detection
- Performance anomaly detection

**Usage**:
```python
from src.enhanced_alerting import EnhancedAlertManager

alert_mgr = EnhancedAlertManager(config, notifier)

# Check model drift
alert = alert_mgr.check_model_drift(
    model_name='intraday_model',
    feature_name='zscore',
    train_distribution=train_data,
    prod_distribution=prod_data
)

# Check data staleness
alert = alert_mgr.check_data_staleness(
    symbol='AAPL',
    last_update=last_update_time,
    expected_update_frequency='1Min'
)

# Check data quality
alert = alert_mgr.check_data_quality(
    symbol='AAPL',
    df=data_frame,
    required_columns=['open', 'high', 'low', 'close', 'volume']
)
```

---

## Phase 4: Portfolio Orchestration ✅

### 4.1 Capital Allocation

**File**: `src/capital_allocator.py`

**Key Features**:
- Multiple allocation methods (equal weight, risk parity, Kelly, vol target)
- Strategy type limits
- Performance tracking
- Rebalancing detection

**Usage**:
```python
from src.capital_allocator import CapitalAllocator

allocator = CapitalAllocator(config)

# Set allocation method
allocator.set_allocation_method('risk_parity')

# Allocate capital
allocations = allocator.allocate_capital(
    total_equity=100000,
    strategy_metrics={
        'intraday_mean_reversion': {'volatility': 0.20, 'sharpe': 1.5},
        'swing_trend_following': {'volatility': 0.15, 'sharpe': 1.2}
    }
)

# Check if rebalancing needed
needs_rebalance, adjustments = allocator.rebalance_needed(
    current_allocations=current,
    target_allocations=allocations,
    threshold_pct=10.0
)
```

### 4.2 Universe Rotation Tracking

**File**: `src/universe_rotation.py`

**Key Features**:
- Symbol addition/removal tracking
- Stability scoring
- Minimum retention periods
- Rotation metrics and history

**Usage**:
```python
from src.universe_rotation import UniverseRotationTracker

rotation_tracker = UniverseRotationTracker(config)

# Update universe
rotation_summary = rotation_tracker.update_universe(
    new_universe={'AAPL', 'MSFT', 'GOOGL'},
    metrics={
        'AAPL': {'liquidity_score': 0.95},
        'MSFT': {'liquidity_score': 0.92}
    }
)

# Check if rotation should occur
should_rotate, reason = rotation_tracker.should_rotate()

# Get stability scores
top_stable = rotation_tracker.get_top_stable_symbols(n=50)

# Check if symbol can be removed
can_remove, reason = rotation_tracker.can_remove_symbol('AAPL')
```

---

## Phase 5: Data Breadth ✅

### 5.1 Streaming Data Adapter

**File**: `src/streaming_adapter.py`

**Key Features**:
- Multi-provider support (Polygon, Alpaca, IEX)
- Quote and trade streaming
- Real-time bar aggregation
- Handler registration pattern

**Usage**:
```python
from src.streaming_adapter import StreamingDataAdapter

adapter = StreamingDataAdapter(config)

# Register handlers
def on_quote(symbol, quote_data):
    logger.info(f"Quote: {symbol} bid={quote_data['bid']} ask={quote_data['ask']}")

def on_bar(symbol, bar_data):
    logger.info(f"Bar: {symbol} close={bar_data['close']} volume={bar_data['volume']}")

adapter.register_quote_handler(on_quote)
adapter.register_bar_handler(on_bar)

# Start streaming (async)
import asyncio
asyncio.run(adapter.start_stream(['AAPL', 'MSFT', 'GOOGL']))
```

---

## Integration Checklist

### Step 1: Update Configuration
Add to `config/config.yaml`:

```yaml
# Enable ML training
ml_training:
  enabled: true
  training_schedule: "weekly"

# Configure rate limiting
execution:
  max_api_calls_per_minute: 200
  enable_time_slicing: true

# Enable streaming data (optional)
alpaca:
  data_sources:
    primary: "alpaca"  # or "polygon"
    secondary: []
```

### Step 2: Wire Components in Main System

Example integration in `src/integrated_system.py`:

```python
from src.ml_pipeline import MLPipeline
from src.capital_allocator import CapitalAllocator
from src.exposure_manager import ExposureManager
from src.rate_limiter import RateLimiter
from src.slippage_model import SlippageModel
from src.dynamic_sizing import DynamicSizer

class TradingSystem:
    def __init__(self, config):
        # ... existing initialization ...
        
        # Phase 2: ML Pipeline
        self.ml_pipeline = MLPipeline(config)
        
        # Phase 3: Execution enhancements
        self.slippage_model = SlippageModel(config)
        self.dynamic_sizer = DynamicSizer(config)
        self.exposure_manager = ExposureManager(config)
        self.rate_limiter = RateLimiter(config)
        
        # Phase 4: Portfolio orchestration
        self.capital_allocator = CapitalAllocator(config)
        self.universe_rotation = UniverseRotationTracker(config)
```

### Step 3: Testing

Run existing tests to ensure no regressions:
```bash
# If tests exist
python -m pytest tests/

# Lint code
python -m pylint src/
```

---

## Configuration Examples

### Enable All Features

```yaml
ml_training:
  enabled: true
  training_schedule: "weekly"
  validation_method: "walk_forward"
  model_registry:
    enabled: true
    backend: "mlflow"

strategies:
  intraday_mean_reversion:
    enabled: true
    ml_model:
      enabled: true
      model_type: "lightgbm"
      features: ["zscore", "rsi", "atr", "volume_zscore"]

execution:
  max_api_calls_per_minute: 200
  enable_time_slicing: true
  enable_toxic_time_filtering: true

risk:
  max_gross_exposure_pct_intraday: 50
  max_gross_exposure_pct_swing: 100
```

---

## Testing Recommendations

1. **Unit Tests**: Test each new component in isolation
2. **Integration Tests**: Test component interactions
3. **Backtest Validation**: Run backtests with new features enabled
4. **Paper Trading**: Validate in paper trading before live
5. **Performance Tests**: Monitor latency and throughput

---

## Monitoring

Key metrics to monitor:

1. **ML Models**:
   - Prediction accuracy
   - Feature drift (KS, PSI)
   - Model age

2. **Execution**:
   - Slippage vs forecast
   - API rate limit utilization
   - Order rejection rate

3. **Portfolio**:
   - Capital allocation distribution
   - Universe rotation frequency
   - Exposure utilization

4. **Data**:
   - Data staleness
   - Data quality scores
   - Stream connection uptime

---

## Support and Maintenance

- All components include comprehensive logging
- Configuration-driven design for easy tuning
- Error handling with graceful degradation
- Persistent storage for models and state
- Extensible architecture for future enhancements
