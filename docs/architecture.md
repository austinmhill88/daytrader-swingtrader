# System Architecture

## Overview
This document describes the architecture of the autonomous trading system, including all components, data flows, and integration points across Phases 1-4.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Trading System                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Data Layer  │───▶│ Feature Store│───▶│  ML Models   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                    │                    │                  │
│         ▼                    ▼                    ▼                  │
│  ┌──────────────────────────────────────────────────────┐          │
│  │           Strategy Engine & Signal Generation         │          │
│  └──────────────────────────────────────────────────────┘          │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │Risk Manager  │◀───│ Execution    │───▶│  Portfolio   │          │
│  │& Kill Switch │    │    Engine    │    │   Manager    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                    │                    │                  │
│         └────────────────────┴────────────────────┘                 │
│                            │                                         │
│                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              Broker API (Alpaca)                         │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Metrics &  │    │    Admin     │    │ Self-Healing │          │
│  │  Monitoring  │    │   Controls   │    │  & Watchdog  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer

#### 1.1 Data Sources (`src/data_sources.py`)
**Purpose**: Multi-source data acquisition with failover

**Responsibilities**:
- Primary data from Alpaca (real-time WebSocket, historical REST API)
- Fallback to Polygon, Tiingo, IEX on primary failure
- Data quality validation and sanity checks
- Rate limiting and error handling

**Inputs**: API credentials, symbol lists
**Outputs**: Raw OHLCV data (bars), quotes, trades

**Key Methods**:
- `fetch_bars()` - Historical OHLCV data
- `stream_quotes()` - Real-time market data
- `handle_failover()` - Switch to backup sources

#### 1.2 Data Storage (`src/data_storage.py`)
**Purpose**: Persistent storage with versioning

**Responsibilities**:
- Parquet-based storage with symbol/date partitioning
- Data versioning and snapshot management
- Compression (Snappy) for efficiency
- Metadata and manifest generation

**Storage Schema**:
```
data/parquet/
├── ohlcv/
│   ├── symbol=AAPL/
│   │   └── date=2024-01-15/data.parquet
│   └── symbol=MSFT/...
└── features/
    └── symbol=AAPL/...
```

**Key Methods**:
- `write_bars()` - Store OHLCV data
- `read_bars()` - Retrieve historical data
- `create_snapshot()` - Generate versioned snapshots

#### 1.3 Data Feed (`src/data_feed.py`)
**Purpose**: Real-time market data streaming

**Responsibilities**:
- WebSocket connection to Alpaca streams (IEX or SIP)
- Automatic reconnection with exponential backoff
- Data buffering and throttling
- Stream health monitoring

**Key Methods**:
- `connect()` - Establish WebSocket connection
- `subscribe()` - Subscribe to symbols
- `on_bar()`, `on_quote()`, `on_trade()` - Event handlers
- `reconnect()` - Handle disconnections

### 2. Feature Store (`src/feature_store.py`)

**Purpose**: Deterministic feature engineering pipeline

**Responsibilities**:
- Compute 30+ technical indicators and features
- Feature versioning and reproducibility
- Incremental feature updates
- Feature validation and drift detection

**Feature Categories**:
1. **Price-based**: Returns, log returns, price ratios
2. **Technical**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
3. **Volume**: Volume ratios, VWAP, volume z-score
4. **Volatility**: Realized volatility, Garman-Klass, Parkinson
5. **Microstructure**: Bid-ask spread, tick imbalance
6. **Cross-sectional**: Relative strength, sector correlation

**Key Methods**:
- `compute_features()` - Generate features for a symbol
- `validate_features()` - Check for NaN, inf, outliers
- `get_feature_version()` - Return feature set version hash

### 3. Universe Analytics (`src/universe_analytics.py`)

**Purpose**: Dynamic universe construction and management

**Responsibilities**:
- Daily universe refresh based on liquidity criteria
- Liquidity scoring (ADV, spread, volatility)
- Volatility bucketing (low/medium/high)
- Shortability validation
- Earnings blackout detection
- Multi-tier universe (Core 200, Extended 1000)

**Selection Criteria**:
- Average Dollar Volume (ADV) > threshold
- Average spread < 50 bps
- Price range: $5 - $500
- Market cap > $100M
- No earnings within 2 days
- Shortable (if shorting enabled)

**Key Methods**:
- `build_universe()` - Generate daily universe
- `score_liquidity()` - Compute liquidity metrics
- `assign_tiers()` - Classify symbols into tiers
- `check_shortability()` - Validate short availability

### 4. ML Models (`src/ml_trainer.py`, `src/models.py`)

**Purpose**: Machine learning model training and inference

**Responsibilities**:
- Model training with walk-forward validation
- Purged k-fold cross-validation with embargo
- Hyperparameter tuning
- Anti-overfitting diagnostics (PBO, Deflated Sharpe)
- Model versioning and registry integration
- Feature importance analysis (SHAP)

**Supported Models**:
- LightGBM (primary, GPU-accelerated)
- XGBoost (alternative, GPU support)
- Random Forest (baseline)

**Validation Methodology**:
- **Walk-forward**: Train 252 days, test 63 days, step 21 days
- **Purged CV**: Remove leakage across folds with embargo
- **Embargo**: 1% of data between train/test
- **Out-of-sample gates**: Must pass promotion thresholds

**Key Methods**:
- `train_model()` - Train with CV
- `validate_model()` - Run anti-overfitting checks
- `compute_shap_values()` - Feature importance
- `promote_model()` - Register approved models

### 5. Strategy Engine

#### 5.1 Base Strategy (`src/strategies/base.py`)
**Purpose**: Abstract base class for all strategies

**Interface**:
- `generate_signals()` - Produce buy/sell/hold signals
- `calculate_position_size()` - Determine trade size
- `validate_signal()` - Pre-trade checks

#### 5.2 Intraday Mean Reversion (`src/strategies/intraday_mean_reversion.py`)
**Purpose**: Mean reversion on minute bars

**Logic**:
- Entry: Z-score > 2.0 relative to VWAP, RSI confirmation
- Exit: Z-score < 0.5 or stop loss (1.5x ATR) or end-of-day
- Holding period: Minutes to hours (flatten EOD)

**ML Enhancement** (optional):
- ML model predicts probability of mean reversion
- Combine z-score signal with ML probability
- Adjust position size based on model confidence

#### 5.3 Swing Trend Following (`src/strategies/swing_trend_following.py`)
**Purpose**: Multi-day trend following

**Logic**:
- Entry: EMA crossover (20/50), ADX > 20, pullback entry
- Exit: Trailing stop (3x ATR) or EMA cross in opposite direction
- Holding period: 1-10 days

**ML Enhancement** (optional):
- ML model predicts trend continuation probability
- Filter trades based on model score
- Dynamic stop adjustments

### 6. Risk Manager (`src/risk_manager.py`)

**Purpose**: Pre-trade and intra-day risk controls

**Risk Checks**:
1. **Daily Drawdown**: < 2% from day's high
2. **Per-trade Risk**: < 0.4% of equity per trade
3. **Position Size**: < 5% of equity per position
4. **Gross Exposure**: < 50% (intraday) or 100% (swing)
5. **Short Exposure**: < 50% of gross
6. **Max Positions**: < 30 total
7. **Consecutive Losses**: Pause after 3 losses
8. **Max Trades/Day**: < 100

**Kill Switch Triggers**:
- Daily drawdown breach
- Broker API errors (multiple retries failed)
- Data feed disconnection > 2 minutes
- Manual emergency halt

**Key Methods**:
- `check_trade_risk()` - Pre-trade validation
- `update_pnl()` - Track daily P&L
- `should_trigger_kill_switch()` - Monitor thresholds
- `flatten_all_positions()` - Emergency exit

### 7. Execution Engine (`src/execution_engine.py`)

**Purpose**: Smart order execution and management

**Order Types**:
- Limit orders (default, 5 bps offset)
- Market orders (emergencies only)
- Bracket orders (entry + stop + take profit)
- Time-sliced orders (large sizes)

**Smart Execution**:
- Avoid toxic windows (first 2-5 min, last 5 min)
- Adaptive limit offsets based on spread/volatility
- Partial fill handling (>80% acceptance)
- Order timeout management (60 seconds)
- Retry logic with exponential backoff

**Key Methods**:
- `execute_signal()` - Place order for strategy signal
- `place_bracket_order()` - Entry with stop/target
- `monitor_fills()` - Track order status
- `calculate_limit_offset()` - Dynamic pricing

### 8. Portfolio Manager (`src/portfolio.py`)

**Purpose**: Real-time position and P&L tracking

**Responsibilities**:
- Track all open positions
- Compute realized and unrealized P&L
- Calculate exposures (gross, net, long, short)
- Reconcile with broker positions (daily)

**Metrics**:
- Daily P&L ($ and %)
- Intraday high-water mark
- Drawdown from HWM
- Position-level P&L
- Strategy-level attribution

**Key Methods**:
- `update_positions()` - Sync with broker
- `calculate_pnl()` - Compute P&L
- `get_exposures()` - Return exposure metrics
- `reconcile()` - Match internal vs broker state

### 9. Backtester (`backtest/backtester.py`)

**Purpose**: Historical strategy validation

**Features**:
- Event-driven simulation (bar-by-bar)
- Slippage modeling (spread-based, volatility-adjusted)
- Transaction costs (commission, borrow costs)
- Partial fills and limit order queue simulation
- Realistic latency modeling

**Metrics Computed**:
- Sharpe ratio, Sortino ratio
- Max drawdown, Calmar ratio
- Win rate, average win/loss
- Turnover, slippage impact
- Exposure statistics

**Key Methods**:
- `run_backtest()` - Execute simulation
- `apply_slippage()` - Model execution costs
- `generate_performance_report()` - Summary metrics

### 10. Walk-Forward Validation (`backtest/walk_forward.py`)

**Purpose**: Anti-overfitting validation framework

**Process**:
1. Train model on window (e.g., 252 days)
2. Test on next window (e.g., 63 days)
3. Step forward (e.g., 21 days)
4. Repeat across full history
5. Aggregate out-of-sample results

**Gates**:
- Min Sharpe > 1.0
- Max drawdown < 10%
- Min trades > 100
- Min win rate > 45%

**Key Methods**:
- `run_walk_forward()` - Execute rolling validation
- `compute_aggregate_metrics()` - Combine results
- `check_promotion_gates()` - Validate thresholds

### 11. Admin Controls (`src/admin_controls.py`)

**Purpose**: Manual intervention and emergency controls

**Commands**:
- `pause_trading()` - Halt new entries, allow exits
- `resume_trading()` - Resume normal operation
- `emergency_halt()` - Flatten all, stop everything
- `override_risk_limit()` - Temporary limit adjustment

**Audit Trail**:
- All commands logged with timestamp, user, reason
- Requires confirmation for critical actions

**Key Methods**:
- `pause_trading()`, `resume_trading()`
- `emergency_halt()`, `flatten_all()`
- `get_system_state()` - Current status

### 12. Self-Healing (`src/self_healing.py`)

**Purpose**: Watchdogs and automatic recovery

**Watchdogs**:
- Data feed health (< 30s since last bar)
- Broker API connectivity (periodic ping)
- Execution latency (< 1s target)
- Memory usage (< 80% of 64GB)
- Disk space (> 10% free)

**Recovery Actions**:
- Reconnect data streams
- Switch to backup data sources
- Restart failed components
- Alert operators for critical issues

**Key Methods**:
- `check_health()` - Monitor all components
- `trigger_recovery()` - Execute recovery actions
- `send_alert()` - Notify operators

### 13. Metrics Tracker (`src/metrics_tracker.py`)

**Purpose**: Performance and operational metrics

**Tracked Metrics**:
- **Latency**: Order placement, data feed, execution
- **Fill Quality**: Fill rate, price improvement, adverse selection
- **Slippage**: Realized vs expected execution price
- **System**: CPU, memory, network, disk I/O

**Export**:
- Prometheus exporter (port 9101)
- JSON logs for archival
- Real-time dashboards (Grafana)

**Key Methods**:
- `record_latency()`, `record_fill()`
- `get_latency_summary()`, `get_fill_quality_summary()`
- `export_prometheus()` - Metrics endpoint

### 14. Scheduler (`src/scheduler.py`)

**Purpose**: Automated task orchestration

**Scheduled Tasks**:
- **Pre-market (08:30 ET)**:
  - Sync data from previous day
  - Refresh universe (liquidity, shortability)
  - Validate model freshness
  - Check for corporate actions
  
- **End-of-day (16:30 ET)**:
  - Flatten intraday positions
  - Generate daily reports
  - Backup data
  - Reconcile positions with broker
  
- **Nightly (02:00 ET)**:
  - Retrain models (if scheduled)
  - Run walk-forward backtests
  - Clean up old logs/data
  - Generate performance analytics

**Key Methods**:
- `schedule_task()` - Register task
- `run_task()` - Execute task
- `get_task_status()` - Check execution history

### 15. Integrated System (`src/integrated_system.py`)

**Purpose**: Unified orchestration of all components

**Lifecycle**:
1. **Initialization**: Load config, connect to broker, initialize components
2. **Pre-market**: Run scheduler tasks, prepare for trading
3. **Market Hours**: Run main trading loop, process signals, execute trades
4. **Post-market**: Clean up, generate reports, prepare for next day
5. **Shutdown**: Flatten positions, close connections, archive logs

**Main Loop** (simplified):
```python
while market_open and not kill_switch:
    1. Receive market data
    2. Update features
    3. Generate signals from strategies
    4. Check risk limits
    5. Execute approved trades
    6. Update portfolio state
    7. Monitor health and metrics
    8. Check for manual commands
```

**Key Methods**:
- `start()` - Initialize and start system
- `stop()` - Graceful shutdown
- `run_trading_loop()` - Main execution loop
- `get_system_status()` - Comprehensive status report

## Data Flows

### Trading Flow (Live)
```
Market Data (Alpaca WebSocket)
    │
    ▼
Data Feed (real-time bars/quotes)
    │
    ▼
Feature Store (compute indicators)
    │
    ▼
Strategy Engine (generate signals)
    │
    ▼
Risk Manager (validate trades)
    │
    ▼
Execution Engine (place orders)
    │
    ▼
Broker API (Alpaca)
    │
    ▼
Portfolio Manager (update positions)
```

### Model Training Flow
```
Historical Data (Data Storage)
    │
    ▼
Feature Store (compute training features)
    │
    ▼
ML Trainer (train with purged CV)
    │
    ▼
Walk-Forward Validator (out-of-sample test)
    │
    ▼
Anti-overfitting Gates (PBO, Deflated Sharpe)
    │
    ├─ Pass ─▶ Model Registry (promote)
    │
    └─ Fail ─▶ Reject and log
```

### Nightly Workflow
```
Scheduler (02:00 ET)
    │
    ├─▶ Data Sync (fetch missing bars)
    │
    ├─▶ Universe Builder (refresh liquidity scores)
    │
    ├─▶ ML Trainer (retrain models if scheduled)
    │
    ├─▶ Walk-Forward Backtest (validate new models)
    │
    ├─▶ Cleanup (archive logs, old data)
    │
    └─▶ Generate Reports (email/Slack summary)
```

## Deployment Architecture

### Development Environment
- Python 3.11+ on local machine
- SQLite for development database
- File-based storage for data
- Paper trading with Alpaca

### Production Environment (Local PC)
- **OS**: Ubuntu LTS or WSL2
- **CPU**: AMD Ryzen 9 9950x (16 cores, 32 threads)
- **GPU**: RTX 5070 OC (ML training acceleration)
- **RAM**: 64GB DDR5
- **Storage**: 2TB NVMe SSD (Samsung 990 Pro)
- **Network**: Wired Ethernet, UPS for power continuity

### Docker Deployment
```
Docker Container
├── Python 3.11
├── Trading System (all components)
├── Volume Mounts:
│   ├── /app/data (persistent storage)
│   ├── /app/logs (log files)
│   └── /app/config (configuration)
└── Environment Variables:
    ├── APCA_API_KEY_ID
    ├── APCA_API_SECRET_KEY
    └── APCA_API_BASE_URL
```

### Resource Allocation
- **CPU Pinning**: Assign cores for data processing vs trading loop
- **GPU**: LightGBM/XGBoost training, SHAP computation
- **Memory**: ~8GB for live trading, up to 32GB for backtests
- **Disk I/O**: Parquet reads/writes on NVMe for low latency

## Integration Points

### External APIs
- **Alpaca Markets API**: Trading, market data, account info
- **Polygon.io**: Premium market data (fallback)
- **Tiingo**: Historical data (fallback)
- **IEX Cloud**: Free tier data (fallback)

### Monitoring
- **Prometheus**: Metrics scraping (port 9101)
- **Grafana**: Dashboard visualization
- **Slack**: Alerts and notifications
- **Email**: Daily reports (optional)

### Model Registry
- **MLflow**: Model versioning, tracking, promotion
- **Storage**: Local filesystem or S3-compatible

## Security Considerations

### Secrets Management
- Never commit API keys to Git
- Use environment variables (.env file)
- Rotate keys quarterly
- Use separate keys for paper vs live trading

### Network Security
- No inbound ports required (system initiates all connections)
- Use HTTPS/WSS for all API calls
- Validate SSL certificates

### Access Control
- File permissions: 0600 for .env, 0700 for data directories
- Run as non-root user in Docker
- Admin commands require confirmation

## Disaster Recovery

### System Failure
1. Self-healing attempts automatic recovery
2. If recovery fails, emergency_halt() triggered
3. All positions flattened (if possible)
4. Alert operators via Slack/email
5. Manual intervention required

### Data Loss
1. Restore from latest snapshot (daily backups)
2. Re-fetch missing market data from broker
3. Reconcile positions with broker API
4. Resume trading after validation

### Broker Outage
1. Flatten positions using broker's web interface (manual)
2. Wait for broker restoration
3. Reconcile state before resuming

## Performance Targets

### Latency
- Data feed lag: < 100ms (median)
- Signal generation: < 50ms per symbol
- Order placement: < 200ms
- End-to-end (signal to order): < 500ms

### Throughput
- Process 500 symbols at 1-minute bars
- Generate signals for 30 strategies
- Execute up to 100 orders per day

### Availability
- Target uptime: 99.5% during market hours
- Planned maintenance: Pre-market or post-market only

## Testing Strategy

### Unit Tests
- Individual component methods
- Mock external dependencies (broker API, data sources)

### Integration Tests
- Multi-component workflows
- Paper trading validation

### End-to-End Tests
- Full system in paper trading mode
- Compare backtest vs paper trading results

### Performance Tests
- Latency benchmarks
- Load testing with full universe

## Future Enhancements

- [ ] Multi-broker support (Interactive Brokers, TD Ameritrade)
- [ ] Options strategies
- [ ] Futures trading
- [ ] Distributed execution (multiple machines)
- [ ] Real-time model retraining
- [ ] Reinforcement learning for execution
- [ ] Portfolio optimization (Markowitz, Black-Litterman)
- [ ] Event-driven strategies (earnings, news)

## References
- [Alpaca Markets API Docs](https://alpaca.markets/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Prometheus Exporter Guide](https://prometheus.io/docs/instrumenting/writing_exporters/)
- [Advances in Financial Machine Learning (Lopez de Prado)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
