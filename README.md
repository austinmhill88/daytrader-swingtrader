# Alpaca Swing/Day Trading Bot - Production Grade

A robust, production-ready automated trading system with ML-driven strategies, comprehensive risk management, and fail-safe mechanisms. Now featuring autonomous operation with Phase 1-4 enhancements.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ⚠️ Important Disclaimers

- **Trading involves significant risk of loss**. This software is provided for educational purposes only.
- Past performance does not guarantee future results.
- This system is designed for U.S. equities with Alpaca's paper and live endpoints.
- Always confirm broker permissions, tax implications, and compliance requirements.
- **NEVER run live trading without extensive paper trading validation first**.

## 🎯 Features

### Core Trading System
- **Multi-Strategy Framework**: Intraday mean reversion and swing trend following strategies
- **Real-time Data Streaming**: WebSocket-based live market data with automatic reconnection
- **Comprehensive Risk Management**: Multiple layers of risk controls and kill-switch functionality
- **Smart Order Execution**: Position sizing, bracket orders, and slippage management
- **Portfolio Management**: Real-time tracking of positions, P&L, and exposures

### Phase 1: Foundation (Data & Infrastructure)
- **Multi-Source Data Layer**: Primary/fallback data sources (Alpaca, Polygon, Tiingo, IEX)
- **Parquet Storage**: Partitioned by symbol/date with versioning
- **Scheduler Framework**: Automated pre-market, EOD, and nightly tasks
- **Walk-Forward Backtesting**: Rolling train/test windows with realistic costs

### Phase 2: ML & Overfitting Prevention
- **Feature Store**: 30+ computed features with deterministic versioning
- **ML Model Training**: LightGBM, XGBoost, Random Forest with purged CV
- **Overfitting Controls**: Purged time-series CV, embargo periods, walk-forward validation
- **Promotion Gates**: Min Sharpe, max DD, min trades, min win rate checks
- **Universe Analytics**: Liquidity scoring, volatility bucketing, tier management

### Phase 3: Execution & Autonomy
- **Admin Controls**: Pause/resume, emergency halt, manual interventions
- **Self-Healing**: Watchdogs, automatic recovery, health monitoring
- **Enhanced Metrics**: Latency tracking, fill quality, slippage measurement
- **Connection Management**: Exponential backoff reconnection

### Phase 4: Production Integration
- **Integrated System**: Unified orchestration of all subsystems
- **Command Interface**: API/CLI for manual controls
- **System Status**: Comprehensive health and performance monitoring

### Risk Controls
- Daily maximum drawdown limits with automatic kill-switch
- Per-trade risk limits based on ATR
- Position size limits (percentage of equity)
- Gross and net exposure caps
- Maximum positions and trades per day

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/austinmhill88/daytrader-swingtrader.git
cd daytrader-swingtrader
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Edit .env with your Alpaca API credentials
```

Set these environment variables in `.env`:
```bash
APCA_API_KEY_ID=your_key_id_here
APCA_API_SECRET_KEY=your_secret_key_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
# Optional: Finnhub for earnings blackout calendar
FINNHUB_API_KEY=your_finnhub_key_here
```

### 3. Paper Trading (REQUIRED)

```bash
python -m src.main --config config/config.yaml --paper
```

**Monitor for at least 2-4 weeks before considering live trading!**

## 📊 System Architecture

### Components (Phase 1-4)

### Components (Phase 1-4)

**Core Trading (Original)**
1. **Data Feed** - Live streaming via WebSocket with automatic reconnection
2. **Strategies** - Intraday mean reversion and swing trend following
3. **Risk Manager** - Pre-trade checks and kill-switch mechanism  
4. **Execution Engine** - Smart position sizing and bracket orders
5. **Portfolio Manager** - Real-time position tracking and P&L
6. **Orchestrator** - Main trading loop and session management

**Phase 1: Foundation**
7. **Multi-Source Data** - Redundant data providers with fallback
8. **Data Storage** - Parquet-based storage with partitioning
9. **Scheduler** - Automated task orchestration
10. **Walk-Forward Backtester** - Realistic validation framework

**Phase 2: ML & Analytics**
11. **Feature Store** - Deterministic feature computation
12. **ML Trainer** - Model training with overfitting prevention
13. **Universe Analytics** - Liquidity scoring and tier management

**Phase 3: Operations**
14. **Admin Controls** - Manual trading controls and emergency halt
15. **Self-Healing** - Watchdogs and automatic recovery
16. **Metrics Tracker** - Latency and fill quality monitoring

**Phase 4: Integration**
17. **Integrated System** - Unified orchestration layer
18. **Command Interface** - API/CLI for operations

## 📚 Strategy Details

### Intraday Mean Reversion
- Z-score based entries relative to VWAP
- RSI confirmation
- ATR-based stop loss
- End of day flatten

### Swing Trend Following  
- EMA crossover signals
- ADX trend strength filter
- Pullback entries
- Trailing stop loss

## 🔧 Configuration Highlights

### Phase 1-4 Configuration Sections

```yaml
# Data Sources (Phase 1)
data_sources:
  primary: "alpaca"
  secondary: []  # polygon, tiingo, iex

# Data Storage (Phase 1)
data_storage:
  format: "parquet"
  compression: "snappy"
  partitioning:
    by_symbol: true
    by_date: true

# Scheduler (Phase 1)
scheduler:
  enabled: false
  tasks:
    pre_market:
      time: "08:30"
      actions: ["data_sync", "universe_refresh", "model_check"]
    end_of_day:
      time: "16:30"
      actions: ["flatten_intraday", "generate_reports", "backup_data"]

# ML Training (Phase 2)
ml_training:
  enabled: false
  validation_method: "walk_forward"
  promotion_gates:
    min_sharpe: 1.0
    max_drawdown_pct: 10.0

# Backtesting (Phase 1-2)
backtesting:
  enable_walk_forward: true
  train_window_days: 252
  test_window_days: 63
  enable_purged_cv: false
  embargo_pct: 0.01

# Metrics (Phase 3)
metrics:
  track_latency: true
  track_fill_quality: true
  track_slippage: true
```

## 🐳 Docker Deployment

```bash
docker build -t trading-bot .
docker run -d --name trading-bot \
  -e APCA_API_KEY_ID=your_key \
  -e APCA_API_SECRET_KEY=your_secret \
  -e APCA_API_BASE_URL=https://paper-api.alpaca.markets \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  trading-bot
```

## 📈 Monitoring

### Admin Commands

```python
from src.integrated_system import TradingSystem

system = TradingSystem('config/config.yaml')
system.start()

# Manual controls
system.admin_controls.pause_trading(reason="Market volatility")
system.admin_controls.resume_trading(confirm=True)
system.admin_controls.emergency_halt(reason="Broker issue")

# Get status
status = system.get_system_status()
```

### Health Checks

```python
# System health
health = system.self_healing.get_system_health()

# Component status
system.self_healing.check_health('data_feed')
```

### Metrics

```python
# Latency summary
latency = system.metrics_tracker.get_latency_summary(minutes=60)

# Fill quality
fills = system.metrics_tracker.get_fill_quality_summary(minutes=60)
```

Logs are written to `logs/` directory:
- `runtime.log` - General application logs
- `errors.log` - Error-level logs  
- `trades.log` - Trade activities

## 🔐 Security Best Practices

1. Never commit API keys - Use environment variables only
2. Use paper trading first - Validate thoroughly
3. Start with small capital - Scale up gradually
4. Monitor closely - Especially during first weeks
5. Set conservative limits - Better safe than sorry

## 🎓 Phase Implementation Summary

### Phase 1: Foundation ✅
- Multi-source data redundancy
- Parquet storage with versioning
- Automated scheduling
- Walk-forward backtesting

### Phase 2: ML & Analytics ✅
- Feature store (30+ features)
- ML training with purged CV
- Promotion gates for quality
- Universe analytics

### Phase 3: Operations ✅
- Admin controls & emergency halt
- Self-healing watchdogs
- Enhanced metrics tracking
- Connection resilience

### Phase 4: Integration ✅
- Unified system orchestration
- Command interface
- Comprehensive status monitoring
- Production-ready architecture

## ⚖️ Legal

This software is provided "as is" without warranty of any kind. Trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through use of this software.

---

**Remember: Paper trade first, start small, and never risk more than you can afford to lose!**
