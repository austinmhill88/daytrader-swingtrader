# Alpaca Swing/Day Trading Bot

A robust, production-ready automated trading system leveraging the Alpaca Trading API with sophisticated swing and day trading strategies, comprehensive risk management, and fail-safe mechanisms.

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

### Risk Controls
- Daily maximum drawdown limits with automatic kill-switch
- Per-trade risk limits based on ATR
- Position size limits (percentage of equity)
- Gross and net exposure caps
- Maximum positions and trades per day

### Technical Features
- Event-driven architecture for reliability
- Structured logging with rotation and retention
- Configuration management with environment variables
- Docker support for containerized deployment
- Graceful shutdown handling

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
```

### 3. Paper Trading (REQUIRED)

```bash
python -m src.main --config config/config.yaml --paper
```

**Monitor for at least 2-4 weeks before considering live trading!**

## 📊 System Architecture

### Components

1. **Data Feed** - Live streaming via WebSocket with automatic reconnection
2. **Strategies** - Intraday mean reversion and swing trend following
3. **Risk Manager** - Pre-trade checks and kill-switch mechanism  
4. **Execution Engine** - Smart position sizing and bracket orders
5. **Portfolio Manager** - Real-time position tracking and P&L
6. **Orchestrator** - Main trading loop and session management

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

## ⚖️ Legal

This software is provided "as is" without warranty of any kind. Trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through use of this software.

---

**Remember: Paper trade first, start small, and never risk more than you can afford to lose!**
