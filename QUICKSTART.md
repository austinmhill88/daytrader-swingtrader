# Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will get you up and running with the trading system quickly.

## Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/austinmhill88/daytrader-swingtrader.git
cd daytrader-swingtrader

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure

### Trading Configuration

Create `.env` file:
```bash
# Copy example
copy .env.example .env  # Windows
# or
cp .env.example .env    # Linux/Mac
```

Edit `.env`:
```bash
APCA_API_KEY_ID=your_alpaca_key_here
APCA_API_SECRET_KEY=your_alpaca_secret_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

Get Alpaca keys:
1. Sign up at https://alpaca.markets
2. Use paper trading account (free)
3. Get API keys from dashboard

### AI Configuration (Optional)

If you want AI features:

1. **Download a model** (recommended: Qwen 2.5 3B):
   - Go to: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
   - Download: `qwen2.5-3b-instruct-q4_k_m.gguf` (~2GB)
   - Save to: `C:\models\` (or create MODELS_DIR)

2. **Edit `config/ai-coder.yaml`**:
   ```yaml
   models:
     - alias: "qwen2.5:3b-instruct"
       gguf_path: "C:/models/qwen2.5-3b-instruct-q4_k_m.gguf"
       ctx_size: 2048
       gpu_layers: -1  # Use GPU
   
   default_model: "qwen2.5:3b-instruct"
   root_dir: "C:/daytrader-swingtrader"  # Your project path
   ```

3. **Enable in `config/config.yaml`**:
   ```yaml
   ai_server:
     enabled: true
   ```

## Step 3: Verify Setup

Run verification script:
```bash
python verify_setup.py
```

This checks:
- ✓ Python version
- ✓ Dependencies installed
- ✓ Configuration files present
- ✓ Directory structure
- ✓ GPU support (if available)

Fix any issues before proceeding.

## Step 4: Run the System

### Option A: Trading Bot Only (No GUI)

```bash
python -m src.main --config config/config.yaml --paper
```

### Option B: With GUI (Recommended for Windows)

**Double-click:**
- `start_trading_gui.bat` - Trading + GUI
- `start_complete_system.bat` - Trading + AI + GUI
- `start_ai_server.bat` - AI server only

**Or command line:**
```bash
# Trading + GUI
python launcher.py --paper

# With AI
python launcher.py --with-ai --paper
```

### Option C: AI Server Only

```bash
python -m server.main
```

Then access at: http://127.0.0.1:8000

## Step 5: Monitor

### With GUI
- **Dashboard Tab**: View positions, P&L, signals
- **Portfolio Tab**: Track equity curve, performance
- **AI Console Tab**: Ask questions, get insights

### Without GUI
Check logs:
```bash
# Windows
type logs\runtime.log
tail logs\trades.log  # if tail available

# Linux/Mac
tail -f logs/runtime.log
tail -f logs/trades.log
```

## Common Issues

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Alpaca authentication failed
- Check API keys in .env
- Verify paper trading URL
- Ensure keys are for paper account

### GUI won't start
```bash
pip install PyQt6 pyqtgraph matplotlib
```

### AI server errors
- Check model file exists
- Verify path in ai-coder.yaml
- For GPU issues, see AI_SERVER_GUIDE.md

### No trades executing
- Market must be open (9:30-16:00 ET, Mon-Fri)
- Check strategies are enabled in config.yaml
- Verify universe has tradeable symbols
- Check risk limits aren't too restrictive

## Safety Reminders

⚠️ **Important:**
- Always start with **paper trading** (`--paper` flag)
- Monitor for 2-4 weeks before considering live
- Start with small capital if going live
- Never risk more than you can afford to lose
- Keep kill-switch enabled (default)

## Next Steps

1. **Watch it run**: Monitor for a few hours
2. **Check logs**: Review runtime.log and trades.log
3. **Adjust config**: Tune risk limits, strategies
4. **Backtest**: Run backtest to validate strategies
5. **Read docs**: Study all .md files in docs/

## Getting Help

- **Setup Issues**: Run `python verify_setup.py`
- **Trading Issues**: Check `logs/runtime.log`
- **AI Issues**: Check `logs/ai_tools.log`
- **GUI Issues**: See GUI_GUIDE.md
- **AI Server**: See AI_SERVER_GUIDE.md

## System Architecture

```
┌─────────────────────────────────────────┐
│         Trading GUI (Optional)           │
│  ┌────────┬──────────┬──────────────┐   │
│  │Dashboard│Portfolio │ AI Console  │   │
│  └────────┴──────────┴──────────────┘   │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         Trading System (Core)            │
│  ┌──────────────────────────────────┐   │
│  │ Strategies │ Risk │ Execution   │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │ Data Feed │ Portfolio │ Metrics  │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│      AI Runtime (Optional)               │
│  ┌──────────────────────────────────┐   │
│  │ Model API │ Web Tools │ File Ops │   │
│  └──────────────────────────────────┘   │
│  ┌──────────────────────────────────┐   │
│  │  Llama.cpp + GGUF + GPU (RTX)   │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Quick Reference

### Start Commands
```bash
# Basic trading
python -m src.main --paper

# With GUI
python launcher.py --paper

# Full system
python launcher.py --with-ai --paper

# AI only
python -m server.main

# Verify setup
python verify_setup.py
```

### Log Files
- `logs/runtime.log` - System activity
- `logs/trades.log` - Trade execution
- `logs/errors.log` - Errors only
- `logs/ai_tools.log` - AI requests

### Config Files
- `config/config.yaml` - Trading system
- `config/ai-coder.yaml` - AI server
- `.env` - API keys

### API Endpoints (AI Server)
- `http://127.0.0.1:8000/` - Health check
- `http://127.0.0.1:8000/api/version` - Version info
- `http://127.0.0.1:8000/api/chat` - Chat completion
- `http://127.0.0.1:8000/web/fetch` - Fetch pages

---

**You're now ready to trade! Start with paper mode and monitor carefully.**
