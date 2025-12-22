# Alpaca Swing/Day Trading Bot

This repository implements a robust, modular trading system using the Alpaca Trading API with intraday mean-reversion and swing trend-following strategies, regime detection, strict risk management, and realistic execution.

## Quick Start

1. Set environment variables:
   - `APCA_API_KEY_ID`
   - `APCA_API_SECRET_KEY`
   - `APCA_API_BASE_URL` (paper: `https://paper-api.alpaca.markets`, live: `https://api.alpaca.markets`)
2. Install dependencies: `pip install -r requirements.txt`
3. Configure `config/config.yaml`
4. Backtest: `python -m backtest.backtester --config config/config.yaml`
5. Paper trade: `python -m src.main --config config/config.yaml --paper`
6. Live trade: `python -m src.main --config config/config.yaml --live`

## Disclaimer

Trading involves risk. This code is for educational purposes and must be validated extensively before live deployment. Past performance does not guarantee future results.
