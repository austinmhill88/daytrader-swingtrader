"""
Main trading bot orchestration and execution loop.
"""
import argparse
import signal
import sys
import time
import pandas as pd
from datetime import datetime, time as dtime
from typing import List, Dict, Optional
from loguru import logger

from src.config import load_config, validate_config
from src.logging_utils import setup_logging, log_signal
from src.alpaca_client import AlpacaClient
from src.portfolio import PortfolioState
from src.execution_engine import ExecutionEngine
from src.risk_manager import RiskManager
from src.data_feed import HistoricalDataFeed
from src.strategies.intraday_mean_reversion import IntradayMeanReversion
from src.strategies.swing_trend_following import SwingTrendFollowing
from src.models import Signal
from src.regime_detector import RegimeDetector
from src.notifier import AlertNotifier
from src.prometheus_exporter import PrometheusExporter


class TradingBot:
    """
    Main trading bot orchestrator.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trading bot.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        validate_config(self.config)
        
        # Setup logging
        logging_config = self.config.get('logging', {})
        setup_logging(
            logs_dir=self.config['storage']['logs_dir'],
            level=logging_config.get('level', 'INFO'),
            rotation=logging_config.get('rotation', '1 day'),
            retention=logging_config.get('retention', '30 days'),
            format_type=logging_config.get('format', 'json')
        )
        
        logger.info("=" * 80)
        logger.info("Trading Bot Starting...")
        logger.info(f"Environment: {self.config['environment']}")
        logger.info("=" * 80)
        
        # Initialize components
        self.client = AlpacaClient(
            key_id=self.config['alpaca']['key_id'],
            secret_key=self.config['alpaca']['secret_key'],
            base_url=self.config['alpaca']['base_url'],
            data_feed=self.config['alpaca']['data_feed']
        )
        
        self.portfolio = PortfolioState(self.client)
        self.execution_engine = ExecutionEngine(self.client, self.config['execution'])
        self.risk_manager = RiskManager(self.config['risk'], self.portfolio, self.client)
        
        # Initialize alert notifier
        self.notifier = AlertNotifier('config/alerts.yaml')
        
        # Initialize Prometheus exporter
        self.prometheus = PrometheusExporter(self.config)
        if self.prometheus.enabled:
            self.prometheus.start()
        
        # Link prometheus to execution engine
        self.execution_engine.prometheus = self.prometheus
        
        # Initialize regime detector
        self.regime_detector = RegimeDetector(self.config)
        
        # Initialize strategies
        self.strategies = []
        if self.config['strategies']['intraday_mean_reversion']['enabled']:
            self.strategies.append(
                IntradayMeanReversion(self.config['strategies']['intraday_mean_reversion'])
            )
        
        if self.config['strategies']['swing_trend_following']['enabled']:
            self.strategies.append(
                SwingTrendFollowing(self.config['strategies']['swing_trend_following'])
            )
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
        
        # Trading state
        self.is_running = False
        self.universe = self._build_universe()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum} - initiating graceful shutdown...")
        self.stop()
    
    def _build_universe(self) -> List[str]:
        """
        Build trading universe using UniverseAnalytics.
        
        Returns:
            List of stock symbols
        """
        try:
            from src.universe_analytics import UniverseAnalytics
            
            # Initialize UniverseAnalytics
            universe_analytics = UniverseAnalytics(self.config)
            
            # Get symbols from config as base candidates
            default_symbols = self.config['universe'].get('default_symbols', [])
            
            # For production, we would fetch data for many symbols
            # For now, start with default symbols and fetch their data
            hist_feed = HistoricalDataFeed(self.client)
            
            # Fetch recent data for universe analysis
            symbol_data = hist_feed.get_multi_symbol_bars(
                symbols=default_symbols,
                timeframe="1Day",
                limit=60  # 60 days for ADV calculation
            )
            
            # Build universe with filters
            universe_result = universe_analytics.build_universe_with_filters(
                symbol_data=symbol_data,
                tier='core',
                alpaca_client=self.client,
                apply_earnings_filter=True,
                apply_shortability_filter=False
            )
            
            symbols = universe_result.get('universe', [])
            
            # Fallback to default symbols if universe building fails
            if not symbols:
                logger.warning("Universe building returned empty, using default symbols")
                symbols = default_symbols
            
            logger.info(
                f"Universe: {len(symbols)} symbols - {', '.join(symbols[:5])}... | "
                f"Filtered: {universe_result.get('earnings_blackout_count', 0)} in earnings blackout"
            )
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error building universe: {e}, falling back to default symbols")
            symbols = self.config['universe'].get('default_symbols', [])
            logger.info(f"Universe (fallback): {len(symbols)} symbols - {', '.join(symbols[:5])}...")
            return symbols
    
    def _warmup_strategies(self) -> None:
        """Warm up strategies with historical data."""
        logger.info("Warming up strategies with historical data...")
        
        hist_feed = HistoricalDataFeed(self.client)
        
        # Fetch warmup data
        warmup_data = hist_feed.get_warmup_data(
            symbols=self.universe,
            timeframe="1Min",
            lookback_days=self.config['universe'].get('adv_lookback_days', 60)
        )
        
        # Warm up each strategy
        for strategy in self.strategies:
            try:
                strategy.warmup(warmup_data)
                logger.info(f"Strategy {strategy.name} warmed up successfully")
            except Exception as e:
                logger.error(f"Error warming up {strategy.name}: {e}")
        
        logger.info("Strategy warmup complete")
    
    def _is_market_hours(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open
        """
        try:
            return self.client.is_market_open()
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    def _process_signals(self, signals: List[Signal]) -> None:
        """
        Process signals and execute trades.
        
        Args:
            signals: List of trading signals
        """
        if not signals:
            return
        
        logger.info(f"Processing {len(signals)} signals")
        
        # Group signals by strategy
        strategy_signals = {}
        for signal in signals:
            if signal.strategy_name not in strategy_signals:
                strategy_signals[signal.strategy_name] = []
            strategy_signals[signal.strategy_name].append(signal)
        
        # Process each strategy's signals
        for strategy_name, strat_signals in strategy_signals.items():
            self._process_strategy_signals(strategy_name, strat_signals)
    
    def _process_strategy_signals(self, strategy_name: str, signals: List[Signal]) -> None:
        """
        Process signals from a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            signals: Signals from that strategy
        """
        # Create order intents
        intents = []
        
        for signal in signals:
            # Log signal
            log_signal(
                strategy=signal.strategy_name,
                symbol=signal.symbol,
                strength=signal.strength,
                confidence=signal.confidence,
                reason=signal.reason
            )
            
            # Skip exit signals (strength = 0) for now - would need position management
            if signal.strength == 0:
                # This is an exit signal - close position if we have one
                if self.portfolio.has_position(signal.symbol):
                    self.execution_engine.close_position(
                        signal.symbol,
                        reason=f"Strategy exit: {signal.reason}"
                    )
                continue
            
            # Get current price and ATR from signal metadata
            if not signal.metadata:
                logger.warning(f"Signal for {signal.symbol} missing metadata")
                continue
            
            current_price = signal.metadata.get('entry_price', 0)
            atr = signal.metadata.get('atr', current_price * 0.02)  # Fallback to 2% of price
            
            if current_price <= 0:
                logger.warning(f"Invalid price for {signal.symbol}")
                continue
            
            # Apply regime-based position sizing adjustment
            strategy_type = "intraday" if "intraday" in strategy_name.lower() else "swing"
            regime_multiplier = self.regime_detector.get_position_size_multiplier(strategy_type)
            adjusted_risk_pct = self.config['risk']['per_trade_risk_pct'] * regime_multiplier
            
            logger.debug(
                f"Regime adjustment | {strategy_name} | "
                f"Multiplier: {regime_multiplier:.2f}, "
                f"Risk: {self.config['risk']['per_trade_risk_pct']:.2f}% -> {adjusted_risk_pct:.2f}%"
            )
            
            # Create order intent
            intent = self.execution_engine.create_order_intent(
                signal=signal,
                portfolio=self.portfolio,
                current_price=current_price,
                atr=atr,
                per_trade_risk_pct=adjusted_risk_pct
            )
            
            if intent:
                intents.append(intent)
        
        if not intents:
            return
        
        # Run risk checks
        strategy_type = "intraday" if "intraday" in strategy_name.lower() else "swing"
        approved_intents, rejections = self.risk_manager.pre_trade_checks(
            intents, strategy_type
        )
        
        # Log rejections
        for rejection in rejections:
            logger.warning(f"Trade rejected: {rejection}")
        
        # Execute approved intents
        if approved_intents:
            logger.info(f"Executing {len(approved_intents)} approved orders")
            self.execution_engine.execute_intents(approved_intents)
    
    def _run_trading_loop(self) -> None:
        """Main trading loop for polling-based execution."""
        logger.info("Starting trading loop (polling mode)")
        
        poll_interval = 60  # Poll every 60 seconds
        last_eod_run = None
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check if market is open
                if not self._is_market_hours():
                    logger.debug("Market is closed")
                    time.sleep(60)
                    continue
                
                # Update portfolio
                self.portfolio.update()
                
                # Update regime detector with market data (every 5 minutes)
                if current_time.minute % 5 == 0:
                    for symbol in ['SPY', 'QQQ']:
                        if symbol in self.universe or symbol == 'SPY':
                            try:
                                # Get recent bars for regime calculation
                                bars_df = self.client.get_bars(
                                    symbol, 
                                    "1Min", 
                                    limit=100
                                )
                                if bars_df is not None and len(bars_df) > 0:
                                    df = pd.DataFrame({
                                        'close': [float(b.close) for b in bars_df],
                                        'high': [float(b.high) for b in bars_df],
                                        'low': [float(b.low) for b in bars_df],
                                        'volume': [int(b.volume) for b in bars_df]
                                    })
                                    self.regime_detector.update_market_data(symbol, df)
                            except Exception as e:
                                logger.warning(f"Error updating regime data for {symbol}: {e}")
                    
                    # Detect regime
                    current_regime = self.regime_detector.detect_regime('SPY')
                
                # Get latest bars for each symbol and generate signals
                all_signals = []
                
                for symbol in self.universe:
                    try:
                        # Get latest bar
                        bar = self.client.get_latest_bar(symbol)
                        
                        if bar is None:
                            continue
                        
                        # Run through each strategy (check regime gating)
                        for strategy in self.strategies:
                            # Check if strategy should be enabled based on regime
                            if not self.regime_detector.should_enable_strategy(strategy.name):
                                logger.debug(
                                    f"Strategy {strategy.name} disabled by regime detector"
                                )
                                continue
                            
                            if strategy.is_enabled():
                                signals = strategy.on_bar(bar)
                                all_signals.extend(signals)
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Process any signals
                if all_signals:
                    self._process_signals(all_signals)
                
                # End of day processing
                if current_time.hour >= 16 and current_time.minute >= 0:
                    if last_eod_run is None or last_eod_run.date() < current_time.date():
                        logger.info("Running end-of-day processing")
                        self._end_of_day_processing()
                        last_eod_run = current_time
                
                # Log periodic summary
                if current_time.minute % 15 == 0:
                    self.portfolio.log_summary()
                    self.risk_manager.log_risk_summary()
                    
                    # Update Prometheus metrics
                    self.prometheus.update_portfolio_metrics(
                        equity=self.portfolio.equity(),
                        cash=self.portfolio.cash,
                        positions_value=self.portfolio.positions_value(),
                        daily_pnl=self.portfolio.daily_pnl,
                        daily_pnl_pct=self.portfolio.daily_pnl_pct
                    )
                    
                    exposure = self.portfolio.calculate_exposure()
                    self.prometheus.update_exposure_metrics(
                        gross_exposure_pct=exposure['gross_pct'],
                        net_exposure_pct=exposure['net_pct'],
                        long_exposure=exposure['long'],
                        short_exposure=exposure['short']
                    )
                    
                    num_positions = len(self.portfolio.positions)
                    num_long = sum(1 for p in self.portfolio.positions.values() if p.side == 'long')
                    num_short = num_positions - num_long
                    self.prometheus.update_position_metrics(num_positions, num_long, num_short)
                    
                    # Check for alert conditions
                    # Daily P&L threshold alert
                    pnl_threshold = self.config['alerts'].get('alert_on_daily_pnl_threshold', 5.0)
                    if abs(self.portfolio.daily_pnl_pct) >= pnl_threshold:
                        self.notifier.send_daily_pnl_alert(
                            self.portfolio.daily_pnl_pct,
                            self.portfolio.daily_pnl
                        )
                    
                    # Kill-switch status
                    kill_switch_active = self.risk_manager.is_kill_switch_active()
                    self.prometheus.update_risk_metrics(
                        kill_switch_active=kill_switch_active,
                        daily_drawdown_pct=self.portfolio.daily_drawdown_pct,
                        max_drawdown_pct=0.0  # Would need to track this
                    )
                    
                    if kill_switch_active:
                        self.notifier.send_kill_switch_alert(
                            reason=f"Daily drawdown: {self.portfolio.daily_drawdown_pct:.2f}%",
                            drawdown_pct=self.portfolio.daily_drawdown_pct
                        )
                    
                    # Log regime status
                    regime_summary = self.regime_detector.get_regime_summary()
                    if regime_summary['enabled']:
                        logger.info(
                            f"Regime Status | "
                            f"Current: {regime_summary['current_regime']} | "
                            f"Duration: {regime_summary['duration_minutes']:.1f}m | "
                            f"Transitions: {regime_summary['num_transitions']}"
                        )
                
                # Sleep until next poll
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(60)  # Wait before retrying
    
    def _end_of_day_processing(self) -> None:
        """End of day processing."""
        logger.info("=" * 80)
        logger.info("End of Day Processing")
        logger.info("=" * 80)
        
        # Run strategy EOD processing
        for strategy in self.strategies:
            try:
                strategy.on_end_of_day()
            except Exception as e:
                logger.error(f"Error in EOD processing for {strategy.name}: {e}")
        
        # Log final summary
        self.portfolio.log_summary()
        self.risk_manager.log_risk_summary()
        
        # Reset daily stats
        self.portfolio.reset_daily_stats()
        self.risk_manager.reset_daily_limits()
        
        logger.info("End of day processing complete")
    
    def start(self) -> None:
        """Start the trading bot."""
        logger.info("=" * 80)
        logger.info("STARTING TRADING BOT")
        logger.info("=" * 80)
        
        # Warm up strategies
        self._warmup_strategies()
        
        # Log initial state
        self.portfolio.log_summary()
        self.risk_manager.log_risk_summary()
        
        # Start trading
        self.is_running = True
        self._run_trading_loop()
    
    def stop(self) -> None:
        """Stop the trading bot gracefully."""
        logger.info("=" * 80)
        logger.info("STOPPING TRADING BOT")
        logger.info("=" * 80)
        
        self.is_running = False
        
        # Cancel all pending orders
        logger.info("Cancelling all open orders...")
        self.execution_engine.cancel_all_orders("Shutdown")
        
        # Log final state
        self.portfolio.log_summary()
        
        logger.info("Trading bot stopped successfully")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Alpaca Trading Bot")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Run in paper trading mode (default if not live)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Run in LIVE trading mode (requires confirmation)'
    )
    
    args = parser.parse_args()
    
    # Safety check for live trading
    if args.live:
        print("\n" + "!" * 80)
        print("WARNING: You are about to start LIVE TRADING with REAL MONEY")
        print("!" * 80)
        response = input("Type 'YES' to confirm live trading: ")
        if response != "YES":
            print("Live trading cancelled")
            sys.exit(0)
    
    # Create and start bot
    try:
        bot = TradingBot(args.config)
        bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
