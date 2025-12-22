"""
Simple backtesting framework for strategy validation.
"""
import argparse
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List

from src.config import load_config
from src.logging_utils import setup_logging
from src.strategies.intraday_mean_reversion import IntradayMeanReversion
from src.strategies.swing_trend_following import SwingTrendFollowing
from src.models import Bar


class Backtester:
    """
    Event-driven backtester for strategy validation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize backtester.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        
        # Setup logging
        setup_logging(
            logs_dir="./logs/backtest",
            level="INFO",
            format_type="text"
        )
        
        logger.info("Backtester initialized")
        
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
        
        # Backtest results
        self.results = {
            'trades': [],
            'equity_curve': [],
            'signals': []
        }
        
        self.initial_capital = 100000.0
        self.current_capital = self.initial_capital
        self.positions = {}
        
        # Cost and slippage modeling
        backtest_config = self.config.get('backtesting', {})
        transaction_costs = backtest_config.get('transaction_costs', {})
        
        self.commission_per_share = transaction_costs.get('commission_per_share', 0.0)
        self.slippage_bps = transaction_costs.get('slippage_bps', 5)
        self.spread_cost_model = transaction_costs.get('spread_cost_model', 'dynamic')
        self.short_borrow_rate_annual = transaction_costs.get('short_borrow_rate_annual', 0.003)  # 0.3% annual
        
        # Realistic fill simulation
        fill_simulation = backtest_config.get('fill_simulation', {})
        self.enable_partial_fills = fill_simulation.get('enable_partial_fills', True)
        self.partial_fill_probability = fill_simulation.get('partial_fill_probability', 0.2)
        self.min_partial_fill_pct = fill_simulation.get('min_partial_fill_pct', 0.5)
        self.enable_queue_priority = fill_simulation.get('enable_queue_priority', True)
        self.limit_order_fill_probability = fill_simulation.get('limit_order_fill_probability', 0.85)
        
        # Time-of-day adjustments
        self.enable_tod_adjustments = fill_simulation.get('enable_tod_adjustments', True)
        self.market_open_volatility_multiplier = fill_simulation.get('market_open_volatility_multiplier', 2.0)
        self.market_close_volatility_multiplier = fill_simulation.get('market_close_volatility_multiplier', 1.5)
    
    def load_data(self, symbol: str, csv_path: str) -> pd.DataFrame:
        """
        Load historical data from CSV.
        
        Args:
            symbol: Stock symbol
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            df = pd.read_csv(csv_path)
            df['symbol'] = symbol
            df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df['ts'])
            df = df.set_index('ts')
            
            logger.info(f"Loaded {len(df)} bars for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {csv_path}: {e}")
            return None
    
    def _get_tod_volatility_multiplier(self, timestamp: datetime) -> float:
        """
        Get time-of-day volatility multiplier for realistic spread/slippage simulation.
        
        Args:
            timestamp: Bar timestamp
            
        Returns:
            Volatility multiplier (1.0 = normal, >1.0 = higher volatility)
        """
        if not self.enable_tod_adjustments:
            return 1.0
        
        # Market open/close hours have higher volatility
        hour = timestamp.hour
        minute = timestamp.minute
        
        # First 30 minutes after open (9:30-10:00 ET)
        if hour == 9 and minute >= 30:
            return self.market_open_volatility_multiplier
        elif hour == 10 and minute < 30:
            # Gradually reduce multiplier
            return 1.0 + (self.market_open_volatility_multiplier - 1.0) * (30 - minute) / 30
        
        # Last 30 minutes before close (15:30-16:00 ET)
        elif hour == 15 and minute >= 30:
            # Gradually increase multiplier
            return 1.0 + (self.market_close_volatility_multiplier - 1.0) * (minute - 30) / 30
        elif hour >= 16:
            return self.market_close_volatility_multiplier
        
        return 1.0
    
    def _simulate_limit_order_fill(
        self,
        bar: Bar,
        limit_price: float,
        is_buy: bool
    ) -> tuple[bool, Optional[float]]:
        """
        Simulate whether a limit order would fill and at what price.
        
        Args:
            bar: Market bar data
            limit_price: Limit price
            is_buy: True for buy order, False for sell order
            
        Returns:
            Tuple of (filled, fill_price)
        """
        import random
        
        # Check if limit price is within bar's range
        if is_buy:
            if limit_price < bar.low:
                return False, None  # Limit too low, no fill
            elif limit_price >= bar.high:
                # Would definitely fill, but at what price?
                fill_price = min(limit_price, bar.close)
                return True, fill_price
        else:
            if limit_price > bar.high:
                return False, None  # Limit too high, no fill
            elif limit_price <= bar.low:
                # Would definitely fill
                fill_price = max(limit_price, bar.close)
                return True, fill_price
        
        # Price is within range - use queue priority simulation
        if self.enable_queue_priority:
            # Simulate queue position - limit orders have some probability of filling
            if random.random() < self.limit_order_fill_probability:
                # Filled - use limit price or better
                fill_price = limit_price
                return True, fill_price
        
        return False, None
    
    def _simulate_partial_fill(self, qty: int) -> int:
        """
        Simulate partial fill for an order.
        
        Args:
            qty: Requested quantity
            
        Returns:
            Actual filled quantity (may be partial)
        """
        import random
        
        if not self.enable_partial_fills:
            return qty
        
        # Randomly decide if this order gets partial fill
        if random.random() < self.partial_fill_probability:
            # Partial fill - return percentage of requested quantity
            fill_pct = self.min_partial_fill_pct + (1.0 - self.min_partial_fill_pct) * random.random()
            partial_qty = max(1, int(qty * fill_pct))
            logger.debug(f"Partial fill: {partial_qty}/{qty} ({fill_pct*100:.1f}%)")
            return partial_qty
        
        return qty
    
    def run(self, data_by_symbol: Dict[str, pd.DataFrame]) -> None:
        """
        Run backtest on historical data.
        
        Args:
            data_by_symbol: Dictionary mapping symbol to DataFrame
        """
        logger.info(f"Starting backtest with {len(data_by_symbol)} symbols")
        
        # Warm up strategies
        for strategy in self.strategies:
            warmup_data = {}
            for symbol, df in data_by_symbol.items():
                # Use first 1000 bars for warmup
                warmup_data[symbol] = df.iloc[:1000].copy()
            
            strategy.warmup(warmup_data)
        
        # Run event-driven simulation
        # For simplicity, process each symbol sequentially
        # In production, would process in timestamp order across all symbols
        
        for symbol, df in data_by_symbol.items():
            logger.info(f"Processing {symbol}...")
            
            # Skip warmup period
            for i in range(1000, len(df)):
                row = df.iloc[i]
                
                # Create bar object
                bar = Bar(
                    symbol=symbol,
                    ts=row.name,
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume']),
                    vwap=float(row['vwap']) if 'vwap' in row and pd.notna(row['vwap']) else None
                )
                
                # Generate signals from each strategy
                for strategy in self.strategies:
                    try:
                        signals = strategy.on_bar(bar)
                        
                        for signal in signals:
                            self.results['signals'].append({
                                'timestamp': bar.ts,
                                'symbol': signal.symbol,
                                'strategy': signal.strategy_name,
                                'strength': signal.strength,
                                'confidence': signal.confidence,
                                'reason': signal.reason
                            })
                            
                            # Simple fill simulation (would be more sophisticated in production)
                            if signal.strength != 0:
                                self._simulate_trade(bar, signal)
                    
                    except Exception as e:
                        logger.error(f"Error processing bar for {symbol}: {e}")
        
        # Calculate metrics
        self._calculate_metrics()
    
    def _simulate_trade(self, bar: Bar, signal) -> None:
        """
        Simulate trade execution with realistic slippage and costs.
        
        Args:
            bar: Current bar
            signal: Trading signal
        """
        # Calculate trade size based on signal strength and confidence
        position_size_pct = self.config.get('backtesting', {}).get('position_size_pct', 0.1)
        trade_value_target = self.current_capital * position_size_pct * abs(signal.strength) * signal.confidence
        trade_size = int(trade_value_target / bar.close)
        
        if trade_size <= 0:
            return
        
        # Calculate slippage
        slippage = self._calculate_slippage(bar, signal.strength > 0, trade_size)
        
        # Calculate spread cost
        spread_cost = self._calculate_spread_cost(bar, trade_size)
        
        # Calculate commission
        commission = self.commission_per_share * trade_size
        
        # Determine fill price
        base_price = bar.close
        
        if signal.strength > 0:
            # Buy - pay slippage and spread
            fill_price = base_price * (1 + slippage + spread_cost)
        else:
            # Sell - receive less due to slippage and spread
            fill_price = base_price * (1 - slippage - spread_cost)
        
        # Execute trade
        if signal.strength > 0:
            # Buy
            total_cost = (fill_price * trade_size) + commission
            
            if total_cost <= self.current_capital:
                self.current_capital -= total_cost
                self.positions[bar.symbol] = self.positions.get(bar.symbol, 0) + trade_size
                
                self.results['trades'].append({
                    'timestamp': bar.ts,
                    'symbol': bar.symbol,
                    'action': 'BUY',
                    'qty': trade_size,
                    'price': bar.close,
                    'fill_price': fill_price,
                    'slippage_bps': slippage * 10000,
                    'spread_cost_bps': spread_cost * 10000,
                    'commission': commission,
                    'total_cost': total_cost,
                    'strategy': signal.strategy_name
                })
        
        elif signal.strength < 0:
            # Sell
            if bar.symbol in self.positions and self.positions[bar.symbol] >= trade_size:
                # Initialize short cost
                short_cost = 0
                
                # Calculate short borrow cost if this is a short position
                if self.positions[bar.symbol] < 0:
                    # Daily short borrow cost
                    short_cost = abs(self.positions[bar.symbol]) * fill_price * (self.short_borrow_rate_annual / 252)
                
                total_proceeds = (fill_price * trade_size) - commission - short_cost
                
                self.current_capital += total_proceeds
                self.positions[bar.symbol] -= trade_size
                
                self.results['trades'].append({
                    'timestamp': bar.ts,
                    'symbol': bar.symbol,
                    'action': 'SELL',
                    'qty': trade_size,
                    'price': bar.close,
                    'fill_price': fill_price,
                    'slippage_bps': slippage * 10000,
                    'spread_cost_bps': spread_cost * 10000,
                    'commission': commission,
                    'short_borrow_cost': short_cost,
                    'total_proceeds': total_proceeds,
                    'strategy': signal.strategy_name
                })
        
        # Record equity (mark positions to market)
        # Note: This is simplified - in production would track last known price per symbol
        total_equity = self.current_capital
        for symbol, qty in self.positions.items():
            if symbol == bar.symbol:
                total_equity += qty * bar.close
            else:
                # Simplified: assume position hasn't changed
                # In production, would maintain price cache per symbol
                total_equity += qty * bar.close
        
        self.results['equity_curve'].append({
            'timestamp': bar.ts,
            'equity': total_equity
        })
    
    def _calculate_slippage(self, bar: Bar, is_buy: bool, qty: int) -> float:
        """
        Calculate slippage as a fraction of price.
        
        Args:
            bar: Current bar
            is_buy: True if buying, False if selling
            qty: Order quantity
            
        Returns:
            Slippage as decimal (e.g., 0.0005 for 5 bps)
        """
        # Base slippage from config
        base_slippage_bps = self.slippage_bps
        
        # Adjust for time of day (higher slippage at open/close)
        hour = bar.ts.hour
        minute = bar.ts.minute
        time_multiplier = 1.0
        
        if hour == 9 and minute < 45:
            # First 15 minutes
            time_multiplier = 2.0
        elif hour == 15 and minute >= 45:
            # Last 15 minutes
            time_multiplier = 1.5
        
        # Adjust for order size (larger orders have more slippage)
        # Assume average volume is in bar.volume
        if bar.volume > 0:
            volume_impact = min((qty / bar.volume) * 100, 2.0)  # Cap at 2x multiplier
            size_multiplier = 1.0 + volume_impact
        else:
            size_multiplier = 1.0
        
        # Adjust for volatility (use high-low range as proxy)
        if bar.high > bar.low:
            daily_range_pct = (bar.high - bar.low) / bar.close
            vol_multiplier = 1.0 + (daily_range_pct * 10)  # Scale up volatility impact
        else:
            vol_multiplier = 1.0
        
        # Combined slippage
        adjusted_slippage_bps = base_slippage_bps * time_multiplier * size_multiplier * vol_multiplier
        
        return adjusted_slippage_bps / 10000
    
    def _calculate_spread_cost(self, bar: Bar, qty: int) -> float:
        """
        Calculate spread cost as a fraction of price.
        
        Args:
            bar: Current bar
            qty: Order quantity
            
        Returns:
            Spread cost as decimal
        """
        if self.spread_cost_model == 'fixed':
            # Fixed spread of 2 bps
            return 0.0002
        
        # Dynamic spread based on high-low range
        if bar.high > bar.low:
            spread = (bar.high - bar.low) / bar.close
            # Half spread for crossing
            return spread / 2
        
        # Fallback to 2 bps
        return 0.0002
    
    
    def _calculate_metrics(self) -> None:
        """Calculate performance metrics."""
        if not self.results['equity_curve']:
            logger.warning("No equity curve data to calculate metrics")
            return
        
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df = equity_df.set_index('timestamp')
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        mean_return = equity_df['returns'].mean()
        std_return = equity_df['returns'].std()
        sharpe = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        
        # Max drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        # Win rate
        trades_df = pd.DataFrame(self.results['trades'])
        if len(trades_df) > 0:
            # Simple win rate calculation (would be more sophisticated in production)
            num_trades = len(trades_df)
        else:
            num_trades = 0
        
        # Log results
        logger.info("=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Equity: ${final_equity:,.2f}")
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Number of Trades: {num_trades}")
        logger.info(f"Number of Signals: {len(self.results['signals'])}")
        logger.info("=" * 80)


def main():
    """Main entry point for backtester."""
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to historical data CSV file or directory'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Symbol to backtest (if single file)'
    )
    
    args = parser.parse_args()
    
    # Create backtester
    backtester = Backtester(args.config)
    
    # Load data (simplified - single symbol for now)
    data_by_symbol = {}
    df = backtester.load_data(args.symbol, args.data)
    
    if df is not None:
        data_by_symbol[args.symbol] = df
        
        # Run backtest
        backtester.run(data_by_symbol)
    else:
        logger.error("Failed to load data")


if __name__ == "__main__":
    main()
