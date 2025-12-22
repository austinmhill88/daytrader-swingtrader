"""
Enhanced backtesting with walk-forward validation (Phase 1).
Implements realistic cost modeling and performance metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.config import load_config
from src.logging_utils import setup_logging
from src.strategies.intraday_mean_reversion import IntradayMeanReversion
from src.strategies.swing_trend_following import SwingTrendFollowing
from src.models import Bar


@dataclass
class BacktestPeriod:
    """Represents a train/test period for walk-forward validation."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    name: str


@dataclass
class PerformanceMetrics:
    """Performance metrics for a backtest period."""
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return_pct: float
    period_name: str


class WalkForwardBacktester:
    """
    Walk-forward backtester with realistic cost modeling.
    Phase 1 implementation with hooks for Phase 2 ML integration.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize walk-forward backtester.
        
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
        
        # Backtesting configuration
        bt_config = self.config.get('backtesting', {})
        self.enable_walk_forward = bt_config.get('enable_walk_forward', True)
        self.train_window_days = bt_config.get('train_window_days', 252)
        self.test_window_days = bt_config.get('test_window_days', 63)
        self.step_days = bt_config.get('step_days', 21)
        self.min_trades_per_period = bt_config.get('min_trades_per_period', 10)
        
        # Cost modeling
        cost_config = bt_config.get('transaction_costs', {})
        self.commission_per_share = cost_config.get('commission_per_share', 0.0)
        self.slippage_bps = cost_config.get('slippage_bps', 5)
        
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
        
        # Results storage
        self.period_results: List[PerformanceMetrics] = []
        
        logger.info(
            f"WalkForwardBacktester initialized | "
            f"Train: {self.train_window_days}d, Test: {self.test_window_days}d, "
            f"Step: {self.step_days}d"
        )
    
    def generate_walk_forward_periods(
        self,
        data_start: datetime,
        data_end: datetime
    ) -> List[BacktestPeriod]:
        """
        Generate walk-forward train/test periods.
        
        Args:
            data_start: Start date of available data
            data_end: End date of available data
            
        Returns:
            List of BacktestPeriod objects
        """
        periods = []
        current_start = data_start
        period_num = 1
        
        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_window_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_window_days)
            
            # Check if we have enough data
            if test_end > data_end:
                break
            
            period = BacktestPeriod(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                name=f"Period_{period_num}"
            )
            periods.append(period)
            
            # Step forward
            current_start += timedelta(days=self.step_days)
            period_num += 1
            
            # Safety check
            if period_num > 100:
                logger.warning("Generated 100+ periods, stopping")
                break
        
        logger.info(f"Generated {len(periods)} walk-forward periods")
        return periods
    
    def run_single_period(
        self,
        period: BacktestPeriod,
        data_by_symbol: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0
    ) -> Tuple[PerformanceMetrics, pd.DataFrame]:
        """
        Run backtest for a single period.
        
        Args:
            period: BacktestPeriod to test
            data_by_symbol: Dictionary of symbol -> DataFrame
            initial_capital: Starting capital
            
        Returns:
            Tuple of (PerformanceMetrics, equity_curve_df)
        """
        logger.info(f"Running backtest for {period.name}")
        
        # Warm up strategies with training data
        for strategy in self.strategies:
            train_data = {}
            for symbol, df in data_by_symbol.items():
                mask = (df.index >= period.train_start) & (df.index <= period.train_end)
                train_data[symbol] = df[mask].copy()
            
            if train_data:
                strategy.warmup(train_data)
        
        # Run test period
        capital = initial_capital
        positions = {}
        trades = []
        equity_curve = []
        
        # Get test data
        test_data = {}
        for symbol, df in data_by_symbol.items():
            mask = (df.index >= period.test_start) & (df.index <= period.test_end)
            test_data[symbol] = df[mask].copy()
        
        # Simulate trading
        for symbol, df in test_data.items():
            if len(df) == 0:
                continue
            
            for idx, row in df.iterrows():
                # Create bar
                bar = Bar(
                    symbol=symbol,
                    ts=idx,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    vwap=row.get('vwap')
                )
                
                # Generate signals
                for strategy in self.strategies:
                    try:
                        signals = strategy.on_bar(bar)
                        
                        for signal in signals:
                            if signal.strength == 0:
                                # Exit signal
                                if symbol in positions:
                                    exit_price = self._apply_slippage(bar.close, "sell")
                                    pnl = (exit_price - positions[symbol]['entry_price']) * positions[symbol]['qty']
                                    pnl -= self._calculate_costs(positions[symbol]['qty'], exit_price)
                                    
                                    capital += pnl
                                    trades.append({
                                        'symbol': symbol,
                                        'entry_time': positions[symbol]['entry_time'],
                                        'exit_time': idx,
                                        'pnl': pnl,
                                        'return_pct': (pnl / (positions[symbol]['entry_price'] * positions[symbol]['qty'])) * 100
                                    })
                                    del positions[symbol]
                            
                            elif signal.strength > 0 and symbol not in positions:
                                # Entry signal
                                qty = max(1, int(capital * 0.1 / bar.close))  # Simple sizing
                                entry_price = self._apply_slippage(bar.close, "buy")
                                cost = qty * entry_price + self._calculate_costs(qty, entry_price)
                                
                                if cost <= capital:
                                    positions[symbol] = {
                                        'qty': qty,
                                        'entry_price': entry_price,
                                        'entry_time': idx
                                    }
                                    capital -= cost
                    
                    except Exception as e:
                        logger.error(f"Error processing signal: {e}")
                
                # Calculate current equity
                positions_value = sum(
                    pos['qty'] * bar.close for pos in positions.values()
                )
                total_equity = capital + positions_value
                equity_curve.append({
                    'timestamp': idx,
                    'equity': total_equity,
                    'cash': capital,
                    'positions_value': positions_value
                })
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        metrics = self._calculate_metrics(equity_df, trades, initial_capital, period.name)
        
        return metrics, equity_df
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        slippage = price * (self.slippage_bps / 10000)
        if side == "buy":
            return price + slippage
        else:
            return price - slippage
    
    def _calculate_costs(self, qty: int, price: float) -> float:
        """Calculate transaction costs."""
        commission = qty * self.commission_per_share
        return commission
    
    def _calculate_metrics(
        self,
        equity_df: pd.DataFrame,
        trades: List[Dict],
        initial_capital: float,
        period_name: str
    ) -> PerformanceMetrics:
        """Calculate performance metrics."""
        if len(equity_df) == 0 or len(trades) == 0:
            return PerformanceMetrics(
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                num_trades=0,
                avg_trade_return_pct=0.0,
                period_name=period_name
            )
        
        # Returns
        final_equity = equity_df['equity'].iloc[-1]
        total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
        
        # Sharpe & Sortino
        equity_df['returns'] = equity_df['equity'].pct_change()
        mean_return = equity_df['returns'].mean()
        std_return = equity_df['returns'].std()
        
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        downside_returns = equity_df['returns'][equity_df['returns'] < 0]
        downside_std = downside_returns.std()
        sortino = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown_pct = abs(equity_df['drawdown'].min() * 100)
        
        # Trade metrics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_trade_return = np.mean([t['return_pct'] for t in trades]) if trades else 0
        
        return PerformanceMetrics(
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            num_trades=len(trades),
            avg_trade_return_pct=avg_trade_return,
            period_name=period_name
        )
    
    def run_walk_forward(
        self,
        data_by_symbol: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """
        Run complete walk-forward validation.
        
        Args:
            data_by_symbol: Dictionary of symbol -> DataFrame with OHLCV data
            initial_capital: Starting capital for each period
            
        Returns:
            DataFrame with aggregated results
        """
        if not self.enable_walk_forward:
            logger.warning("Walk-forward validation is disabled")
            return pd.DataFrame()
        
        # Determine data range
        all_dates = []
        for df in data_by_symbol.values():
            if len(df) > 0:
                all_dates.extend(df.index.tolist())
        
        if not all_dates:
            logger.error("No data available for backtesting")
            return pd.DataFrame()
        
        data_start = min(all_dates)
        data_end = max(all_dates)
        
        logger.info(f"Data range: {data_start} to {data_end}")
        
        # Generate periods
        periods = self.generate_walk_forward_periods(data_start, data_end)
        
        if not periods:
            logger.error("No valid walk-forward periods generated")
            return pd.DataFrame()
        
        # Run each period
        results = []
        for period in periods:
            try:
                metrics, equity_curve = self.run_single_period(
                    period, data_by_symbol, initial_capital
                )
                
                if metrics.num_trades >= self.min_trades_per_period:
                    results.append(metrics)
                    logger.info(
                        f"{period.name} | Return: {metrics.total_return_pct:.2f}% | "
                        f"Sharpe: {metrics.sharpe_ratio:.2f} | Trades: {metrics.num_trades}"
                    )
                else:
                    logger.warning(
                        f"{period.name} had only {metrics.num_trades} trades "
                        f"(min: {self.min_trades_per_period}), skipping"
                    )
            
            except Exception as e:
                logger.error(f"Error in {period.name}: {e}", exc_info=True)
        
        # Aggregate results
        if results:
            results_df = pd.DataFrame([vars(m) for m in results])
            
            logger.info("=" * 80)
            logger.info("WALK-FORWARD VALIDATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"Periods tested: {len(results)}")
            logger.info(f"Avg Return: {results_df['total_return_pct'].mean():.2f}%")
            logger.info(f"Avg Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
            logger.info(f"Avg Max DD: {results_df['max_drawdown_pct'].mean():.2f}%")
            logger.info(f"Avg Win Rate: {results_df['win_rate'].mean():.2f}%")
            logger.info(f"Total Trades: {results_df['num_trades'].sum()}")
            logger.info("=" * 80)
            
            return results_df
        else:
            logger.error("No valid results from walk-forward validation")
            return pd.DataFrame()
