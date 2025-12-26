"""
Comprehensive KPI Dashboard for post-trade analytics and performance monitoring.
Tracks hit ratio, win/loss, turnover, slippage, participation rates, and regime-wise performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from collections import defaultdict


class KPIDashboard:
    """
    Comprehensive KPI tracking and reporting for trading performance.
    Includes regime-wise performance breakdown and execution quality metrics.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize KPI dashboard.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Trade tracking
        self.trades: List[Dict] = []
        self.max_trades_history = 10000  # Keep last 10k trades
        
        # Performance metrics
        self.daily_pnl: Dict[str, float] = {}  # date -> pnl
        self.cumulative_pnl = 0.0
        
        # Execution quality metrics
        self.slippage_records: List[Dict] = []
        self.fill_quality_records: List[Dict] = []
        
        # Regime-wise tracking
        self.regime_performance: Dict[str, Dict] = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'total_return': 0.0
        })
        
        logger.info("KPIDashboard initialized")
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        strategy: str,
        regime: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record a completed trade.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            qty: Quantity
            entry_price: Entry price
            exit_price: Exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            pnl: Profit/loss
            strategy: Strategy name
            regime: Market regime at entry
            metadata: Additional metadata
        """
        # Calculate return
        if side == 'buy':
            ret = (exit_price - entry_price) / entry_price
        else:  # sell/short
            ret = (entry_price - exit_price) / entry_price
        
        # Calculate holding period
        holding_minutes = (exit_time - entry_time).total_seconds() / 60
        
        trade_record = {
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'holding_minutes': holding_minutes,
            'pnl': pnl,
            'return': ret,
            'strategy': strategy,
            'regime': regime,
            'is_win': pnl > 0,
            'date': entry_time.date(),
            'metadata': metadata or {}
        }
        
        self.trades.append(trade_record)
        
        # Keep only recent trades
        if len(self.trades) > self.max_trades_history:
            self.trades.pop(0)
        
        # Update cumulative P&L
        self.cumulative_pnl += pnl
        
        # Update daily P&L
        date_str = entry_time.strftime('%Y-%m-%d')
        if date_str not in self.daily_pnl:
            self.daily_pnl[date_str] = 0.0
        self.daily_pnl[date_str] += pnl
        
        # Update regime-wise performance
        if regime:
            regime_stats = self.regime_performance[regime]
            regime_stats['trades'] += 1
            regime_stats['total_pnl'] += pnl
            regime_stats['total_return'] += ret
            
            if pnl > 0:
                regime_stats['wins'] += 1
            else:
                regime_stats['losses'] += 1
        
        logger.debug(
            f"Trade recorded: {symbol} {side} | "
            f"P&L: ${pnl:.2f}, Return: {ret:.2%}, "
            f"Holding: {holding_minutes:.1f}m"
        )
    
    def record_slippage(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
        qty: int,
        timestamp: datetime,
        spread_bps: Optional[float] = None
    ) -> None:
        """
        Record slippage on a fill.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            expected_price: Expected/theoretical price
            actual_price: Actual fill price
            qty: Quantity filled
            timestamp: Fill timestamp
            spread_bps: Spread at time of execution
        """
        # Calculate slippage
        if side == 'buy':
            slippage_dollars = (actual_price - expected_price) * qty
        else:  # sell
            slippage_dollars = (expected_price - actual_price) * qty
        
        slippage_bps = (abs(actual_price - expected_price) / expected_price) * 10000
        
        record = {
            'symbol': symbol,
            'side': side,
            'expected_price': expected_price,
            'actual_price': actual_price,
            'qty': qty,
            'slippage_dollars': slippage_dollars,
            'slippage_bps': slippage_bps,
            'spread_bps': spread_bps,
            'timestamp': timestamp
        }
        
        self.slippage_records.append(record)
        
        # Keep only recent records
        if len(self.slippage_records) > 1000:
            self.slippage_records.pop(0)
    
    def record_fill_quality(
        self,
        symbol: str,
        order_type: str,
        requested_qty: int,
        filled_qty: int,
        fill_time_seconds: float,
        timestamp: datetime
    ) -> None:
        """
        Record fill quality metrics.
        
        Args:
            symbol: Stock symbol
            order_type: 'market' or 'limit'
            requested_qty: Requested quantity
            filled_qty: Actual filled quantity
            fill_time_seconds: Time to fill in seconds
            timestamp: Fill timestamp
        """
        fill_rate = filled_qty / requested_qty if requested_qty > 0 else 0
        
        record = {
            'symbol': symbol,
            'order_type': order_type,
            'requested_qty': requested_qty,
            'filled_qty': filled_qty,
            'fill_rate': fill_rate,
            'fill_time_seconds': fill_time_seconds,
            'timestamp': timestamp
        }
        
        self.fill_quality_records.append(record)
        
        # Keep only recent records
        if len(self.fill_quality_records) > 1000:
            self.fill_quality_records.pop(0)
    
    def get_hit_ratio(
        self,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> Tuple[float, int, int]:
        """
        Calculate hit ratio (win rate).
        
        Args:
            strategy: Optional strategy filter
            regime: Optional regime filter
            lookback_days: Optional lookback period
            
        Returns:
            Tuple of (hit_ratio, wins, total_trades)
        """
        trades = self._filter_trades(strategy, regime, lookback_days)
        
        if not trades:
            return 0.0, 0, 0
        
        wins = sum(1 for t in trades if t['is_win'])
        total = len(trades)
        
        return wins / total, wins, total
    
    def get_avg_win_loss(
        self,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate average win, average loss, and win/loss ratio.
        
        Args:
            strategy: Optional strategy filter
            regime: Optional regime filter
            lookback_days: Optional lookback period
            
        Returns:
            Tuple of (avg_win, avg_loss, win_loss_ratio)
        """
        trades = self._filter_trades(strategy, regime, lookback_days)
        
        if not trades:
            return 0.0, 0.0, 0.0
        
        wins = [t['pnl'] for t in trades if t['is_win']]
        losses = [abs(t['pnl']) for t in trades if not t['is_win']]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        return avg_win, avg_loss, win_loss_ratio
    
    def get_turnover(
        self,
        lookback_days: int = 30
    ) -> Tuple[float, int]:
        """
        Calculate portfolio turnover.
        
        Args:
            lookback_days: Lookback period in days
            
        Returns:
            Tuple of (daily_turnover, total_trades)
        """
        trades = self._filter_trades(lookback_days=lookback_days)
        
        if not trades:
            return 0.0, 0
        
        total_trades = len(trades)
        daily_turnover = total_trades / lookback_days
        
        return daily_turnover, total_trades
    
    def get_slippage_stats(
        self,
        lookback_days: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get slippage statistics.
        
        Args:
            lookback_days: Optional lookback period
            
        Returns:
            Dictionary with slippage statistics
        """
        records = self._filter_slippage_records(lookback_days)
        
        if not records:
            return {
                'avg_slippage_bps': 0.0,
                'median_slippage_bps': 0.0,
                'total_slippage_dollars': 0.0,
                'samples': 0
            }
        
        slippage_bps = [r['slippage_bps'] for r in records]
        slippage_dollars = [r['slippage_dollars'] for r in records]
        
        return {
            'avg_slippage_bps': np.mean(slippage_bps),
            'median_slippage_bps': np.median(slippage_bps),
            'p95_slippage_bps': np.percentile(slippage_bps, 95),
            'total_slippage_dollars': sum(slippage_dollars),
            'samples': len(records)
        }
    
    def get_regime_wise_performance(self) -> Dict[str, Dict]:
        """
        Get performance breakdown by regime.
        
        Returns:
            Dictionary of regime -> performance metrics
        """
        results = {}
        
        for regime, stats in self.regime_performance.items():
            if stats['trades'] == 0:
                continue
            
            win_rate = stats['wins'] / stats['trades']
            avg_return = stats['total_return'] / stats['trades']
            avg_pnl = stats['total_pnl'] / stats['trades']
            
            results[regime] = {
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': win_rate,
                'total_pnl': stats['total_pnl'],
                'avg_pnl': avg_pnl,
                'avg_return': avg_return
            }
        
        return results
    
    def get_comprehensive_kpis(
        self,
        lookback_days: int = 30
    ) -> Dict:
        """
        Get comprehensive KPI summary.
        
        Args:
            lookback_days: Lookback period for metrics
            
        Returns:
            Dictionary with all KPIs
        """
        # Hit ratio
        hit_ratio, wins, total_trades = self.get_hit_ratio(lookback_days=lookback_days)
        
        # Win/loss stats
        avg_win, avg_loss, wl_ratio = self.get_avg_win_loss(lookback_days=lookback_days)
        
        # Turnover
        daily_turnover, _ = self.get_turnover(lookback_days)
        
        # Slippage
        slippage_stats = self.get_slippage_stats(lookback_days)
        
        # Drawdown (from daily P&L)
        drawdown_pct, max_dd_pct = self._calculate_drawdown(lookback_days)
        
        # Sharpe ratio (approximate)
        sharpe = self._calculate_sharpe_ratio(lookback_days)
        
        # Regime-wise performance
        regime_performance = self.get_regime_wise_performance()
        
        return {
            'lookback_days': lookback_days,
            'total_trades': total_trades,
            'win_rate': hit_ratio,
            'wins': wins,
            'losses': total_trades - wins,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': wl_ratio,
            'daily_turnover': daily_turnover,
            'current_drawdown_pct': drawdown_pct,
            'max_drawdown_pct': max_dd_pct,
            'sharpe_ratio': sharpe,
            'slippage': slippage_stats,
            'regime_performance': regime_performance,
            'cumulative_pnl': self.cumulative_pnl
        }
    
    def _filter_trades(
        self,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> List[Dict]:
        """Filter trades by criteria."""
        trades = self.trades.copy()
        
        if strategy:
            trades = [t for t in trades if t['strategy'] == strategy]
        
        if regime:
            trades = [t for t in trades if t.get('regime') == regime]
        
        if lookback_days:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            trades = [t for t in trades if t['entry_time'] >= cutoff]
        
        return trades
    
    def _filter_slippage_records(
        self,
        lookback_days: Optional[int] = None
    ) -> List[Dict]:
        """Filter slippage records by lookback."""
        records = self.slippage_records.copy()
        
        if lookback_days:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            records = [r for r in records if r['timestamp'] >= cutoff]
        
        return records
    
    def _calculate_drawdown(self, lookback_days: int) -> Tuple[float, float]:
        """Calculate current and maximum drawdown."""
        # Get daily P&L for lookback period
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        relevant_pnl = {
            date: pnl
            for date, pnl in self.daily_pnl.items()
            if date >= cutoff
        }
        
        if not relevant_pnl:
            return 0.0, 0.0
        
        # Calculate cumulative P&L
        dates = sorted(relevant_pnl.keys())
        cumulative = []
        cum_pnl = 0
        
        for date in dates:
            cum_pnl += relevant_pnl[date]
            cumulative.append(cum_pnl)
        
        if not cumulative:
            return 0.0, 0.0
        
        # Calculate drawdown
        peak = cumulative[0]
        max_dd = 0
        current_dd = 0
        
        for val in cumulative:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
            current_dd = peak - cumulative[-1]
        
        # Convert to percentage (assume $100k starting capital)
        starting_capital = 100000
        current_dd_pct = (current_dd / starting_capital) * 100
        max_dd_pct = (max_dd / starting_capital) * 100
        
        return current_dd_pct, max_dd_pct
    
    def _calculate_sharpe_ratio(self, lookback_days: int) -> float:
        """Calculate Sharpe ratio from daily returns."""
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        relevant_pnl = [
            pnl for date, pnl in self.daily_pnl.items()
            if date >= cutoff
        ]
        
        if len(relevant_pnl) < 2:
            return 0.0
        
        # Simple Sharpe: mean / std * sqrt(252)
        mean_daily = np.mean(relevant_pnl)
        std_daily = np.std(relevant_pnl)
        
        if std_daily == 0:
            return 0.0
        
        sharpe = (mean_daily / std_daily) * np.sqrt(252)
        
        return sharpe
