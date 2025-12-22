"""
Portfolio state management and tracking.
"""
from loguru import logger
from typing import Dict, List, Optional
from datetime import datetime
from src.models import Position, PerformanceMetrics, OrderIntent


class PortfolioState:
    """
    Manages portfolio state, positions, and performance tracking.
    """
    
    def __init__(self, alpaca_client):
        """
        Initialize portfolio state.
        
        Args:
            alpaca_client: AlpacaClient instance
        """
        self.client = alpaca_client
        self._equity_start = None
        self._equity_high = None
        self._daily_equity_start = None
        self._positions_cache: Dict[str, Position] = {}
        self._last_update = None
        self._trade_count_today = 0
        self._wins_today = 0
        self._losses_today = 0
        
        logger.info("PortfolioState initialized")
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize starting values."""
        try:
            account = self.client.get_account()
            equity = float(account.equity)
            
            self._equity_start = equity
            self._equity_high = equity
            self._daily_equity_start = equity
            
            logger.info(f"Portfolio initialized | Starting equity: ${equity:,.2f}")
        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")
    
    def update(self) -> None:
        """Update portfolio state with latest data."""
        try:
            # Update positions
            positions = self.client.get_positions()
            self._positions_cache = {pos.symbol: pos for pos in positions}
            
            # Update equity tracking
            account = self.client.get_account()
            equity = float(account.equity)
            
            if self._equity_high is None or equity > self._equity_high:
                self._equity_high = equity
            
            self._last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    def equity(self) -> float:
        """
        Get current equity.
        
        Returns:
            Current equity value
        """
        try:
            account = self.client.get_account()
            equity = float(account.equity)
            
            # Update high water mark
            if self._equity_high is None or equity > self._equity_high:
                self._equity_high = equity
            
            return equity
        except Exception as e:
            logger.error(f"Error getting equity: {e}")
            return 0.0
    
    def cash(self) -> float:
        """
        Get current cash balance.
        
        Returns:
            Cash balance
        """
        try:
            account = self.client.get_account()
            return float(account.cash)
        except Exception as e:
            logger.error(f"Error getting cash: {e}")
            return 0.0
    
    def buying_power(self) -> float:
        """
        Get current buying power.
        
        Returns:
            Buying power
        """
        try:
            account = self.client.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            return 0.0
    
    def positions(self) -> List[Position]:
        """
        Get current positions.
        
        Returns:
            List of Position objects
        """
        self.update()
        return list(self._positions_cache.values())
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position or None if no position
        """
        if symbol in self._positions_cache:
            return self._positions_cache[symbol]
        
        # Try to fetch from API
        pos = self.client.get_position(symbol)
        if pos:
            self._positions_cache[symbol] = pos
        return pos
    
    def has_position(self, symbol: str) -> bool:
        """
        Check if we have a position in symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if position exists
        """
        return self.get_position(symbol) is not None
    
    def position_count(self) -> int:
        """
        Get number of open positions.
        
        Returns:
            Position count
        """
        return len(self._positions_cache)
    
    def daily_pnl(self) -> float:
        """
        Calculate daily P&L.
        
        Returns:
            Daily P&L in dollars
        """
        if self._daily_equity_start is None:
            return 0.0
        
        current_equity = self.equity()
        return current_equity - self._daily_equity_start
    
    def daily_pnl_pct(self) -> float:
        """
        Calculate daily P&L percentage.
        
        Returns:
            Daily P&L as percentage
        """
        if self._daily_equity_start is None or self._daily_equity_start == 0:
            return 0.0
        
        return (self.daily_pnl() / self._daily_equity_start) * 100.0
    
    def total_pnl(self) -> float:
        """
        Calculate total P&L since start.
        
        Returns:
            Total P&L in dollars
        """
        if self._equity_start is None:
            return 0.0
        
        return self.equity() - self._equity_start
    
    def total_pnl_pct(self) -> float:
        """
        Calculate total P&L percentage.
        
        Returns:
            Total P&L as percentage
        """
        if self._equity_start is None or self._equity_start == 0:
            return 0.0
        
        return (self.total_pnl() / self._equity_start) * 100.0
    
    def daily_drawdown_pct(self) -> float:
        """
        Calculate current drawdown from daily high.
        
        Returns:
            Drawdown percentage
        """
        if self._equity_high is None or self._equity_high == 0:
            return 0.0
        
        current_equity = self.equity()
        drawdown = (self._equity_high - current_equity) / self._equity_high * 100.0
        return max(drawdown, 0.0)
    
    def max_drawdown_pct(self) -> float:
        """
        Calculate maximum drawdown from all-time high.
        
        Returns:
            Max drawdown percentage
        """
        return self.daily_drawdown_pct()  # Same calculation for now
    
    def gross_exposure_pct(self) -> float:
        """
        Calculate gross exposure as percentage of equity.
        
        Returns:
            Gross exposure percentage
        """
        total_exposure = sum(abs(pos.market_value) for pos in self._positions_cache.values())
        equity = self.equity()
        
        if equity == 0:
            return 0.0
        
        return (total_exposure / equity) * 100.0
    
    def net_exposure_pct(self) -> float:
        """
        Calculate net exposure as percentage of equity.
        
        Returns:
            Net exposure percentage
        """
        net_exposure = sum(pos.market_value for pos in self._positions_cache.values())
        equity = self.equity()
        
        if equity == 0:
            return 0.0
        
        return (net_exposure / equity) * 100.0
    
    def projected_gross_exposure_pct(self, intent: OrderIntent) -> float:
        """
        Calculate projected gross exposure if order is executed.
        
        Args:
            intent: Order intent to simulate
            
        Returns:
            Projected gross exposure percentage
        """
        # Calculate current exposure
        current_exposure = sum(abs(pos.market_value) for pos in self._positions_cache.values())
        
        # Calculate intent value
        intent_value = abs(intent.qty * (intent.limit_price or 0))
        
        # Check if we're reducing an existing position
        pos = self.get_position(intent.symbol)
        if pos:
            if intent.side.value == "sell" and pos.qty > 0:
                # Closing/reducing long position
                reduction = min(intent_value, abs(pos.market_value))
                projected_exposure = current_exposure - reduction + max(0, intent_value - reduction)
            elif intent.side.value == "buy" and pos.qty < 0:
                # Closing/reducing short position
                reduction = min(intent_value, abs(pos.market_value))
                projected_exposure = current_exposure - reduction + max(0, intent_value - reduction)
            else:
                # Adding to existing position
                projected_exposure = current_exposure + intent_value
        else:
            # New position
            projected_exposure = current_exposure + intent_value
        equity = self.equity()
        
        if equity == 0:
            return 0.0
        
        return (projected_exposure / equity) * 100.0
    
    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.
        
        Returns:
            PerformanceMetrics object
        """
        account = self.client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            equity=equity,
            cash=cash,
            positions_value=equity - cash,
            daily_pnl=self.daily_pnl(),
            daily_pnl_pct=self.daily_pnl_pct(),
            total_pnl=self.total_pnl(),
            total_pnl_pct=self.total_pnl_pct(),
            num_positions=self.position_count(),
            num_trades_today=self._trade_count_today,
            gross_exposure_pct=self.gross_exposure_pct(),
            net_exposure_pct=self.net_exposure_pct(),
            max_drawdown_pct=self.max_drawdown_pct(),
            win_rate=self._calculate_win_rate()
        )
    
    def _calculate_win_rate(self) -> Optional[float]:
        """
        Calculate win rate for today.
        
        Returns:
            Win rate as percentage or None
        """
        total_trades = self._wins_today + self._losses_today
        if total_trades == 0:
            return None
        
        return (self._wins_today / total_trades) * 100.0
    
    def record_trade(self, pnl: float) -> None:
        """
        Record a completed trade.
        
        Args:
            pnl: P&L of the trade
        """
        self._trade_count_today += 1
        if pnl > 0:
            self._wins_today += 1
        elif pnl < 0:
            self._losses_today += 1
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of day)."""
        self._daily_equity_start = self.equity()
        self._trade_count_today = 0
        self._wins_today = 0
        self._losses_today = 0
        logger.info("Daily portfolio stats reset")
    
    def log_summary(self) -> None:
        """Log portfolio summary."""
        metrics = self.get_metrics()
        logger.info(
            f"Portfolio Summary | "
            f"Equity: ${metrics.equity:,.2f} | "
            f"Cash: ${metrics.cash:,.2f} | "
            f"Positions: {metrics.num_positions} | "
            f"Daily P&L: ${metrics.daily_pnl:,.2f} ({metrics.daily_pnl_pct:+.2f}%) | "
            f"Exposure: {metrics.gross_exposure_pct:.1f}% | "
            f"Drawdown: {metrics.max_drawdown_pct:.2f}%"
        )
