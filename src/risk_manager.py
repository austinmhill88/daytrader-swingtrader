"""
Risk management with pre-trade checks and kill-switch functionality.
"""
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime

from src.models import OrderIntent, AlertMessage
from src.portfolio import PortfolioState


class RiskManager:
    """
    Risk management system with pre-trade checks, position limits, and kill-switch.
    """
    
    def __init__(self, config: Dict, portfolio: PortfolioState):
        """
        Initialize risk manager.
        
        Args:
            config: Risk configuration dictionary
            portfolio: PortfolioState instance
        """
        self.config = config
        self.portfolio = portfolio
        
        # Risk limits
        self.daily_max_drawdown_pct = config.get('daily_max_drawdown_pct', 2.0)
        self.per_trade_risk_pct = config.get('per_trade_risk_pct', 0.4)
        self.max_position_size_pct = config.get('max_position_size_pct', 5.0)
        self.max_gross_exposure_pct_intraday = config.get('max_gross_exposure_pct_intraday', 50)
        self.max_gross_exposure_pct_swing = config.get('max_gross_exposure_pct_swing', 100)
        self.short_exposure_cap_pct = config.get('short_exposure_cap_pct', 50)
        self.kill_switch_enabled = config.get('kill_switch', True)
        self.max_positions_total = config.get('max_positions_total', 30)
        self.min_trade_size_usd = config.get('min_trade_size_usd', 100)
        self.max_trade_size_usd = config.get('max_trade_size_usd', 50000)
        self.max_trades_per_day = config.get('max_trades_per_day', 100)
        self.max_losses_per_day = config.get('max_losses_per_day', 3)
        
        # State tracking
        self.kill_switch_triggered = False
        self.consecutive_losses = 0
        self.trades_today = 0
        self.alerts: List[AlertMessage] = []
        
        logger.info(
            f"RiskManager initialized | "
            f"Max DD: {self.daily_max_drawdown_pct}%, "
            f"Per trade risk: {self.per_trade_risk_pct}%, "
            f"Kill switch: {self.kill_switch_enabled}"
        )
    
    def pre_trade_checks(
        self,
        intents: List[OrderIntent],
        strategy_type: str = "intraday"
    ) -> tuple[List[OrderIntent], List[str]]:
        """
        Run pre-trade checks on order intents.
        
        Args:
            intents: List of order intents
            strategy_type: "intraday" or "swing" for different exposure limits
            
        Returns:
            Tuple of (approved_intents, rejection_reasons)
        """
        if self.kill_switch_triggered:
            logger.error("Kill switch is active - all trades blocked")
            return [], ["Kill switch is active"]
        
        approved = []
        rejections = []
        
        for intent in intents:
            passed, reason = self._check_single_intent(intent, strategy_type)
            
            if passed:
                approved.append(intent)
            else:
                rejections.append(f"{intent.symbol}: {reason}")
                logger.warning(f"Trade rejected | {intent.symbol} | {reason}")
        
        return approved, rejections
    
    def _check_single_intent(
        self,
        intent: OrderIntent,
        strategy_type: str
    ) -> tuple[bool, str]:
        """
        Check a single order intent against all risk rules.
        
        Args:
            intent: Order intent to check
            strategy_type: Strategy type for exposure limits
            
        Returns:
            Tuple of (passed, reason_if_failed)
        """
        # 1. Check kill switch
        if self.kill_switch_triggered:
            return False, "Kill switch active"
        
        # 2. Check daily drawdown
        dd = self.portfolio.daily_drawdown_pct()
        if dd >= self.daily_max_drawdown_pct:
            self._trigger_kill_switch("Daily drawdown limit breached")
            return False, f"Daily drawdown {dd:.2f}% >= {self.daily_max_drawdown_pct}%"
        
        # 3. Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            return False, f"Max trades per day ({self.max_trades_per_day}) reached"
        
        # 4. Check consecutive losses
        if self.consecutive_losses >= self.max_losses_per_day:
            logger.warning(f"Pausing trading after {self.consecutive_losses} consecutive losses")
            return False, f"Max consecutive losses ({self.max_losses_per_day}) reached"
        
        # 5. Check position count
        current_positions = self.portfolio.position_count()
        if current_positions >= self.max_positions_total:
            return False, f"Max positions ({self.max_positions_total}) reached"
        
        # 6. Check trade size limits
        trade_value = intent.qty * (intent.limit_price or 0)
        if trade_value < self.min_trade_size_usd:
            return False, f"Trade size ${trade_value:.2f} < min ${self.min_trade_size_usd}"
        
        if trade_value > self.max_trade_size_usd:
            return False, f"Trade size ${trade_value:.2f} > max ${self.max_trade_size_usd}"
        
        # 7. Check position size as % of equity
        equity = self.portfolio.equity()
        if equity > 0:
            position_pct = (trade_value / equity) * 100
            if position_pct > self.max_position_size_pct:
                return False, f"Position size {position_pct:.1f}% > max {self.max_position_size_pct}%"
        
        # 8. Check gross exposure limits
        max_exposure = (self.max_gross_exposure_pct_intraday 
                       if strategy_type == "intraday" 
                       else self.max_gross_exposure_pct_swing)
        
        projected_exposure = self.portfolio.projected_gross_exposure_pct(intent)
        if projected_exposure > max_exposure:
            return False, f"Gross exposure would be {projected_exposure:.1f}% > max {max_exposure}%"
        
        # 9. Check short exposure cap (if selling/shorting)
        if intent.side.value == "sell":
            # Calculate current short exposure
            short_exposure = 0.0
            for pos in self.portfolio.positions():
                if pos.side == "short":
                    short_exposure += abs(pos.market_value)
            
            # Add this trade
            short_exposure += trade_value
            
            if equity > 0:
                short_exposure_pct = (short_exposure / equity) * 100
                if short_exposure_pct > self.short_exposure_cap_pct:
                    return False, f"Short exposure would be {short_exposure_pct:.1f}% > max {self.short_exposure_cap_pct}%"
        
        # 10. Check if we already have a position (might want to reject or limit)
        existing_pos = self.portfolio.get_position(intent.symbol)
        if existing_pos:
            # If adding to position, check if it's in same direction
            if intent.side.value == "buy" and existing_pos.qty < 0:
                pass  # Closing short position - OK
            elif intent.side.value == "sell" and existing_pos.qty > 0:
                pass  # Closing long position - OK
            else:
                # Adding to existing position - could implement additional checks
                logger.debug(f"Adding to existing position in {intent.symbol}")
        
        # All checks passed
        return True, ""
    
    def _trigger_kill_switch(self, reason: str) -> None:
        """
        Trigger the kill switch to stop all trading.
        
        Args:
            reason: Reason for triggering kill switch
        """
        if not self.kill_switch_enabled:
            logger.warning(f"Kill switch trigger ignored (disabled): {reason}")
            return
        
        self.kill_switch_triggered = True
        
        alert = AlertMessage(
            level="CRITICAL",
            title="KILL SWITCH TRIGGERED",
            message=f"Trading halted: {reason}",
            metadata={"reason": reason, "timestamp": datetime.now().isoformat()}
        )
        self.alerts.append(alert)
        
        logger.critical(f"🛑 KILL SWITCH TRIGGERED: {reason}")
        logger.critical("All trading is now HALTED - manual intervention required")
    
    def reset_kill_switch(self) -> None:
        """Reset the kill switch (manual intervention required)."""
        self.kill_switch_triggered = False
        logger.warning("Kill switch manually reset")
    
    def record_trade_result(self, pnl: float) -> None:
        """
        Record the result of a completed trade.
        
        Args:
            pnl: P&L of the trade
        """
        self.trades_today += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
            logger.info(f"Loss recorded | Consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0  # Reset on win
        
        # Record in portfolio
        self.portfolio.record_trade(pnl)
    
    def reset_daily_limits(self) -> None:
        """Reset daily tracking (call at start of day)."""
        self.trades_today = 0
        self.consecutive_losses = 0
        logger.info("Daily risk limits reset")
    
    def get_risk_summary(self) -> Dict:
        """
        Get current risk status summary.
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            'kill_switch_active': self.kill_switch_triggered,
            'daily_drawdown_pct': self.portfolio.daily_drawdown_pct(),
            'max_drawdown_pct': self.daily_max_drawdown_pct,
            'gross_exposure_pct': self.portfolio.gross_exposure_pct(),
            'net_exposure_pct': self.portfolio.net_exposure_pct(),
            'position_count': self.portfolio.position_count(),
            'max_positions': self.max_positions_total,
            'trades_today': self.trades_today,
            'max_trades_per_day': self.max_trades_per_day,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_losses_per_day
        }
    
    def log_risk_summary(self) -> None:
        """Log risk summary."""
        summary = self.get_risk_summary()
        logger.info(
            f"Risk Summary | "
            f"Kill switch: {summary['kill_switch_active']} | "
            f"DD: {summary['daily_drawdown_pct']:.2f}% / {summary['max_drawdown_pct']:.2f}% | "
            f"Exposure: {summary['gross_exposure_pct']:.1f}% | "
            f"Positions: {summary['position_count']}/{summary['max_positions']} | "
            f"Trades: {summary['trades_today']}/{summary['max_trades_per_day']} | "
            f"Losses: {summary['consecutive_losses']}/{summary['max_consecutive_losses']}"
        )
    
    def get_alerts(self) -> List[AlertMessage]:
        """
        Get all alerts.
        
        Returns:
            List of alert messages
        """
        return self.alerts
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
