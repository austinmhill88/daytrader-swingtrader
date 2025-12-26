"""
Intraday exposure caps with auto-flatten windows and emergency flatten logic (Phase 3).
Manages intraday risk limits and automatic position flattening.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from loguru import logger


class ExposureManager:
    """
    Manages intraday exposure limits and auto-flatten logic.
    Phase 3 implementation for execution sophistication.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize exposure manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        risk_config = config.get('risk', {})
        session_config = config.get('session', {})
        
        # Exposure limits
        self.max_gross_exposure_pct_intraday = risk_config.get(
            'max_gross_exposure_pct_intraday', 50
        )
        self.max_gross_exposure_pct_swing = risk_config.get(
            'max_gross_exposure_pct_swing', 100
        )
        
        # Auto-flatten windows
        self.market_close_time = self._parse_time(
            session_config.get('market_close', '16:00')
        )
        self.flatten_intraday_time = self._parse_time('15:50')  # 10 min before close
        self.flatten_intraday_enabled = True
        
        # Emergency flatten parameters
        self.emergency_flatten_enabled = True
        self.emergency_flatten_drawdown_pct = 5.0  # Flatten all if DD > 5%
        self.emergency_flatten_loss_streak = 10  # Flatten after 10 consecutive losses
        
        # Tracking
        self.positions_flattened_today = 0
        self.last_flatten_time: Optional[datetime] = None
        self.emergency_flatten_triggered = False
        
        logger.info(
            f"ExposureManager initialized | "
            f"Intraday limit: {self.max_gross_exposure_pct_intraday}%, "
            f"Flatten time: {self.flatten_intraday_time.strftime('%H:%M')}"
        )
    
    def check_exposure_limits(
        self,
        equity: float,
        current_positions: Dict[str, Dict],
        new_position_value: float,
        strategy_type: str = "intraday"
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if adding a new position would violate exposure limits.
        
        Args:
            equity: Current account equity
            current_positions: Dict of symbol -> position info
            new_position_value: Value of proposed new position
            strategy_type: "intraday" or "swing"
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Calculate current gross exposure
        current_exposure = sum(
            abs(pos['market_value'])
            for pos in current_positions.values()
        )
        
        # Calculate new gross exposure
        new_exposure = current_exposure + abs(new_position_value)
        new_exposure_pct = (new_exposure / equity) * 100.0 if equity > 0 else 0.0
        
        # Check limit based on strategy type
        if strategy_type == "intraday":
            limit = self.max_gross_exposure_pct_intraday
        else:
            limit = self.max_gross_exposure_pct_swing
        
        if new_exposure_pct > limit:
            reason = (
                f"Exposure limit exceeded: {new_exposure_pct:.1f}% > {limit}% "
                f"({strategy_type} limit)"
            )
            return False, reason
        
        return True, None
    
    def should_flatten_intraday(
        self,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if it's time to flatten intraday positions.
        
        Args:
            current_time: Current time (defaults to now)
            
        Returns:
            True if positions should be flattened
        """
        if not self.flatten_intraday_enabled:
            return False
        
        if current_time is None:
            current_time = datetime.now()
        
        current_time_only = current_time.time()
        
        # Check if past flatten time
        return current_time_only >= self.flatten_intraday_time
    
    def should_emergency_flatten(
        self,
        daily_drawdown_pct: float,
        consecutive_losses: int,
        portfolio_heat: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if emergency flatten should be triggered.
        
        Args:
            daily_drawdown_pct: Current daily drawdown percentage
            consecutive_losses: Number of consecutive losses
            portfolio_heat: Portfolio heat percentage
            
        Returns:
            Tuple of (should_flatten, reason)
        """
        if not self.emergency_flatten_enabled:
            return False, None
        
        # Already triggered today
        if self.emergency_flatten_triggered:
            return False, "Emergency flatten already triggered today"
        
        # Check drawdown threshold
        if daily_drawdown_pct >= self.emergency_flatten_drawdown_pct:
            reason = (
                f"Emergency flatten: Daily drawdown {daily_drawdown_pct:.1f}% "
                f">= {self.emergency_flatten_drawdown_pct}%"
            )
            self.emergency_flatten_triggered = True
            return True, reason
        
        # Check loss streak
        if consecutive_losses >= self.emergency_flatten_loss_streak:
            reason = (
                f"Emergency flatten: {consecutive_losses} consecutive losses "
                f">= {self.emergency_flatten_loss_streak}"
            )
            self.emergency_flatten_triggered = True
            return True, reason
        
        # Check excessive portfolio heat
        if portfolio_heat > 20.0:
            reason = f"Emergency flatten: Portfolio heat {portfolio_heat:.1f}% > 20%"
            self.emergency_flatten_triggered = True
            return True, reason
        
        return False, None
    
    def get_positions_to_flatten(
        self,
        current_positions: Dict[str, Dict],
        strategy_type: str = "intraday",
        current_time: Optional[datetime] = None
    ) -> List[str]:
        """
        Get list of symbols that should be flattened.
        
        Args:
            current_positions: Dict of symbol -> position info
            strategy_type: Filter by strategy type ("intraday" or "swing")
            current_time: Current time (defaults to now)
            
        Returns:
            List of symbols to flatten
        """
        to_flatten = []
        
        for symbol, pos in current_positions.items():
            pos_strategy = pos.get('strategy_type', 'intraday')
            
            # Flatten intraday positions at EOD
            if (pos_strategy == 'intraday' and 
                strategy_type == 'intraday' and
                self.should_flatten_intraday(current_time)):
                to_flatten.append(symbol)
        
        return to_flatten
    
    def flatten_all_positions(
        self,
        current_positions: Dict[str, Dict],
        reason: str
    ) -> List[str]:
        """
        Mark all positions for emergency flatten.
        
        Args:
            current_positions: Dict of symbol -> position info
            reason: Reason for emergency flatten
            
        Returns:
            List of all symbols to flatten
        """
        self.positions_flattened_today += len(current_positions)
        self.last_flatten_time = datetime.now()
        
        logger.critical(
            f"EMERGENCY FLATTEN: {reason} | "
            f"Flattening {len(current_positions)} positions"
        )
        
        return list(current_positions.keys())
    
    def reduce_exposure(
        self,
        current_positions: Dict[str, Dict],
        equity: float,
        target_exposure_pct: float
    ) -> List[Tuple[str, float]]:
        """
        Calculate position reductions to meet target exposure.
        
        Args:
            current_positions: Dict of symbol -> position info
            equity: Current equity
            target_exposure_pct: Target exposure percentage
            
        Returns:
            List of (symbol, reduction_pct) tuples
        """
        # Calculate current exposure
        current_exposure = sum(
            abs(pos['market_value'])
            for pos in current_positions.values()
        )
        current_exposure_pct = (current_exposure / equity) * 100.0 if equity > 0 else 0.0
        
        if current_exposure_pct <= target_exposure_pct:
            return []
        
        # Calculate required reduction
        reduction_ratio = target_exposure_pct / current_exposure_pct
        
        # Proportionally reduce all positions
        reductions = []
        for symbol, pos in current_positions.items():
            reduction_pct = (1.0 - reduction_ratio) * 100.0
            if reduction_pct > 1.0:  # Only reduce if > 1%
                reductions.append((symbol, reduction_pct))
        
        logger.info(
            f"Exposure reduction: {current_exposure_pct:.1f}% -> {target_exposure_pct:.1f}% "
            f"({len(reductions)} positions affected)"
        )
        
        return reductions
    
    def get_max_new_position_size(
        self,
        equity: float,
        current_positions: Dict[str, Dict],
        strategy_type: str = "intraday"
    ) -> float:
        """
        Calculate maximum size for a new position given current exposure.
        
        Args:
            equity: Current account equity
            current_positions: Dict of symbol -> position info
            strategy_type: "intraday" or "swing"
            
        Returns:
            Maximum position value in USD
        """
        # Calculate current exposure
        current_exposure = sum(
            abs(pos['market_value'])
            for pos in current_positions.values()
        )
        
        # Determine limit
        if strategy_type == "intraday":
            limit = self.max_gross_exposure_pct_intraday
        else:
            limit = self.max_gross_exposure_pct_swing
        
        # Calculate remaining capacity
        max_total_exposure = equity * (limit / 100.0)
        remaining_capacity = max_total_exposure - current_exposure
        
        return max(0.0, remaining_capacity)
    
    def reset_daily_limits(self) -> None:
        """Reset daily tracking (call at start of day)."""
        self.positions_flattened_today = 0
        self.last_flatten_time = None
        self.emergency_flatten_triggered = False
        logger.info("Exposure manager daily limits reset")
    
    def _parse_time(self, time_str: str) -> time:
        """
        Parse time string to time object.
        
        Args:
            time_str: Time string in HH:MM format
            
        Returns:
            time object
        """
        try:
            return datetime.strptime(time_str, '%H:%M').time()
        except ValueError:
            logger.error(f"Invalid time format: {time_str}, using default")
            return time(16, 0)
    
    def get_exposure_summary(
        self,
        equity: float,
        current_positions: Dict[str, Dict]
    ) -> Dict:
        """
        Get summary of current exposure status.
        
        Args:
            equity: Current equity
            current_positions: Dict of symbol -> position info
            
        Returns:
            Dictionary with exposure statistics
        """
        if not current_positions:
            return {
                'gross_exposure_usd': 0.0,
                'gross_exposure_pct': 0.0,
                'net_exposure_usd': 0.0,
                'net_exposure_pct': 0.0,
                'long_exposure_usd': 0.0,
                'short_exposure_usd': 0.0,
                'num_positions': 0,
                'intraday_flatten_active': self.should_flatten_intraday()
            }
        
        # Calculate exposures
        long_exposure = sum(
            pos['market_value']
            for pos in current_positions.values()
            if pos['market_value'] > 0
        )
        short_exposure = sum(
            abs(pos['market_value'])
            for pos in current_positions.values()
            if pos['market_value'] < 0
        )
        
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        return {
            'gross_exposure_usd': gross_exposure,
            'gross_exposure_pct': (gross_exposure / equity * 100.0) if equity > 0 else 0.0,
            'net_exposure_usd': net_exposure,
            'net_exposure_pct': (net_exposure / equity * 100.0) if equity > 0 else 0.0,
            'long_exposure_usd': long_exposure,
            'short_exposure_usd': short_exposure,
            'num_positions': len(current_positions),
            'intraday_flatten_active': self.should_flatten_intraday(),
            'emergency_flatten_triggered': self.emergency_flatten_triggered,
            'positions_flattened_today': self.positions_flattened_today
        }
