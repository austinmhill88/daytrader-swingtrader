"""
Dynamic position sizing via regime detection and volatility targeting (Phase 3).
Adjusts position sizes based on market regime and volatility conditions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger
from datetime import datetime


class DynamicSizer:
    """
    Dynamic position sizing based on regime and volatility.
    Phase 3 implementation for execution sophistication.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize dynamic sizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        risk_config = config.get('risk', {})
        regime_config = config.get('regime', {})
        
        # Base sizing parameters
        self.base_per_trade_risk_pct = risk_config.get('per_trade_risk_pct', 0.4)
        self.max_position_size_pct = risk_config.get('max_position_size_pct', 5.0)
        
        # Regime-based adjustments
        self.regime_enabled = regime_config.get('enabled', False)
        
        # Volatility targeting
        self.target_volatility = 0.15  # 15% annualized target
        self.vol_scaling_enabled = True
        
        # Regime multipliers
        self.regime_multipliers = {
            'low_volatility': 1.2,    # More aggressive in low vol
            'medium_volatility': 1.0, # Normal sizing
            'high_volatility': 0.7,   # More conservative in high vol
            'crisis': 0.3            # Very conservative in crisis
        }
        
        # Kelly criterion parameters
        self.use_kelly = False  # Disabled by default (needs win rate data)
        self.kelly_fraction = 0.5  # Half-kelly for safety
        
        logger.info(
            f"DynamicSizer initialized | "
            f"Base risk: {self.base_per_trade_risk_pct}%, "
            f"Regime-based: {self.regime_enabled}"
        )
    
    def calculate_position_size(
        self,
        equity: float,
        price: float,
        atr: float,
        symbol: str,
        regime: Optional[str] = None,
        realized_vol: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> int:
        """
        Calculate dynamic position size.
        
        Args:
            equity: Current account equity
            price: Entry price
            atr: Average True Range
            symbol: Stock symbol
            regime: Current market regime (optional)
            realized_vol: Realized volatility (optional)
            win_rate: Historical win rate for Kelly (optional)
            avg_win: Average win size for Kelly (optional)
            avg_loss: Average loss size for Kelly (optional)
            
        Returns:
            Position size in shares
        """
        # Start with base risk percentage
        risk_pct = self.base_per_trade_risk_pct
        
        # Adjust for regime if available
        if self.regime_enabled and regime:
            regime_mult = self.regime_multipliers.get(regime, 1.0)
            risk_pct *= regime_mult
            logger.debug(f"Regime adjustment for {symbol}: {regime} -> {regime_mult}x")
        
        # Adjust for volatility targeting if available
        if self.vol_scaling_enabled and realized_vol:
            vol_mult = self._calculate_vol_scaling(realized_vol)
            risk_pct *= vol_mult
            logger.debug(f"Volatility adjustment for {symbol}: {realized_vol:.2%} -> {vol_mult:.2f}x")
        
        # Optional Kelly criterion
        if self.use_kelly and win_rate and avg_win and avg_loss:
            kelly_size = self._calculate_kelly_size(
                equity, win_rate, avg_win, avg_loss
            )
            # Use fraction of Kelly for safety
            kelly_size *= self.kelly_fraction
            logger.debug(f"Kelly size for {symbol}: {kelly_size:.0f} shares")
        
        # Calculate risk amount
        risk_amount = equity * (risk_pct / 100.0)
        
        # Calculate position size based on ATR stop
        stop_distance = atr * 1.5  # 1.5x ATR stop
        qty = int(risk_amount / stop_distance)
        
        # Apply maximum position size limit
        max_qty = int((equity * self.max_position_size_pct / 100.0) / price)
        qty = min(qty, max_qty)
        
        # Ensure positive quantity
        qty = max(1, qty)
        
        logger.debug(
            f"Position size for {symbol}: {qty} shares "
            f"(risk: {risk_pct:.2f}%, ATR stop: ${stop_distance:.2f})"
        )
        
        return qty
    
    def _calculate_vol_scaling(self, realized_vol: float) -> float:
        """
        Calculate volatility scaling multiplier.
        
        Args:
            realized_vol: Realized volatility (annualized)
            
        Returns:
            Scaling multiplier
        """
        # Scale position inversely with volatility
        # If vol is higher than target, reduce position size
        vol_ratio = self.target_volatility / realized_vol
        
        # Cap the adjustment (don't go too extreme)
        vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
        
        return vol_ratio
    
    def _calculate_kelly_size(
        self,
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> int:
        """
        Calculate position size using Kelly criterion.
        
        Args:
            equity: Current equity
            win_rate: Win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive)
            
        Returns:
            Position size in shares
        """
        # Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        loss_rate = 1 - win_rate
        
        kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Ensure non-negative
        kelly_fraction = max(0, kelly_fraction)
        
        # Apply Kelly fraction to equity
        kelly_size_usd = equity * kelly_fraction
        
        # This returns USD amount, caller needs to convert to shares
        return int(kelly_size_usd)
    
    def adjust_for_correlation(
        self,
        qty: int,
        symbol: str,
        portfolio_positions: Dict[str, int],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Adjust position size for portfolio correlation.
        Reduce size if adding highly correlated position.
        
        Args:
            qty: Proposed quantity
            symbol: Stock symbol
            portfolio_positions: Current portfolio positions
            correlation_matrix: Symbol correlation matrix (optional)
            
        Returns:
            Adjusted quantity
        """
        if not correlation_matrix or symbol not in correlation_matrix.index:
            return qty
        
        # Calculate average correlation with existing positions
        correlations = []
        for existing_symbol in portfolio_positions.keys():
            if existing_symbol in correlation_matrix.columns:
                corr = correlation_matrix.loc[symbol, existing_symbol]
                correlations.append(abs(corr))
        
        if not correlations:
            return qty
        
        avg_corr = np.mean(correlations)
        
        # Reduce size if high correlation
        if avg_corr > 0.7:
            adjustment = 0.7  # Reduce by 30%
        elif avg_corr > 0.5:
            adjustment = 0.85  # Reduce by 15%
        else:
            adjustment = 1.0  # No adjustment
        
        adjusted_qty = int(qty * adjustment)
        
        if adjusted_qty < qty:
            logger.debug(
                f"Correlation adjustment for {symbol}: "
                f"{qty} -> {adjusted_qty} (avg corr: {avg_corr:.2f})"
            )
        
        return adjusted_qty
    
    def calculate_portfolio_heat(
        self,
        portfolio_positions: Dict[str, Dict],
        equity: float
    ) -> float:
        """
        Calculate total portfolio risk exposure ('heat').
        
        Args:
            portfolio_positions: Dict of symbol -> {qty, entry_price, stop_price}
            equity: Current equity
            
        Returns:
            Portfolio heat as percentage of equity
        """
        total_risk = 0.0
        
        for symbol, pos in portfolio_positions.items():
            qty = pos.get('qty', 0)
            entry_price = pos.get('entry_price', 0)
            stop_price = pos.get('stop_price', 0)
            
            if qty == 0 or entry_price == 0 or stop_price == 0:
                continue
            
            # Calculate risk per position
            risk_per_share = abs(entry_price - stop_price)
            position_risk = qty * risk_per_share
            total_risk += position_risk
        
        portfolio_heat = (total_risk / equity) * 100.0 if equity > 0 else 0.0
        
        return portfolio_heat
    
    def should_reduce_sizing(
        self,
        portfolio_heat: float,
        consecutive_losses: int
    ) -> Tuple[bool, float]:
        """
        Determine if position sizing should be reduced.
        
        Args:
            portfolio_heat: Current portfolio heat percentage
            consecutive_losses: Number of consecutive losses
            
        Returns:
            Tuple of (should_reduce, multiplier)
        """
        # Reduce sizing if portfolio heat is high
        if portfolio_heat > 15.0:
            return True, 0.5  # Reduce by 50%
        elif portfolio_heat > 10.0:
            return True, 0.75  # Reduce by 25%
        
        # Reduce sizing after consecutive losses
        if consecutive_losses >= 5:
            return True, 0.5  # Reduce by 50%
        elif consecutive_losses >= 3:
            return True, 0.75  # Reduce by 25%
        
        return False, 1.0
    
    def get_sizing_summary(
        self,
        equity: float,
        regime: Optional[str] = None
    ) -> Dict:
        """
        Get summary of current sizing parameters.
        
        Args:
            equity: Current equity
            regime: Current market regime
            
        Returns:
            Dictionary with sizing parameters
        """
        risk_pct = self.base_per_trade_risk_pct
        
        if self.regime_enabled and regime:
            regime_mult = self.regime_multipliers.get(regime, 1.0)
            risk_pct *= regime_mult
        
        return {
            'base_risk_pct': self.base_per_trade_risk_pct,
            'adjusted_risk_pct': risk_pct,
            'max_position_pct': self.max_position_size_pct,
            'regime': regime,
            'regime_enabled': self.regime_enabled,
            'vol_scaling_enabled': self.vol_scaling_enabled,
            'use_kelly': self.use_kelly,
            'max_position_usd': equity * self.max_position_size_pct / 100.0
        }
