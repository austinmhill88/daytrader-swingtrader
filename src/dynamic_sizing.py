"""
Dynamic position sizing via regime detection and volatility targeting (Phase 3).
Adjusts position sizes based on market regime and volatility conditions.
Includes EWMA volatility targeting and ADV-based constraints.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
from datetime import datetime
from collections import defaultdict


class DynamicSizer:
    """
    Dynamic position sizing based on regime and volatility.
    Phase 3 implementation for execution sophistication with EWMA volatility targeting.
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
        sizing_config = risk_config.get('dynamic_sizing', {})
        
        # Base sizing parameters
        self.base_per_trade_risk_pct = risk_config.get('per_trade_risk_pct', 0.4)
        self.max_position_size_pct = risk_config.get('max_position_size_pct', 5.0)
        
        # Regime-based adjustments
        self.regime_enabled = regime_config.get('enabled', False)
        
        # EWMA volatility targeting
        self.enable_vol_targeting = sizing_config.get('enable_vol_targeting', True)
        self.target_portfolio_volatility = sizing_config.get('target_portfolio_volatility', 0.10)  # 10% annualized
        self.vol_ewma_span = sizing_config.get('vol_ewma_span', 20)  # EWMA span for volatility
        self.vol_lookback_days = sizing_config.get('vol_lookback_days', 60)  # Lookback for realized vol
        
        # Per-symbol EWMA volatility tracking
        self.symbol_ewma_vols: Dict[str, float] = {}
        self.symbol_vol_history: Dict[str, list] = defaultdict(list)
        
        # Portfolio volatility state
        self.portfolio_realized_vol = None
        self.portfolio_vol_history = []
        
        # ADV-based participation constraints
        self.enable_adv_constraints = sizing_config.get('enable_adv_constraints', True)
        self.max_participation_rate_intraday = sizing_config.get('max_participation_rate_intraday', 0.01)  # 1% of ADV
        self.max_participation_rate_swing = sizing_config.get('max_participation_rate_swing', 0.02)  # 2% of ADV
        self.min_adv_usd = config.get('universe', {}).get('min_adv_usd', 1000000)  # $1M minimum ADV
        
        # Regime multipliers
        self.regime_multipliers = {
            'low_volatility': 1.2,    # More aggressive in low vol
            'medium_volatility': 1.0, # Normal sizing
            'high_volatility': 0.7,   # More conservative in high vol
            'crisis': 0.3,           # Very conservative in crisis
            'trending_up': 1.1,
            'trending_down': 0.8,
            'ranging': 1.0
        }
        
        # Kelly criterion parameters
        self.use_kelly = False  # Disabled by default (needs win rate data)
        self.kelly_fraction = 0.5  # Half-kelly for safety
        
        logger.info(
            f"DynamicSizer initialized | "
            f"Base risk: {self.base_per_trade_risk_pct}%, "
            f"Vol targeting: {self.enable_vol_targeting}, "
            f"Target vol: {self.target_portfolio_volatility:.1%}, "
            f"ADV constraints: {self.enable_adv_constraints}"
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
    
    def update_symbol_volatility(
        self,
        symbol: str,
        returns: pd.Series
    ) -> float:
        """
        Update EWMA volatility estimate for a symbol.
        
        Args:
            symbol: Stock symbol
            returns: Series of returns
            
        Returns:
            Updated EWMA volatility (annualized)
        """
        if len(returns) == 0:
            return 0.15  # Default 15% vol
        
        # Calculate EWMA volatility
        ewma_vol = returns.ewm(span=self.vol_ewma_span).std().iloc[-1]
        
        # Annualize (assuming daily returns)
        annualized_vol = ewma_vol * np.sqrt(252)
        
        # Store in cache
        self.symbol_ewma_vols[symbol] = annualized_vol
        
        # Update history (keep last 100 observations)
        self.symbol_vol_history[symbol].append(annualized_vol)
        if len(self.symbol_vol_history[symbol]) > 100:
            self.symbol_vol_history[symbol].pop(0)
        
        logger.debug(f"Updated EWMA vol for {symbol}: {annualized_vol:.2%}")
        
        return annualized_vol
    
    def get_symbol_volatility(self, symbol: str) -> float:
        """
        Get current EWMA volatility estimate for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            EWMA volatility (annualized)
        """
        return self.symbol_ewma_vols.get(symbol, 0.15)  # Default 15%
    
    def update_portfolio_volatility(
        self,
        portfolio_returns: pd.Series
    ) -> float:
        """
        Update portfolio-level volatility estimate.
        
        Args:
            portfolio_returns: Series of portfolio returns
            
        Returns:
            Portfolio realized volatility (annualized)
        """
        if len(portfolio_returns) == 0:
            return self.target_portfolio_volatility
        
        # Calculate EWMA portfolio volatility
        realized_vol = portfolio_returns.ewm(span=self.vol_ewma_span).std().iloc[-1]
        
        # Annualize
        annualized_vol = realized_vol * np.sqrt(252)
        
        self.portfolio_realized_vol = annualized_vol
        
        # Update history
        self.portfolio_vol_history.append(annualized_vol)
        if len(self.portfolio_vol_history) > 100:
            self.portfolio_vol_history.pop(0)
        
        logger.info(f"Portfolio realized vol: {annualized_vol:.2%} (target: {self.target_portfolio_volatility:.2%})")
        
        return annualized_vol
    
    def calculate_vol_target_multiplier(self) -> float:
        """
        Calculate position size multiplier based on volatility targeting.
        Scale positions to target portfolio volatility.
        
        Returns:
            Multiplier for position sizing (0.5 to 2.0)
        """
        if not self.enable_vol_targeting or self.portfolio_realized_vol is None:
            return 1.0
        
        # Calculate how much to scale positions
        # If realized vol > target, scale down; if realized vol < target, scale up
        multiplier = self.target_portfolio_volatility / self.portfolio_realized_vol
        
        # Clamp to reasonable range
        multiplier = np.clip(multiplier, 0.5, 2.0)
        
        logger.debug(
            f"Vol targeting multiplier: {multiplier:.2f} "
            f"(realized: {self.portfolio_realized_vol:.2%}, target: {self.target_portfolio_volatility:.2%})"
        )
        
        return multiplier
    
    def check_adv_constraints(
        self,
        symbol: str,
        qty: int,
        price: float,
        adv_shares: Optional[float] = None,
        strategy_type: str = 'intraday'
    ) -> Tuple[bool, int, str]:
        """
        Check and enforce ADV-based participation rate constraints.
        
        Args:
            symbol: Stock symbol
            qty: Proposed quantity
            price: Current price
            adv_shares: Average daily volume in shares
            strategy_type: 'intraday' or 'swing'
            
        Returns:
            Tuple of (is_valid, adjusted_qty, reason)
        """
        if not self.enable_adv_constraints or adv_shares is None:
            return True, qty, ""
        
        # Calculate ADV in USD
        adv_usd = adv_shares * price
        
        # Check minimum ADV requirement
        if adv_usd < self.min_adv_usd:
            return False, 0, f"ADV ${adv_usd:,.0f} below minimum ${self.min_adv_usd:,.0f}"
        
        # Determine max participation rate
        if strategy_type == 'intraday':
            max_participation = self.max_participation_rate_intraday
        else:
            max_participation = self.max_participation_rate_swing
        
        # Calculate maximum quantity based on participation rate
        max_qty = int(adv_shares * max_participation)
        
        if qty > max_qty:
            logger.info(
                f"ADV constraint for {symbol}: reducing {qty} -> {max_qty} shares "
                f"({max_participation:.1%} of ADV)"
            )
            return True, max_qty, f"Reduced to {max_participation:.1%} of ADV"
        
        return True, qty, ""
    
    def calculate_dollar_risk_per_trade(
        self,
        equity: float,
        symbol_vol: Optional[float] = None
    ) -> float:
        """
        Calculate dollar risk per trade with volatility adjustment.
        
        Args:
            equity: Current account equity
            symbol_vol: Symbol's EWMA volatility (optional)
            
        Returns:
            Dollar risk amount for this trade
        """
        # Base risk as percentage of equity
        base_risk_dollars = equity * (self.base_per_trade_risk_pct / 100.0)
        
        # Adjust for portfolio volatility targeting
        vol_multiplier = self.calculate_vol_target_multiplier()
        base_risk_dollars *= vol_multiplier
        
        # Adjust for symbol-specific volatility if available
        if symbol_vol is not None and self.enable_vol_targeting:
            # Scale inversely with symbol volatility
            # Higher vol stocks get smaller position sizes
            symbol_vol_adjustment = 0.15 / symbol_vol  # 15% baseline
            symbol_vol_adjustment = np.clip(symbol_vol_adjustment, 0.5, 2.0)
            base_risk_dollars *= symbol_vol_adjustment
        
        return base_risk_dollars
    
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
