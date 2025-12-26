"""
Slippage and spread forecasting by symbol and time-of-day (Phase 3).
Provides symbol-specific slippage curves and time-based spread modeling.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, time
from loguru import logger
from pathlib import Path
import json


class SlippageModel:
    """
    Symbol-specific slippage forecasting with time-of-day modeling.
    Phase 3 implementation for execution sophistication.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize slippage model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        execution_config = config.get('execution', {})
        
        self.max_slippage_bps = execution_config.get('max_slippage_bps', 20)
        self.model_dir = Path(config.get('storage', {}).get('model_dir', './data/models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Slippage curves by symbol (historical averages)
        self.symbol_slippage: Dict[str, Dict] = {}
        
        # Time-of-day spread patterns
        self.time_of_day_patterns: Dict[str, Dict] = {}
        
        # Default slippage parameters
        self.default_slippage_bps = 5.0
        self.market_open_slippage_multiplier = 2.0  # Higher slippage at open
        self.market_close_slippage_multiplier = 1.5  # Higher slippage at close
        
        # Time windows for different slippage regimes
        self.market_open_time = time(9, 30)
        self.market_open_window_minutes = 30
        self.market_close_time = time(16, 0)
        self.market_close_window_minutes = 30
        
        logger.info("SlippageModel initialized")
    
    def forecast_slippage(
        self,
        symbol: str,
        side: str,
        qty: int,
        current_price: float,
        current_time: Optional[datetime] = None,
        adv_shares: Optional[float] = None,
        spread_bps: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Forecast expected slippage for a trade with detailed breakdown.
        Includes half-spread cost and market impact.
        
        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            qty: Order quantity
            current_price: Current price
            current_time: Time of trade (defaults to now)
            adv_shares: Average daily volume in shares (for impact model)
            spread_bps: Current bid-ask spread in bps (if known)
            
        Returns:
            Dictionary with slippage components:
            - total_slippage_bps: Total expected slippage
            - half_spread_bps: Half-spread crossing cost
            - market_impact_bps: Market impact cost
            - timing_cost_bps: Time-of-day premium
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Component 1: Half-spread cost
        if spread_bps is None:
            spread_bps = self.forecast_spread(symbol, current_time)
        half_spread = spread_bps / 2.0
        
        # Component 2: Market impact
        market_impact = self._calculate_market_impact(
            qty, current_price, adv_shares
        )
        
        # Component 3: Time-of-day adjustment
        time_multiplier = self._get_time_multiplier(current_time.time())
        timing_cost = self._get_symbol_baseline(symbol) * (time_multiplier - 1.0)
        
        # Component 4: Side adjustment (sells slightly higher slippage)
        side_adjustment = 0.5 if side.lower() == "sell" else 0.0
        
        # Total slippage = half_spread + market_impact + timing + side
        total_slippage = half_spread + market_impact + timing_cost + side_adjustment
        
        # Cap at maximum
        total_slippage = min(total_slippage, self.max_slippage_bps)
        
        breakdown = {
            'total_slippage_bps': total_slippage,
            'half_spread_bps': half_spread,
            'market_impact_bps': market_impact,
            'timing_cost_bps': timing_cost,
            'side_adjustment_bps': side_adjustment,
            'time_multiplier': time_multiplier
        }
        
        logger.debug(
            f"Slippage forecast for {symbol}: {total_slippage:.2f} bps "
            f"(spread: {half_spread:.2f}, impact: {market_impact:.2f}, "
            f"timing: {timing_cost:.2f}, side: {side_adjustment:.2f})"
        )
        
        return breakdown
    
    def forecast_spread(
        self,
        symbol: str,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Forecast expected bid-ask spread.
        
        Args:
            symbol: Stock symbol
            current_time: Time of forecast (defaults to now)
            
        Returns:
            Estimated spread in basis points
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Get symbol-specific spread pattern
        if symbol in self.time_of_day_patterns:
            pattern = self.time_of_day_patterns[symbol]
            hour = current_time.hour
            
            # Use hour-specific spread if available
            if str(hour) in pattern.get('hourly_spreads', {}):
                return pattern['hourly_spreads'][str(hour)]
        
        # Default spread estimate (wider at open/close)
        base_spread = 10.0  # 10 bps default
        time_multiplier = self._get_time_multiplier(current_time.time())
        
        return base_spread * time_multiplier
    
    def _get_symbol_baseline(self, symbol: str) -> float:
        """Get symbol-specific baseline slippage."""
        if symbol in self.symbol_slippage:
            return self.symbol_slippage[symbol].get('avg_slippage_bps', self.default_slippage_bps)
        
        return self.default_slippage_bps
    
    def _get_time_multiplier(self, current_time: time) -> float:
        """
        Calculate time-of-day multiplier for slippage/spread.
        
        Args:
            current_time: Time of day
            
        Returns:
            Multiplier (1.0 = normal, >1.0 = higher slippage)
        """
        # Convert to minutes since market open
        current_minutes = current_time.hour * 60 + current_time.minute
        open_minutes = self.market_open_time.hour * 60 + self.market_open_time.minute
        close_minutes = self.market_close_time.hour * 60 + self.market_close_time.minute
        
        # High slippage window at market open
        if abs(current_minutes - open_minutes) <= self.market_open_window_minutes:
            return self.market_open_slippage_multiplier
        
        # High slippage window at market close
        if abs(current_minutes - close_minutes) <= self.market_close_window_minutes:
            return self.market_close_slippage_multiplier
        
        # Normal hours
        return 1.0
    
    def _get_size_multiplier(self, qty: int, price: float) -> float:
        """
        Calculate size-based multiplier for slippage.
        Larger orders have more impact.
        
        Args:
            qty: Order quantity
            price: Current price
            
        Returns:
            Multiplier (1.0 = small order, >1.0 = large order)
        """
        notional = qty * price
        
        # Size tiers (in USD)
        if notional < 5000:
            return 1.0  # Small order
        elif notional < 10000:
            return 1.1
        elif notional < 25000:
            return 1.2
        elif notional < 50000:
            return 1.3
        else:
            return 1.5  # Large order
    
    def _calculate_market_impact(
        self,
        qty: int,
        price: float,
        adv_shares: Optional[float] = None
    ) -> float:
        """
        Calculate market impact cost in basis points.
        Uses square-root impact model: impact ~ (qty / ADV)^0.5
        
        Args:
            qty: Order quantity
            price: Current price
            adv_shares: Average daily volume in shares
            
        Returns:
            Market impact in basis points
        """
        if adv_shares is None or adv_shares <= 0:
            # Fallback to simple size-based estimate
            notional = qty * price
            if notional < 10000:
                return 2.0  # 2 bps for small orders
            elif notional < 50000:
                return 5.0
            else:
                return 10.0  # 10 bps for large orders
        
        # Participation rate
        participation = qty / adv_shares
        
        # Square-root impact model
        # impact = base_impact * sqrt(participation_rate)
        base_impact = 10.0  # 10 bps at 1% participation
        impact_bps = base_impact * np.sqrt(participation / 0.01)
        
        # Cap impact at reasonable level
        impact_bps = min(impact_bps, 50.0)  # Max 50 bps impact
        
        return impact_bps
    
    def update_symbol_stats(
        self,
        symbol: str,
        actual_slippage_bps: float,
        spread_bps: float,
        trade_time: datetime,
        theoretical_price: Optional[float] = None,
        filled_price: Optional[float] = None
    ) -> None:
        """
        Update symbol-specific slippage statistics from actual fills.
        Learn from realized slippage vs. forecasted.
        
        Args:
            symbol: Stock symbol
            actual_slippage_bps: Actual observed slippage
            spread_bps: Actual observed spread
            trade_time: Time of trade
            theoretical_price: Expected/theoretical execution price
            filled_price: Actual fill price
        """
        if symbol not in self.symbol_slippage:
            self.symbol_slippage[symbol] = {
                'avg_slippage_bps': actual_slippage_bps,
                'samples': 1,
                'last_updated': trade_time.isoformat(),
                'slippage_history': [actual_slippage_bps],
                'forecast_errors': []
            }
        else:
            stats = self.symbol_slippage[symbol]
            n = stats['samples']
            # Running average
            stats['avg_slippage_bps'] = (
                (stats['avg_slippage_bps'] * n + actual_slippage_bps) / (n + 1)
            )
            stats['samples'] = n + 1
            stats['last_updated'] = trade_time.isoformat()
            
            # Track history (keep last 100)
            if 'slippage_history' not in stats:
                stats['slippage_history'] = []
            stats['slippage_history'].append(actual_slippage_bps)
            if len(stats['slippage_history']) > 100:
                stats['slippage_history'].pop(0)
        
        # Update time-of-day patterns
        self._update_time_pattern(symbol, spread_bps, trade_time)
        
        # Learn from forecast errors if we have both theoretical and filled prices
        if theoretical_price is not None and filled_price is not None:
            self._update_forecast_model(
                symbol, theoretical_price, filled_price, actual_slippage_bps
            )
        
        logger.debug(
            f"Updated slippage stats for {symbol}: "
            f"{self.symbol_slippage[symbol]['avg_slippage_bps']:.2f} bps "
            f"({self.symbol_slippage[symbol]['samples']} samples)"
        )
    
    def _update_forecast_model(
        self,
        symbol: str,
        theoretical_price: float,
        filled_price: float,
        actual_slippage_bps: float
    ) -> None:
        """
        Update forecast model by learning from actual fills.
        
        Args:
            symbol: Stock symbol
            theoretical_price: Expected price
            filled_price: Actual fill price
            actual_slippage_bps: Realized slippage
        """
        # Calculate forecast error
        price_error = abs(filled_price - theoretical_price)
        error_bps = (price_error / theoretical_price) * 10000
        
        # Store forecast error
        if symbol in self.symbol_slippage:
            if 'forecast_errors' not in self.symbol_slippage[symbol]:
                self.symbol_slippage[symbol]['forecast_errors'] = []
            
            self.symbol_slippage[symbol]['forecast_errors'].append({
                'error_bps': error_bps,
                'actual_slippage_bps': actual_slippage_bps,
                'theoretical_price': theoretical_price,
                'filled_price': filled_price
            })
            
            # Keep last 100 errors
            if len(self.symbol_slippage[symbol]['forecast_errors']) > 100:
                self.symbol_slippage[symbol]['forecast_errors'].pop(0)
            
            # Calculate rolling forecast accuracy
            errors = [e['error_bps'] for e in self.symbol_slippage[symbol]['forecast_errors']]
            mae = np.mean(errors)  # Mean absolute error
            
            logger.debug(
                f"Forecast error for {symbol}: {error_bps:.2f} bps "
                f"(rolling MAE: {mae:.2f} bps)"
            )
    
    def _update_time_pattern(
        self,
        symbol: str,
        spread_bps: float,
        trade_time: datetime
    ) -> None:
        """Update time-of-day spread patterns."""
        hour = trade_time.hour
        
        if symbol not in self.time_of_day_patterns:
            self.time_of_day_patterns[symbol] = {
                'hourly_spreads': {},
                'hourly_samples': {}
            }
        
        pattern = self.time_of_day_patterns[symbol]
        hour_key = str(hour)
        
        if hour_key not in pattern['hourly_spreads']:
            pattern['hourly_spreads'][hour_key] = spread_bps
            pattern['hourly_samples'][hour_key] = 1
        else:
            n = pattern['hourly_samples'][hour_key]
            pattern['hourly_spreads'][hour_key] = (
                (pattern['hourly_spreads'][hour_key] * n + spread_bps) / (n + 1)
            )
            pattern['hourly_samples'][hour_key] = n + 1
    
    def save_model(self) -> bool:
        """
        Save slippage model to disk.
        
        Returns:
            True if successful
        """
        try:
            model_data = {
                'symbol_slippage': self.symbol_slippage,
                'time_of_day_patterns': self.time_of_day_patterns,
                'last_saved': datetime.now().isoformat()
            }
            
            filepath = self.model_dir / 'slippage_model.json'
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Slippage model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving slippage model: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load slippage model from disk.
        
        Returns:
            True if successful
        """
        try:
            filepath = self.model_dir / 'slippage_model.json'
            if not filepath.exists():
                logger.info("No existing slippage model found")
                return False
            
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.symbol_slippage = model_data.get('symbol_slippage', {})
            self.time_of_day_patterns = model_data.get('time_of_day_patterns', {})
            
            logger.info(
                f"Slippage model loaded from {filepath} "
                f"({len(self.symbol_slippage)} symbols)"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error loading slippage model: {e}")
            return False
    
    def get_slippage_summary(self) -> Dict:
        """
        Get summary statistics of slippage model.
        
        Returns:
            Dictionary with model statistics
        """
        if not self.symbol_slippage:
            return {'symbols': 0, 'avg_slippage_bps': None}
        
        avg_slippages = [
            stats['avg_slippage_bps']
            for stats in self.symbol_slippage.values()
        ]
        
        # Calculate forecast accuracy if available
        total_errors = []
        for stats in self.symbol_slippage.values():
            if 'forecast_errors' in stats and stats['forecast_errors']:
                errors = [e['error_bps'] for e in stats['forecast_errors']]
                total_errors.extend(errors)
        
        forecast_mae = np.mean(total_errors) if total_errors else None
        
        summary = {
            'symbols': len(self.symbol_slippage),
            'avg_slippage_bps': np.mean(avg_slippages),
            'median_slippage_bps': np.median(avg_slippages),
            'max_slippage_bps': np.max(avg_slippages),
            'min_slippage_bps': np.min(avg_slippages),
            'total_samples': sum(
                stats['samples']
                for stats in self.symbol_slippage.values()
            ),
            'forecast_mae_bps': forecast_mae,
            'forecast_samples': len(total_errors)
        }
        
        return summary
    
    def get_total_transaction_cost(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        commission_per_share: float = 0.0,
        adv_shares: Optional[float] = None,
        spread_bps: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate total transaction cost including commissions, spread, and impact.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            qty: Order quantity
            price: Execution price
            commission_per_share: Commission per share (default 0 for Alpaca)
            adv_shares: Average daily volume
            spread_bps: Current spread in bps
            
        Returns:
            Dictionary with cost breakdown in dollars and basis points
        """
        # Get slippage forecast
        slippage_breakdown = self.forecast_slippage(
            symbol, side, qty, price, 
            adv_shares=adv_shares, 
            spread_bps=spread_bps
        )
        
        # Calculate costs in dollars
        notional = qty * price
        
        # Commission cost
        commission_dollars = qty * commission_per_share
        commission_bps = (commission_dollars / notional) * 10000 if notional > 0 else 0
        
        # Slippage cost
        slippage_bps = slippage_breakdown['total_slippage_bps']
        slippage_dollars = (slippage_bps / 10000) * notional
        
        # Total cost
        total_cost_dollars = commission_dollars + slippage_dollars
        total_cost_bps = commission_bps + slippage_bps
        
        return {
            'total_cost_dollars': total_cost_dollars,
            'total_cost_bps': total_cost_bps,
            'commission_dollars': commission_dollars,
            'commission_bps': commission_bps,
            'slippage_dollars': slippage_dollars,
            'slippage_bps': slippage_bps,
            'half_spread_bps': slippage_breakdown['half_spread_bps'],
            'market_impact_bps': slippage_breakdown['market_impact_bps'],
            'notional': notional
        }
