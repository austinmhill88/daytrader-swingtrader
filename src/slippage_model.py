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
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Forecast expected slippage for a trade.
        
        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            qty: Order quantity
            current_price: Current price
            current_time: Time of trade (defaults to now)
            
        Returns:
            Estimated slippage in basis points
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Start with symbol-specific baseline
        base_slippage = self._get_symbol_baseline(symbol)
        
        # Adjust for time of day
        time_multiplier = self._get_time_multiplier(current_time.time())
        
        # Adjust for order size (larger orders have more slippage)
        size_multiplier = self._get_size_multiplier(qty, current_price)
        
        # Adjust for side (sells typically have slightly higher slippage)
        side_multiplier = 1.1 if side.lower() == "sell" else 1.0
        
        # Calculate total slippage
        estimated_slippage = base_slippage * time_multiplier * size_multiplier * side_multiplier
        
        # Cap at maximum
        estimated_slippage = min(estimated_slippage, self.max_slippage_bps)
        
        logger.debug(
            f"Slippage forecast for {symbol}: {estimated_slippage:.2f} bps "
            f"(base: {base_slippage:.2f}, time: {time_multiplier:.2f}x, "
            f"size: {size_multiplier:.2f}x, side: {side_multiplier:.2f}x)"
        )
        
        return estimated_slippage
    
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
    
    def update_symbol_stats(
        self,
        symbol: str,
        actual_slippage_bps: float,
        spread_bps: float,
        trade_time: datetime
    ) -> None:
        """
        Update symbol-specific slippage statistics from actual fills.
        
        Args:
            symbol: Stock symbol
            actual_slippage_bps: Actual observed slippage
            spread_bps: Actual observed spread
            trade_time: Time of trade
        """
        if symbol not in self.symbol_slippage:
            self.symbol_slippage[symbol] = {
                'avg_slippage_bps': actual_slippage_bps,
                'samples': 1,
                'last_updated': trade_time.isoformat()
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
        
        # Update time-of-day patterns
        self._update_time_pattern(symbol, spread_bps, trade_time)
        
        logger.debug(
            f"Updated slippage stats for {symbol}: "
            f"{self.symbol_slippage[symbol]['avg_slippage_bps']:.2f} bps "
            f"({self.symbol_slippage[symbol]['samples']} samples)"
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
        
        return {
            'symbols': len(self.symbol_slippage),
            'avg_slippage_bps': np.mean(avg_slippages),
            'median_slippage_bps': np.median(avg_slippages),
            'max_slippage_bps': np.max(avg_slippages),
            'min_slippage_bps': np.min(avg_slippages),
            'total_samples': sum(
                stats['samples']
                for stats in self.symbol_slippage.values()
            )
        }
