"""
Base strategy interface for all trading strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from loguru import logger

from src.models import Signal, Bar


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    name: str = "base_strategy"
    
    def __init__(self, config: Dict):
        """
        Initialize strategy.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.state: Dict[str, pd.DataFrame] = {}  # symbol -> DataFrame with indicators
        self.last_signals: Dict[str, Signal] = {}  # symbol -> last signal
        self.is_warmed_up = False
        
        logger.info(f"Strategy '{self.name}' initialized | Enabled: {self.enabled}")
    
    @abstractmethod
    def warmup(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Warm up strategy with historical data.
        
        Args:
            historical_data: Dictionary mapping symbol to DataFrame
        """
        pass
    
    @abstractmethod
    def on_bar(self, bar: Bar) -> List[Signal]:
        """
        Process a new bar and generate signals.
        
        Args:
            bar: New bar data
            
        Returns:
            List of signals (can be empty)
        """
        pass
    
    @abstractmethod
    def on_end_of_day(self) -> None:
        """
        Called at end of trading day for cleanup/finalization.
        """
        pass
    
    def is_enabled(self) -> bool:
        """
        Check if strategy is enabled.
        
        Returns:
            True if enabled
        """
        return self.enabled
    
    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
        logger.info(f"Strategy '{self.name}' enabled")
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
        logger.info(f"Strategy '{self.name}' disabled")
    
    def get_last_signal(self, symbol: str) -> Optional[Signal]:
        """
        Get the last signal generated for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Last signal or None
        """
        return self.last_signals.get(symbol)
    
    def _update_state(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Update internal state for a symbol.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with updated data
        """
        # Keep a reasonable window to avoid memory issues
        max_rows = self.config.get('max_state_rows', 1000)
        if len(df) > max_rows:
            df = df.iloc[-max_rows:]
        
        self.state[symbol] = df
    
    def _record_signal(self, signal: Signal) -> None:
        """
        Record a signal.
        
        Args:
            signal: Signal to record
        """
        self.last_signals[signal.symbol] = signal
    
    def get_state_summary(self) -> str:
        """
        Get a summary of strategy state.
        
        Returns:
            Summary string
        """
        return (
            f"Strategy: {self.name} | "
            f"Enabled: {self.enabled} | "
            f"Warmed up: {self.is_warmed_up} | "
            f"Tracking: {len(self.state)} symbols"
        )
