"""
Multi-source data provider abstraction layer (Phase 1).
Supports Alpaca, Polygon, Tiingo, and IEX Cloud with fallback logic.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data provider.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 10000
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bars.
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe (1Min, 5Min, 1Hour, 1Day)
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars
            
        Returns:
            DataFrame with OHLCV data or None
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured."""
        pass


class AlpacaDataProvider(DataProvider):
    """Alpaca data provider."""
    
    def __init__(self, config: Dict[str, Any], alpaca_client):
        """
        Initialize Alpaca provider.
        
        Args:
            config: Alpaca configuration
            alpaca_client: AlpacaClient instance
        """
        super().__init__(config)
        self.client = alpaca_client
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 10000
    ) -> Optional[pd.DataFrame]:
        """Get bars from Alpaca."""
        try:
            bars = self.client.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=limit
            )
            
            if not bars:
                return None
            
            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.t,
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': int(bar.v),
                    'vwap': float(bar.vw) if hasattr(bar, 'vw') else None
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            logger.error(f"Alpaca data fetch error for {symbol}: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Alpaca is configured."""
        return self.client is not None


class PolygonDataProvider(DataProvider):
    """Polygon.io data provider (stub for Phase 1)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Polygon provider."""
        super().__init__(config)
        self.api_key = config.get('api_key', '')
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 10000
    ) -> Optional[pd.DataFrame]:
        """Get bars from Polygon - stub for future implementation."""
        logger.warning("Polygon provider not yet implemented")
        return None
    
    def is_available(self) -> bool:
        """Check if Polygon is configured."""
        return bool(self.api_key)


class TiingoDataProvider(DataProvider):
    """Tiingo data provider (stub for Phase 1)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Tiingo provider."""
        super().__init__(config)
        self.api_key = config.get('api_key', '')
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 10000
    ) -> Optional[pd.DataFrame]:
        """Get bars from Tiingo - stub for future implementation."""
        logger.warning("Tiingo provider not yet implemented")
        return None
    
    def is_available(self) -> bool:
        """Check if Tiingo is configured."""
        return bool(self.api_key)


class IEXDataProvider(DataProvider):
    """IEX Cloud data provider (stub for Phase 1)."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize IEX provider."""
        super().__init__(config)
        self.api_key = config.get('api_key', '')
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 10000
    ) -> Optional[pd.DataFrame]:
        """Get bars from IEX - stub for future implementation."""
        logger.warning("IEX provider not yet implemented")
        return None
    
    def is_available(self) -> bool:
        """Check if IEX is configured."""
        return bool(self.api_key)


class MultiSourceDataManager:
    """
    Manages multiple data sources with fallback logic.
    Phase 1 implementation with extensibility for Phase 2-4.
    """
    
    def __init__(self, config: Dict[str, Any], alpaca_client=None):
        """
        Initialize multi-source manager.
        
        Args:
            config: Full application configuration
            alpaca_client: AlpacaClient instance
        """
        self.config = config
        self.providers: Dict[str, DataProvider] = {}
        
        # Initialize providers
        data_config = config.get('data_sources', {})
        
        # Primary: Alpaca (always available if client provided)
        if alpaca_client:
            self.providers['alpaca'] = AlpacaDataProvider(
                config.get('alpaca', {}),
                alpaca_client
            )
        
        # Secondary sources (stubs for Phase 1, full implementation in Phase 2)
        if 'polygon' in data_config:
            self.providers['polygon'] = PolygonDataProvider(data_config['polygon'])
        
        if 'tiingo' in data_config:
            self.providers['tiingo'] = TiingoDataProvider(data_config['tiingo'])
        
        if 'iex' in data_config:
            self.providers['iex'] = IEXDataProvider(data_config['iex'])
        
        # Set primary and fallback order
        self.primary = data_config.get('primary', 'alpaca')
        self.fallback_order = data_config.get('secondary', [])
        
        logger.info(
            f"MultiSourceDataManager initialized | "
            f"Primary: {self.primary}, Fallback: {self.fallback_order}"
        )
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 10000
    ) -> Optional[pd.DataFrame]:
        """
        Get bars with automatic fallback to secondary sources.
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars
            
        Returns:
            DataFrame with OHLCV data or None
        """
        # Try primary source
        if self.primary in self.providers:
            provider = self.providers[self.primary]
            if provider.is_available():
                logger.debug(f"Fetching {symbol} from primary: {self.primary}")
                df = provider.get_bars(symbol, timeframe, start, end, limit)
                if df is not None and len(df) > 0:
                    logger.debug(f"Retrieved {len(df)} bars from {self.primary}")
                    return df
                else:
                    logger.warning(f"Primary source {self.primary} returned no data")
        
        # Try fallback sources
        for fallback in self.fallback_order:
            if fallback in self.providers:
                provider = self.providers[fallback]
                if provider.is_available():
                    logger.info(f"Trying fallback source: {fallback}")
                    df = provider.get_bars(symbol, timeframe, start, end, limit)
                    if df is not None and len(df) > 0:
                        logger.info(f"Retrieved {len(df)} bars from {fallback}")
                        return df
        
        logger.error(f"All data sources failed for {symbol}")
        return None
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available and configured providers.
        
        Returns:
            List of provider names
        """
        return [
            name for name, provider in self.providers.items()
            if provider.is_available()
        ]
