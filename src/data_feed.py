"""
Data feed classes for live streaming and historical data.
"""
from alpaca_trade_api.stream import Stream
from loguru import logger
from typing import Callable, List, Dict, Optional
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from src.models import Bar


class LiveDataFeed:
    """
    Live streaming data feed using Alpaca's WebSocket API.
    """
    
    def __init__(
        self,
        key_id: str,
        secret_key: str,
        base_url: str,
        symbols: List[str],
        data_feed: str = "iex"
    ):
        """
        Initialize live data feed.
        
        Args:
            key_id: API key ID
            secret_key: API secret key
            base_url: Base URL
            symbols: List of symbols to subscribe to
            data_feed: Data feed type ("iex" or "sip")
        """
        self.key_id = key_id
        self.secret_key = secret_key
        self.base_url = base_url
        self.symbols = symbols
        self.data_feed = data_feed
        self.callbacks: List[Callable] = []
        self.is_running = False
        self.reconnect_count = 0
        self.max_reconnects = 10
        
        logger.info(f"LiveDataFeed initialized for {len(symbols)} symbols")
    
    def subscribe_bars(self, callback: Callable) -> None:
        """
        Subscribe to bar updates.
        
        Args:
            callback: Function to call when bar received
        """
        self.callbacks.append(callback)
        logger.info("Bar callback registered")
    
    async def _on_bar(self, bar) -> None:
        """
        Internal handler for bar events.
        
        Args:
            bar: Bar object from Alpaca
        """
        try:
            # Convert to our Bar model
            bar_data = Bar(
                symbol=bar.symbol,
                ts=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
                vwap=float(bar.vwap) if hasattr(bar, 'vwap') else None,
                trade_count=int(bar.trade_count) if hasattr(bar, 'trade_count') else None
            )
            
            # Call all registered callbacks
            for callback in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(bar_data)
                    else:
                        callback(bar_data)
                except Exception as e:
                    logger.error(f"Error in bar callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing bar: {e}")
    
    async def _handle_connection_error(self, error: Exception) -> None:
        """
        Handle connection errors with reconnection logic.
        
        Args:
            error: Exception that occurred
        """
        logger.error(f"Data stream error: {error}")
        self.reconnect_count += 1
        
        if self.reconnect_count < self.max_reconnects:
            wait_time = min(60, 2 ** self.reconnect_count)
            logger.info(f"Attempting reconnection #{self.reconnect_count} in {wait_time}s")
            await asyncio.sleep(wait_time)
            # Connection will be re-established by run() loop
        else:
            logger.critical("Max reconnection attempts reached. Manual intervention required.")
            self.is_running = False
    
    async def run_async(self) -> None:
        """
        Run the data stream asynchronously.
        """
        self.is_running = True
        
        while self.is_running and self.reconnect_count < self.max_reconnects:
            try:
                stream = Stream(
                    self.key_id,
                    self.secret_key,
                    base_url=self.base_url,
                    data_feed=self.data_feed
                )
                
                # Subscribe to bars for all symbols
                for symbol in self.symbols:
                    stream.subscribe_bars(self._on_bar, symbol)
                
                logger.info(f"Starting data stream for {len(self.symbols)} symbols")
                await stream._run_forever()
                
                # If we reach here, connection was closed
                if self.is_running:
                    await self._handle_connection_error(Exception("Stream closed unexpectedly"))
                    
            except Exception as e:
                if self.is_running:
                    await self._handle_connection_error(e)
                else:
                    break
        
        logger.info("Data stream stopped")
    
    def run(self) -> None:
        """
        Run the data stream (blocking).
        """
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            logger.info("Data stream interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Fatal error in data stream: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the data stream."""
        self.is_running = False
        logger.info("Data stream stop requested")


class HistoricalDataFeed:
    """
    Historical data feed for backtesting and warmup.
    """
    
    def __init__(self, alpaca_client):
        """
        Initialize historical data feed.
        
        Args:
            alpaca_client: AlpacaClient instance
        """
        self.client = alpaca_client
        logger.info("HistoricalDataFeed initialized")
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bars as DataFrame with standardized schema.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars
            
        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            # Format dates for API
            start_str = start.isoformat() if start else None
            end_str = end.isoformat() if end else None
            
            # Get bars as standardized Bar objects
            bars = self.client.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start_str,
                end=end_str,
                limit=limit
            )
            
            if not bars:
                logger.warning(f"No bars returned for {symbol}")
                return None
            
            # Convert Bar objects to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'symbol': bar.symbol,
                    'ts': bar.ts,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap,
                    'trade_count': bar.trade_count
                })
            
            df = pd.DataFrame(data)
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.set_index('ts')
            
            logger.info(f"Retrieved {len(df)} bars for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical bars for {symbol}: {e}")
            return None
    
    def get_multi_symbol_bars(
        self,
        symbols: List[str],
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical bars for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            timeframe: Timeframe
            start: Start datetime
            end: End datetime
            limit: Maximum bars per symbol
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}
        
        for symbol in symbols:
            df = self.get_bars(symbol, timeframe, start, end, limit)
            if df is not None:
                result[symbol] = df
        
        logger.info(f"Retrieved data for {len(result)}/{len(symbols)} symbols")
        return result
    
    def get_warmup_data(
        self,
        symbols: List[str],
        timeframe: str,
        lookback_days: int = 60
    ) -> Dict[str, pd.DataFrame]:
        """
        Get warmup data for strategies.
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        
        logger.info(f"Fetching {lookback_days} days of warmup data for {len(symbols)} symbols")
        return self.get_multi_symbol_bars(symbols, timeframe, start, end)
