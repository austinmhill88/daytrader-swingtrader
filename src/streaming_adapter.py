"""
Streaming data adapter for Polygon/Alpaca/IEX (Phase 5).
Provides faster bar aggregation and real-time quote/trade data.
"""
import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from loguru import logger
from collections import defaultdict
import pandas as pd

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets library not available - streaming disabled")


class StreamingDataAdapter:
    """
    Adapter for streaming market data from multiple providers.
    Phase 5 implementation for data breadth.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize streaming data adapter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        data_sources = config.get('alpaca', {}).get('data_sources', {})
        
        # Provider configuration
        self.primary_provider = data_sources.get('primary', 'alpaca')
        self.secondary_providers = data_sources.get('secondary', [])
        
        # API keys
        self.polygon_api_key = config.get('alpaca', {}).get('data_sources', {}).get('polygon', {}).get('api_key')
        self.alpaca_key_id = config.get('alpaca', {}).get('key_id')
        self.alpaca_secret = config.get('alpaca', {}).get('secret_key')
        self.iex_api_key = config.get('alpaca', {}).get('data_sources', {}).get('iex', {}).get('api_key')
        
        # Stream handlers
        self.quote_handlers: List[Callable] = []
        self.trade_handlers: List[Callable] = []
        self.bar_handlers: List[Callable] = []
        
        # Bar aggregation (for building bars from trades/quotes)
        self.bar_buffers: Dict[str, Dict] = defaultdict(lambda: {
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0,
            'trade_count': 0,
            'vwap_sum': 0.0,
            'start_time': None
        })
        self.bar_interval_seconds = 60  # 1-minute bars
        
        # Connection state
        self.is_connected = False
        self.connection_errors = 0
        self.last_message_time: Optional[datetime] = None
        
        logger.info(
            f"StreamingDataAdapter initialized | "
            f"Primary: {self.primary_provider}, "
            f"Secondary: {self.secondary_providers}"
        )
    
    def register_quote_handler(self, handler: Callable) -> None:
        """
        Register a callback for quote updates.
        
        Args:
            handler: Callable that accepts (symbol, quote_data)
        """
        self.quote_handlers.append(handler)
        logger.debug(f"Registered quote handler ({len(self.quote_handlers)} total)")
    
    def register_trade_handler(self, handler: Callable) -> None:
        """
        Register a callback for trade updates.
        
        Args:
            handler: Callable that accepts (symbol, trade_data)
        """
        self.trade_handlers.append(handler)
        logger.debug(f"Registered trade handler ({len(self.trade_handlers)} total)")
    
    def register_bar_handler(self, handler: Callable) -> None:
        """
        Register a callback for bar updates.
        
        Args:
            handler: Callable that accepts (symbol, bar_data)
        """
        self.bar_handlers.append(handler)
        logger.debug(f"Registered bar handler ({len(self.bar_handlers)} total)")
    
    async def connect_polygon(self, symbols: List[str]) -> None:
        """
        Connect to Polygon WebSocket stream.
        
        Args:
            symbols: List of symbols to subscribe
        """
        if not self.polygon_api_key:
            logger.warning("Polygon API key not configured")
            return
        
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not installed")
            return
        
        try:
            
            uri = f"wss://socket.polygon.io/stocks"
            
            async with websockets.connect(uri) as ws:
                # Authenticate
                await ws.send(json.dumps({
                    "action": "auth",
                    "params": self.polygon_api_key
                }))
                
                auth_msg = await ws.recv()
                logger.info(f"Polygon auth response: {auth_msg}")
                
                # Subscribe to quotes and trades
                subscribe_msg = {
                    "action": "subscribe",
                    "params": ",".join([f"Q.{s}" for s in symbols] + [f"T.{s}" for s in symbols])
                }
                await ws.send(json.dumps(subscribe_msg))
                
                logger.info(f"Subscribed to Polygon stream for {len(symbols)} symbols")
                self.is_connected = True
                
                # Process messages
                async for message in ws:
                    await self._handle_polygon_message(message)
                    
        except Exception as e:
            logger.error(f"Error connecting to Polygon: {e}")
            self.connection_errors += 1
            self.is_connected = False
    
    async def connect_alpaca(self, symbols: List[str]) -> None:
        """
        Connect to Alpaca WebSocket stream.
        
        Args:
            symbols: List of symbols to subscribe
        """
        if not self.alpaca_key_id or not self.alpaca_secret:
            logger.warning("Alpaca credentials not configured")
            return
        
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not installed")
            return
        
        try:
            
            uri = "wss://stream.data.alpaca.markets/v2/iex"
            
            async with websockets.connect(uri) as ws:
                # Authenticate
                await ws.send(json.dumps({
                    "action": "auth",
                    "key": self.alpaca_key_id,
                    "secret": self.alpaca_secret
                }))
                
                auth_msg = await ws.recv()
                logger.info(f"Alpaca auth response: {auth_msg}")
                
                # Subscribe to quotes and trades
                subscribe_msg = {
                    "action": "subscribe",
                    "quotes": symbols,
                    "trades": symbols
                }
                await ws.send(json.dumps(subscribe_msg))
                
                logger.info(f"Subscribed to Alpaca stream for {len(symbols)} symbols")
                self.is_connected = True
                
                # Process messages
                async for message in ws:
                    await self._handle_alpaca_message(message)
                    
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            self.connection_errors += 1
            self.is_connected = False
    
    async def _handle_polygon_message(self, message: str) -> None:
        """Process Polygon WebSocket message."""
        try:
            data = json.loads(message)
            
            for item in data:
                msg_type = item.get('ev')
                symbol = item.get('sym')
                
                if msg_type == 'Q':  # Quote
                    quote = {
                        'symbol': symbol,
                        'bid': item.get('bp'),
                        'ask': item.get('ap'),
                        'bid_size': item.get('bs'),
                        'ask_size': item.get('as'),
                        'timestamp': datetime.fromtimestamp(item.get('t') / 1000.0)
                    }
                    
                    # Call registered handlers
                    for handler in self.quote_handlers:
                        try:
                            handler(symbol, quote)
                        except Exception as e:
                            logger.error(f"Error in quote handler: {e}")
                
                elif msg_type == 'T':  # Trade
                    trade = {
                        'symbol': symbol,
                        'price': item.get('p'),
                        'size': item.get('s'),
                        'timestamp': datetime.fromtimestamp(item.get('t') / 1000.0),
                        'conditions': item.get('c', [])
                    }
                    
                    # Update bar aggregation
                    self._update_bar_buffer(symbol, trade)
                    
                    # Call registered handlers
                    for handler in self.trade_handlers:
                        try:
                            handler(symbol, trade)
                        except Exception as e:
                            logger.error(f"Error in trade handler: {e}")
            
            self.last_message_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error handling Polygon message: {e}")
    
    async def _handle_alpaca_message(self, message: str) -> None:
        """Process Alpaca WebSocket message."""
        try:
            data = json.loads(message)
            
            for item in data:
                msg_type = item.get('T')
                symbol = item.get('S')
                
                if msg_type == 'q':  # Quote
                    quote = {
                        'symbol': symbol,
                        'bid': item.get('bp'),
                        'ask': item.get('ap'),
                        'bid_size': item.get('bs'),
                        'ask_size': item.get('as'),
                        'timestamp': datetime.fromisoformat(item.get('t').replace('Z', '+00:00'))
                    }
                    
                    for handler in self.quote_handlers:
                        try:
                            handler(symbol, quote)
                        except Exception as e:
                            logger.error(f"Error in quote handler: {e}")
                
                elif msg_type == 't':  # Trade
                    trade = {
                        'symbol': symbol,
                        'price': item.get('p'),
                        'size': item.get('s'),
                        'timestamp': datetime.fromisoformat(item.get('t').replace('Z', '+00:00')),
                        'conditions': item.get('c', [])
                    }
                    
                    self._update_bar_buffer(symbol, trade)
                    
                    for handler in self.trade_handlers:
                        try:
                            handler(symbol, trade)
                        except Exception as e:
                            logger.error(f"Error in trade handler: {e}")
            
            self.last_message_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error handling Alpaca message: {e}")
    
    def _update_bar_buffer(self, symbol: str, trade: Dict) -> None:
        """
        Update bar aggregation buffer with trade data.
        
        Args:
            symbol: Stock symbol
            trade: Trade data
        """
        buffer = self.bar_buffers[symbol]
        price = trade['price']
        size = trade['size']
        timestamp = trade['timestamp']
        
        # Initialize or update bar
        if buffer['start_time'] is None:
            buffer['start_time'] = timestamp
            buffer['open'] = price
            buffer['high'] = price
            buffer['low'] = price
        
        # Update OHLC
        buffer['close'] = price
        buffer['high'] = max(buffer['high'], price) if buffer['high'] else price
        buffer['low'] = min(buffer['low'], price) if buffer['low'] else price
        buffer['volume'] += size
        buffer['trade_count'] += 1
        buffer['vwap_sum'] += price * size
        
        # Check if bar is complete
        elapsed = (timestamp - buffer['start_time']).total_seconds()
        if elapsed >= self.bar_interval_seconds:
            # Emit completed bar
            bar = {
                'symbol': symbol,
                'timestamp': buffer['start_time'],
                'open': buffer['open'],
                'high': buffer['high'],
                'low': buffer['low'],
                'close': buffer['close'],
                'volume': buffer['volume'],
                'trade_count': buffer['trade_count'],
                'vwap': buffer['vwap_sum'] / buffer['volume'] if buffer['volume'] > 0 else buffer['close']
            }
            
            # Call registered handlers
            for handler in self.bar_handlers:
                try:
                    handler(symbol, bar)
                except Exception as e:
                    logger.error(f"Error in bar handler: {e}")
            
            # Reset buffer
            self.bar_buffers[symbol] = {
                'open': None,
                'high': None,
                'low': None,
                'close': None,
                'volume': 0,
                'trade_count': 0,
                'vwap_sum': 0.0,
                'start_time': None
            }
    
    def get_connection_status(self) -> Dict:
        """
        Get connection status.
        
        Returns:
            Dictionary with connection info
        """
        return {
            'is_connected': self.is_connected,
            'connection_errors': self.connection_errors,
            'last_message': self.last_message_time.isoformat() if self.last_message_time else None,
            'registered_handlers': {
                'quotes': len(self.quote_handlers),
                'trades': len(self.trade_handlers),
                'bars': len(self.bar_handlers)
            },
            'active_buffers': len(self.bar_buffers)
        }
    
    async def start_stream(self, symbols: List[str]) -> None:
        """
        Start streaming data for symbols.
        
        Args:
            symbols: List of symbols to stream
        """
        logger.info(f"Starting stream for {len(symbols)} symbols")
        
        if self.primary_provider == 'polygon':
            await self.connect_polygon(symbols)
        elif self.primary_provider == 'alpaca':
            await self.connect_alpaca(symbols)
        else:
            logger.error(f"Unknown primary provider: {self.primary_provider}")
