"""
Alpaca API client wrapper with error handling and reconnection logic.
"""
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError, TimeFrame
from loguru import logger
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
from src.models import Position, Bar


class AlpacaClient:
    """
    Wrapper for Alpaca Trading API with enhanced error handling.
    """
    
    def __init__(
        self,
        key_id: str,
        secret_key: str,
        base_url: str,
        data_feed: str = "iex",
        reconnect_attempts: int = 5,
        reconnect_delay: int = 10
    ):
        """
        Initialize Alpaca client.
        
        Args:
            key_id: API key ID
            secret_key: API secret key
            base_url: Base URL for API (paper or live)
            data_feed: Data feed type ("iex" or "sip")
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts (seconds)
        """
        self.key_id = key_id
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_feed = data_feed
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self._connect()
        logger.info(f"Alpaca client initialized: {base_url}")
    
    def _connect(self) -> None:
        """Establish connection to Alpaca API."""
        try:
            self.api = tradeapi.REST(
                key_id=self.key_id,
                secret_key=self.secret_key,
                base_url=self.base_url
            )
            # Test connection
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca API | Account: {account.account_number}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            raise
    
    def _retry_on_failure(self, func, *args, **kwargs):
        """
        Retry a function call on failure.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.reconnect_attempts):
            try:
                return func(*args, **kwargs)
            except APIError as e:
                logger.warning(f"API error (attempt {attempt + 1}/{self.reconnect_attempts}): {e}")
                if attempt < self.reconnect_attempts - 1:
                    time.sleep(self.reconnect_delay)
                    # Try to reconnect
                    try:
                        self._connect()
                    except Exception as conn_e:
                        logger.error(f"Reconnection failed: {conn_e}")
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
    
    def get_account(self) -> Any:
        """Get account information."""
        return self._retry_on_failure(self.api.get_account)
    
    def get_positions(self) -> List[Position]:
        """
        Get current positions.
        
        Returns:
            List of Position objects
        """
        try:
            alpaca_positions = self._retry_on_failure(self.api.list_positions)
            positions = []
            
            for pos in alpaca_positions:
                position = Position(
                    symbol=pos.symbol,
                    qty=int(pos.qty),
                    side="long" if int(pos.qty) > 0 else "short",
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    cost_basis=float(pos.cost_basis),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_plpc=float(pos.unrealized_plpc),
                    qty_available=int(pos.qty_available) if hasattr(pos, 'qty_available') else None
                )
                positions.append(position)
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position object or None if no position
        """
        try:
            pos = self._retry_on_failure(self.api.get_position, symbol)
            return Position(
                symbol=pos.symbol,
                qty=int(pos.qty),
                side="long" if int(pos.qty) > 0 else "short",
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                cost_basis=float(pos.cost_basis),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_plpc=float(pos.unrealized_plpc)
            )
        except APIError as e:
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Error getting position for {symbol}: {e}")
            return None
    
    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "limit",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        stop_loss: Optional[Dict[str, float]] = None,
        take_profit: Optional[Dict[str, float]] = None,
        client_order_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Place an order.
        
        Args:
            symbol: Stock symbol
            qty: Quantity
            side: "buy" or "sell"
            order_type: Order type ("market", "limit", "stop", "stop_limit")
            time_in_force: Time in force
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (for stop orders)
            stop_loss: Stop loss parameters
            take_profit: Take profit parameters
            client_order_id: Client-specified order ID
            
        Returns:
            Order object or None on failure
        """
        try:
            params = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }
            
            if limit_price is not None:
                params['limit_price'] = str(round(limit_price, 2))
            
            if stop_price is not None:
                params['stop_price'] = str(round(stop_price, 2))
            
            if client_order_id:
                params['client_order_id'] = client_order_id
            
            # Bracket orders
            if stop_loss or take_profit:
                params['order_class'] = 'bracket'
                if take_profit:
                    params['take_profit'] = {
                        'limit_price': str(round(take_profit.get('limit_price', 0), 2))
                    }
                if stop_loss:
                    sl_params = {'stop_price': str(round(stop_loss.get('stop_price', 0), 2))}
                    if 'limit_price' in stop_loss:
                        sl_params['limit_price'] = str(round(stop_loss['limit_price'], 2))
                    params['stop_loss'] = sl_params
            
            order = self._retry_on_failure(self.api.submit_order, **params)
            logger.info(
                f"Order placed | {side.upper()} {qty} {symbol} @ "
                f"{limit_price if limit_price else 'MARKET'} | Order ID: {order.id}"
            )
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._retry_on_failure(self.api.cancel_order, order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._retry_on_failure(self.api.cancel_all_orders)
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    def get_orders(self, status: str = "open", limit: int = 100) -> List[Any]:
        """
        Get orders.
        
        Args:
            status: Order status filter ("open", "closed", "all")
            limit: Maximum number of orders to return
            
        Returns:
            List of order objects
        """
        try:
            return self._retry_on_failure(
                self.api.list_orders,
                status=status,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000
    ) -> Optional[List[Bar]]:
        """
        Get historical bars as standardized Bar objects.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day", etc.)
            start: Start date (ISO format)
            end: End date (ISO format)
            limit: Maximum number of bars
            
        Returns:
            List of Bar objects or None on failure
        """
        try:
            # Map timeframe string to TimeFrame enum
            timeframe_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame.Minute * 5,
                '15Min': TimeFrame.Minute * 15,
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day
            }
            
            tf = timeframe_map.get(timeframe, TimeFrame.Minute)
            
            raw_bars = self._retry_on_failure(
                self.api.get_bars,
                symbol,
                tf,
                start=start,
                end=end,
                limit=limit
            )
            
            if not raw_bars:
                return None
            
            # Convert to standardized Bar objects
            bars = []
            for b in raw_bars:
                try:
                    bar = Bar(
                        symbol=getattr(b, 'S', symbol),
                        ts=getattr(b, 't', datetime.now()),
                        open=float(b.o),
                        high=float(b.h),
                        low=float(b.l),
                        close=float(b.c),
                        volume=int(b.v),
                        vwap=float(b.vw) if hasattr(b, 'vw') and b.vw is not None else None,
                        trade_count=int(b.n) if hasattr(b, 'n') and b.n is not None else None
                    )
                    bars.append(bar)
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Error converting bar for {symbol}: {e}")
                    continue
            
            return bars if bars else None
            
        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}")
            return None
    
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """
        Get the latest bar for a symbol as a standardized Bar object.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Bar object or None
        """
        try:
            bars = self.get_bars(symbol, "1Min", limit=1)
            return bars[0] if bars and len(bars) > 0 else None
        except Exception as e:
            logger.error(f"Error getting latest bar for {symbol}: {e}")
            return None
    
    def get_clock(self) -> Optional[Any]:
        """
        Get market clock information.
        
        Returns:
            Clock object with market status
        """
        try:
            return self._retry_on_failure(self.api.get_clock)
        except Exception as e:
            logger.error(f"Error getting clock: {e}")
            return None
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """
        try:
            clock = self.get_clock()
            return clock.is_open if clock else False
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def close_position(self, symbol: str, qty: Optional[int] = None) -> bool:
        """
        Close a position (or reduce by qty if specified).
        
        Args:
            symbol: Stock symbol
            qty: Quantity to close (None = close entire position)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if qty is None:
                # Close entire position
                self._retry_on_failure(self.api.close_position, symbol)
                logger.info(f"Position closed: {symbol}")
            else:
                # Get current position to determine side
                pos = self.get_position(symbol)
                if not pos:
                    logger.warning(f"No position to close for {symbol}")
                    return False
                
                side = "sell" if int(pos.qty) > 0 else "buy"
                self.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type="market"
                )
                logger.info(f"Reduced position: {symbol} by {qty}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """
        Close all open positions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._retry_on_failure(self.api.close_all_positions)
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False
    
    def is_shortable(self, symbol: str) -> bool:
        """
        Check if a symbol is shortable.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if shortable, False otherwise
        """
        try:
            asset = self._retry_on_failure(self.api.get_asset, symbol)
            is_shortable = getattr(asset, 'shortable', False)
            is_easy_to_borrow = getattr(asset, 'easy_to_borrow', False)
            
            # Symbol is shortable if both conditions are met
            return is_shortable and is_easy_to_borrow
        except Exception as e:
            logger.warning(f"Error checking shortability for {symbol}: {e}")
            return False  # Conservative: assume not shortable on error
    
    def get_asset(self, symbol: str) -> Optional[Any]:
        """
        Get asset information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Asset object or None on failure
        """
        try:
            return self._retry_on_failure(self.api.get_asset, symbol)
        except Exception as e:
            logger.error(f"Error getting asset info for {symbol}: {e}")
            return None
    
    def get_corporate_actions(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[List[Any]]:
        """
        Get corporate actions including earnings announcements for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of corporate action events or None on failure
            
        Note:
            This is a placeholder for earnings calendar integration.
            Alpaca's REST API v2 doesn't provide direct earnings calendar access.
            Consider integrating with external services like:
            - Alpha Vantage Earnings Calendar
            - Financial Modeling Prep
            - Earnings Whispers
            - Bloomberg/Reuters data feeds
        """
        try:
            # Placeholder implementation - returns empty list
            # TODO: Integrate with external earnings calendar service
            logger.debug(f"Corporate actions query for {symbol} (not implemented)")
            return []
        except Exception as e:
            logger.error(f"Error getting corporate actions for {symbol}: {e}")
            return None
    
    def has_upcoming_earnings(
        self,
        symbol: str,
        days_ahead: int = 7
    ) -> bool:
        """
        Check if symbol has earnings announcement within specified days.
        
        Args:
            symbol: Stock symbol
            days_ahead: Number of days to look ahead
            
        Returns:
            True if earnings within timeframe, False otherwise
            
        Note:
            This is a placeholder that returns False until integrated
            with an actual earnings calendar service.
        """
        try:
            # Placeholder - always returns False
            # TODO: Implement with actual earnings calendar service
            # Example integration points:
            # - Alpha Vantage: /query?function=EARNINGS_CALENDAR
            # - FMP: /v3/earnings_calendar/{symbol}
            # - Custom scraper from investor relations pages
            return False
        except Exception as e:
            logger.warning(f"Error checking earnings for {symbol}: {e}")
            return False
