"""
Order execution engine with position sizing and bracket order support.
"""
import math
import time
from datetime import datetime
from typing import List, Optional, Dict
from loguru import logger

from src.models import OrderIntent, Signal, OrderSide, OrderType
from src.alpaca_client import AlpacaClient
from src.portfolio import PortfolioState
from src.logging_utils import log_trade


class ExecutionEngine:
    """
    Handles order execution with smart position sizing and bracket orders.
    """
    
    def __init__(self, client: AlpacaClient, config: Dict):
        """
        Initialize execution engine.
        
        Args:
            client: AlpacaClient instance
            config: Execution configuration
        """
        self.client = client
        self.config = config
        
        # Configuration
        self.default_order_type = config.get('default_order_type', 'limit')
        self.limit_offset_bps = config.get('limit_offset_bps', 5)
        self.use_bracket_orders = config.get('use_bracket_orders', True)
        self.time_in_force = config.get('time_in_force', 'day')
        self.partial_fill_threshold = config.get('partial_fill_threshold', 0.8)
        self.order_timeout_seconds = config.get('order_timeout_seconds', 60)
        self.max_slippage_bps = config.get('max_slippage_bps', 20)
        
        # Time-slicing configuration
        self.enable_time_slicing = config.get('enable_time_slicing', False)
        self.time_slice_size_threshold_usd = config.get('time_slice_size_threshold_usd', 10000)
        self.max_child_order_pct = config.get('max_child_order_pct', 20)
        
        # Rate limiting
        self.api_call_count = 0
        self.api_call_window_start = None
        self.max_api_calls_per_minute = config.get('max_api_calls_per_minute', 200)
        
        # Toxic time window configuration
        self.enable_toxic_time_filtering = config.get('enable_toxic_time_filtering', True)
        self.market_open_blackout_minutes = config.get('market_open_blackout_minutes', 5)
        self.market_close_blackout_minutes = config.get('market_close_blackout_minutes', 5)
        
        # Rejection tracking
        from collections import deque
        from datetime import datetime, timedelta
        self.recent_rejections = deque(maxlen=100)  # Track last 100 rejections
        self.rejection_alert_threshold_pct = 5.0  # Alert if > 5% rejection rate
        self.rejection_lookback_minutes = 60  # Lookback window
        
        # Prometheus exporter reference (will be set externally)
        self.prometheus = None
        
        # Alert notifier reference (will be set externally)
        self.notifier = None
        
        # Order reconciliation tracking
        self.pending_orders: Dict[str, Dict] = {}  # order_id -> {intent, timestamp, status}
        self.order_check_interval = config.get('order_check_interval_seconds', 30)
        self.stale_order_timeout = config.get('stale_order_timeout_seconds', 300)  # 5 minutes
        
        logger.info(
            f"ExecutionEngine initialized | "
            f"Order type: {self.default_order_type}, "
            f"Bracket orders: {self.use_bracket_orders}, "
            f"Time slicing: {self.enable_time_slicing}, "
            f"Toxic time filtering: {self.enable_toxic_time_filtering}, "
            f"Rate limit: {self.max_api_calls_per_minute} calls/min"
        )
    
    def calculate_position_size(
        self,
        signal: Signal,
        equity: float,
        price: float,
        atr: float,
        per_trade_risk_pct: float
    ) -> int:
        """
        Calculate position size based on risk parameters.
        
        Args:
            signal: Trading signal
            equity: Current equity
            price: Current price
            atr: Average True Range
            per_trade_risk_pct: Risk per trade as percentage of equity
            
        Returns:
            Position size in shares
        """
        if equity <= 0 or price <= 0 or atr <= 0:
            return 0
        
        # Calculate risk amount in dollars
        risk_amount = equity * (per_trade_risk_pct / 100.0)
        
        # Calculate stop distance (using ATR or percentage of price)
        stop_distance = max(atr, price * 0.005)  # Minimum 0.5% stop
        
        # Position size = risk amount / stop distance
        qty = risk_amount / stop_distance
        
        # Adjust by signal confidence (reduce size for lower confidence)
        qty = qty * signal.confidence
        
        # Adjust by signal strength (full size for full strength)
        qty = qty * abs(signal.strength)
        
        # Round down to whole shares
        qty = math.floor(qty)
        
        # Ensure at least 1 share if we got this far
        qty = max(qty, 1)
        
        # Check if we can afford it (leave buffer for safety)
        cash_buffer_pct = self.config.get('cash_buffer_pct', 0.95)  # Default 5% buffer
        max_affordable = math.floor((equity * cash_buffer_pct) / price)
        qty = min(qty, max_affordable)
        
        logger.debug(
            f"Position sizing | {signal.symbol} | "
            f"Risk: ${risk_amount:.2f}, Stop: ${stop_distance:.2f}, "
            f"Qty: {qty}, Price: ${price:.2f}, Total: ${qty * price:.2f}"
        )
        
        return qty
    
    def create_order_intent(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        current_price: float,
        atr: float,
        per_trade_risk_pct: float
    ) -> Optional[OrderIntent]:
        """
        Create an order intent from a signal.
        
        Args:
            signal: Trading signal
            portfolio: PortfolioState
            current_price: Current market price
            atr: Average True Range
            per_trade_risk_pct: Risk percentage per trade
            
        Returns:
            OrderIntent or None
        """
        # Calculate position size
        equity = portfolio.equity()
        qty = self.calculate_position_size(
            signal, equity, current_price, atr, per_trade_risk_pct
        )
        
        if qty <= 0:
            logger.warning(f"Position size calculation resulted in 0 shares for {signal.symbol}")
            return None
        
        # Determine order side
        side = OrderSide.BUY if signal.is_buy_signal() else OrderSide.SELL
        
        # Calculate limit price with offset
        if self.default_order_type == "limit":
            offset_multiplier = 1 + (self.limit_offset_bps / 10000)
            if side == OrderSide.BUY:
                # Buy slightly above current price for better fills
                limit_price = current_price * offset_multiplier
            else:
                # Sell slightly below current price for better fills
                limit_price = current_price / offset_multiplier
            
            limit_price = round(limit_price, 2)
        else:
            limit_price = None
        
        # Calculate bracket orders (stop loss and take profit)
        bracket = None
        if self.use_bracket_orders:
            bracket = self._calculate_bracket_prices(
                side, current_price, atr, signal.metadata or {}
            )
        
        intent = OrderIntent(
            symbol=signal.symbol,
            side=side,
            qty=qty,
            order_type=OrderType.LIMIT if self.default_order_type == "limit" else OrderType.MARKET,
            limit_price=limit_price,
            time_in_force=self.time_in_force,
            bracket=bracket,
            strategy_name=signal.strategy_name,
            reason=signal.reason
        )
        
        # Format price string based on order type
        price_str = f"${limit_price:.2f}" if limit_price is not None else "MARKET"
        logger.info(
            f"Order intent created | {intent.side.value.upper()} {intent.qty} {intent.symbol} "
            f"@ {price_str} | Strategy: {signal.strategy_name}"
        )
        
        return intent
    
    def _calculate_bracket_prices(
        self,
        side: OrderSide,
        entry_price: float,
        atr: float,
        metadata: Dict
    ) -> Dict[str, float]:
        """
        Calculate stop loss and take profit prices.
        
        Args:
            side: Order side
            entry_price: Entry price
            atr: Average True Range
            metadata: Signal metadata (may contain stop multipliers)
            
        Returns:
            Dictionary with stop_loss and take_profit prices
        """
        # Get multipliers from metadata or use defaults
        stop_multiplier = metadata.get('atr_stop_multiplier', 2.0)
        tp_multiplier = metadata.get('atr_tp_multiplier', 3.0)
        
        if side == OrderSide.BUY:
            # Long position
            stop_loss = entry_price - (atr * stop_multiplier)
            take_profit = entry_price + (atr * tp_multiplier)
        else:
            # Short position
            stop_loss = entry_price + (atr * stop_multiplier)
            take_profit = entry_price - (atr * tp_multiplier)
        
        bracket = {
            'stop_loss': round(max(stop_loss, 0.01), 2),
            'take_profit': round(max(take_profit, 0.01), 2)
        }
        
        logger.debug(
            f"Bracket calculated | Entry: ${entry_price:.2f}, "
            f"Stop: ${bracket['stop_loss']:.2f}, "
            f"TP: ${bracket['take_profit']:.2f}"
        )
        
        return bracket
    
    def _calculate_adaptive_limit_price(
        self,
        side: OrderSide,
        current_price: float,
        spread: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate adaptive limit price based on spread and volatility.
        
        Args:
            side: Order side
            current_price: Current market price
            spread: Bid-ask spread (if available)
            volatility: Recent volatility (if available)
            
        Returns:
            Adaptive limit price
        """
        # Base offset
        offset_bps = self.limit_offset_bps
        
        # Adjust offset based on spread (if provided)
        if spread is not None and spread > 0:
            # Wider spread requires more aggressive pricing
            spread_bps = (spread / current_price) * 10000
            offset_bps = max(offset_bps, spread_bps * 0.5)  # Use 50% of spread
        
        # Adjust offset based on volatility (if provided)
        if volatility is not None and volatility > 0:
            # Higher volatility requires more buffer
            vol_adjustment = min(volatility * 100, 10)  # Cap at 10 bps adjustment
            offset_bps += vol_adjustment
        
        # Cap maximum offset at reasonable level
        offset_bps = min(offset_bps, 20)  # Max 20 bps offset
        
        offset_multiplier = 1 + (offset_bps / 10000)
        
        if side == OrderSide.BUY:
            limit_price = current_price * offset_multiplier
        else:
            limit_price = current_price / offset_multiplier
        
        return round(limit_price, 2)
    
    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits for API calls.
        Implements token bucket-style rate limiting.
        
        Returns:
            True if we can make the call, False if rate limited
        """
        
        now = datetime.now()
        
        # Reset counter if new window
        if self.api_call_window_start is None or \
           (now - self.api_call_window_start).total_seconds() >= 60:
            self.api_call_window_start = now
            self.api_call_count = 0
        
        # Check if under limit
        if self.api_call_count < self.max_api_calls_per_minute:
            self.api_call_count += 1
            return True
        
        # Rate limited
        logger.warning(
            f"Rate limit reached: {self.api_call_count} calls in last minute"
        )
        return False
    
    def _wait_for_rate_limit(self, max_wait_seconds: int = 30) -> bool:
        """
        Wait until rate limit resets.
        
        Args:
            max_wait_seconds: Maximum time to wait
            
        Returns:
            True if wait successful, False if timed out
        """
        
        if self.api_call_window_start is None:
            return True
        
        elapsed = (datetime.now() - self.api_call_window_start).total_seconds()
        wait_time = max(0, 60 - elapsed)
        
        if wait_time > max_wait_seconds:
            logger.warning(
                f"Rate limit wait time ({wait_time:.1f}s) exceeds max ({max_wait_seconds}s)"
            )
            return False
        
        if wait_time > 0:
            logger.info(f"Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        return True
    
    def _split_order_for_time_slicing(
        self,
        intent: OrderIntent
    ) -> List[OrderIntent]:
        """
        Split a large order into smaller child orders for time-slicing.
        
        Args:
            intent: Original order intent
            
        Returns:
            List of child order intents
        """
        if not self.enable_time_slicing:
            return [intent]
        
        # Use limit price if available, otherwise cannot time-slice market orders
        if intent.limit_price is None or intent.limit_price <= 0:
            logger.warning(
                f"Cannot time-slice order for {intent.symbol}: no limit price available"
            )
            return [intent]
        
        order_value = intent.qty * intent.limit_price
        
        # Check if order exceeds threshold
        if order_value < self.time_slice_size_threshold_usd:
            return [intent]
        
        # Calculate child order size (as % of parent)
        child_pct = self.max_child_order_pct / 100.0
        child_qty = max(1, int(intent.qty * child_pct))
        
        # Create child orders
        child_orders = []
        remaining_qty = intent.qty
        
        while remaining_qty > 0:
            slice_qty = min(child_qty, remaining_qty)
            
            child_intent = OrderIntent(
                symbol=intent.symbol,
                side=intent.side,
                qty=slice_qty,
                order_type=intent.order_type,
                limit_price=intent.limit_price,
                time_in_force=intent.time_in_force,
                bracket=None,  # Don't use brackets on child orders
                strategy_name=intent.strategy_name,
                reason=f"{intent.reason} (slice {len(child_orders)+1})"
            )
            
            child_orders.append(child_intent)
            remaining_qty -= slice_qty
        
        logger.info(
            f"Order time-sliced | {intent.symbol} | "
            f"{intent.qty} shares split into {len(child_orders)} orders"
        )
        
        return child_orders
    
    def execute_intents(self, intents: List[OrderIntent]) -> List[Optional[any]]:
        """
        Execute a list of order intents with time-slicing support.
        
        Args:
            intents: List of OrderIntent objects
            
        Returns:
            List of order results (or None for failures)
        """
        results = []
        
        for intent in intents:
            try:
                # Check if order should be time-sliced
                child_orders = self._split_order_for_time_slicing(intent)
                
                # Execute each child order
                for child_intent in child_orders:
                    result = self.execute_single_intent(child_intent)
                    results.append(result)
                    
                    # Small delay between child orders
                    if len(child_orders) > 1:
                        time.sleep(1)
                        
            except Exception as e:
                logger.error(f"Error executing intent for {intent.symbol}: {e}")
                results.append(None)
        
        return results
    
    def _is_toxic_time_window(self) -> bool:
        """
        Check if current time is within toxic trading windows.
        Toxic windows: first N minutes after open, last N minutes before close.
        
        Returns:
            True if in toxic window, False otherwise
        """
        if not self.enable_toxic_time_filtering:
            return False
        
        try:
            clock = self.client.get_clock()
            if not clock or not clock.is_open:
                return True  # Market closed is toxic
            
            from datetime import datetime, timezone
            import dateutil.parser
            
            # Parse times
            now = datetime.now(timezone.utc)
            market_open = dateutil.parser.parse(clock.next_open) if hasattr(clock, 'next_open') else None
            market_close = dateutil.parser.parse(clock.next_close) if hasattr(clock, 'next_close') else None
            
            if not market_open or not market_close:
                return False  # Can't determine, allow trading
            
            # Calculate minutes from open/close
            minutes_from_open = (now - market_open).total_seconds() / 60
            minutes_to_close = (market_close - now).total_seconds() / 60
            
            # Check if in blackout window
            if minutes_from_open < self.market_open_blackout_minutes:
                logger.debug(f"Toxic window: {minutes_from_open:.1f}min from open")
                return True
            
            if minutes_to_close < self.market_close_blackout_minutes:
                logger.debug(f"Toxic window: {minutes_to_close:.1f}min to close")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking toxic time window: {e}")
            return False  # Allow trading on error
    
    def execute_single_intent(self, intent: OrderIntent) -> Optional[any]:
        """
        Execute a single order intent with rate limiting, toxic time filtering, and fill logging.
        
        Args:
            intent: OrderIntent to execute
            
        Returns:
            Order object or None on failure
        """
        start_time = time.time()
        
        try:
            # Check toxic time window
            if self._is_toxic_time_window():
                logger.warning(f"Order blocked for {intent.symbol}: toxic time window")
                if self.prometheus:
                    self.prometheus.record_order_rejected("toxic_time")
                self._track_rejection("toxic_time")
                return None
            
            # Check rate limit
            if not self._check_rate_limit():
                # Wait for rate limit to reset
                if not self._wait_for_rate_limit(max_wait_seconds=10):
                    logger.error(f"Rate limit exceeded, cannot place order for {intent.symbol}")
                    if self.prometheus:
                        self.prometheus.record_order_rejected("rate_limit")
                    self._track_rejection("rate_limit")
                    return None
            
            # Prepare bracket order parameters
            stop_loss = None
            take_profit = None
            
            if intent.bracket:
                stop_loss = {
                    'stop_price': intent.bracket.get('stop_loss')
                }
                take_profit = {
                    'limit_price': intent.bracket.get('take_profit')
                }
            
            # Place order with exponential backoff retry
            order = self._place_order_with_retry(
                symbol=intent.symbol,
                qty=intent.qty,
                side=intent.side.value,
                order_type=intent.order_type.value,
                time_in_force=intent.time_in_force.value,
                limit_price=intent.limit_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            if self.prometheus:
                self.prometheus.record_order_latency(latency_ms)
            
            if order:
                # Log trade placement
                log_trade(
                    action="ORDER_PLACED",
                    symbol=intent.symbol,
                    qty=intent.qty,
                    price=intent.limit_price or 0,
                    side=intent.side.value,
                    strategy=intent.strategy_name,
                    reason=intent.reason,
                    order_id=order.id
                )
                
                # Record metrics
                if self.prometheus:
                    self.prometheus.record_order_placed(
                        strategy=intent.strategy_name or "unknown",
                        side=intent.side.value
                    )
                
                # Log bracket order details if present
                if intent.bracket:
                    logger.info(
                        f"Bracket order | {intent.symbol} | "
                        f"Stop: ${intent.bracket.get('stop_loss'):.2f}, "
                        f"TP: ${intent.bracket.get('take_profit'):.2f}"
                    )
                
                # Track order for reconciliation
                self.track_order(order.id, intent)
                
                return order
            else:
                logger.error(f"Order placement failed for {intent.symbol}")
                if self.prometheus:
                    self.prometheus.record_order_rejected("placement_failed")
                self._track_rejection("placement_failed")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order for {intent.symbol}: {e}")
            if self.prometheus:
                self.prometheus.record_order_rejected("exception")
            self._track_rejection("exception")
            return None
    
    def _track_rejection(self, reason: str) -> None:
        """
        Track order rejection and check if alert threshold reached.
        
        Args:
            reason: Reason for rejection
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        self.recent_rejections.append((now, reason))
        
        # Check rejection rate
        cutoff = now - timedelta(minutes=self.rejection_lookback_minutes)
        recent_count = sum(1 for ts, _ in self.recent_rejections if ts > cutoff)
        
        # Estimate total orders (rough approximation based on rejection tracking)
        # This is simplified - in production you'd track all order attempts
        estimated_total = max(recent_count * 20, 100)  # Assume rejections are ~5% normally
        rejection_pct = (recent_count / estimated_total) * 100
        
        # Send alert if threshold exceeded
        if rejection_pct >= self.rejection_alert_threshold_pct and self.notifier:
            self.notifier.send_high_rejection_rate_alert(
                rejection_pct=rejection_pct,
                lookback_minutes=self.rejection_lookback_minutes
            )
    
    def _place_order_with_retry(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str,
        time_in_force: str,
        limit_price: Optional[float] = None,
        stop_loss: Optional[Dict] = None,
        take_profit: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Optional[any]:
        """
        Place order with exponential backoff retry.
        
        Args:
            symbol: Stock symbol
            qty: Quantity
            side: Order side
            order_type: Order type
            time_in_force: Time in force
            limit_price: Limit price
            stop_loss: Stop loss params
            take_profit: Take profit params
            max_retries: Maximum retry attempts
            
        Returns:
            Order object or None
        """
        
        for attempt in range(max_retries):
            try:
                order = self.client.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                return order
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Order placement failed (attempt {attempt+1}/{max_retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Order placement failed after {max_retries} attempts: {e}")
                    raise
        
        return None
    
    def close_position(
        self,
        symbol: str,
        qty: Optional[int] = None,
        reason: str = "Manual close"
    ) -> bool:
        """
        Close a position.
        
        Args:
            symbol: Stock symbol
            qty: Quantity to close (None = all)
            reason: Reason for closing
            
        Returns:
            True if successful
        """
        try:
            result = self.client.close_position(symbol, qty)
            
            if result:
                log_trade(
                    action="POSITION_CLOSED",
                    symbol=symbol,
                    qty=qty or 0,
                    price=0,
                    reason=reason
                )
            
            return result
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    def close_all_positions(self, reason: str = "Close all") -> bool:
        """
        Close all positions.
        
        Args:
            reason: Reason for closing
            
        Returns:
            True if successful
        """
        try:
            result = self.client.close_all_positions()
            
            if result:
                logger.info(f"All positions closed | Reason: {reason}")
            
            return result
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False
    
    def cancel_all_orders(self, reason: str = "Cancel all") -> bool:
        """
        Cancel all open orders.
        
        Args:
            reason: Reason for cancelling
            
        Returns:
            True if successful
        """
        try:
            result = self.client.cancel_all_orders()
            
            if result:
                logger.info(f"All orders cancelled | Reason: {reason}")
            
            return result
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False
    
    def track_order(self, order_id: str, intent: OrderIntent) -> None:
        """
        Track an order for reconciliation.
        
        Args:
            order_id: Order ID from broker
            intent: Original order intent
        """
        self.pending_orders[order_id] = {
            'intent': intent,
            'timestamp': datetime.now(),
            'status': 'pending'
        }
        logger.debug(f"Tracking order {order_id} for {intent.symbol}")
    
    def reconcile_orders(self) -> None:
        """
        Reconcile pending orders - check status and handle stale orders.
        Should be called periodically (e.g., every 30 seconds).
        """
        if not self.pending_orders:
            return
        
        logger.debug(f"Reconciling {len(self.pending_orders)} pending orders")
        
        now = datetime.now()
        stale_orders = []
        
        for order_id, order_data in list(self.pending_orders.items()):
            try:
                # Check if order is stale
                age = (now - order_data['timestamp']).total_seconds()
                
                if age > self.stale_order_timeout:
                    stale_orders.append(order_id)
                    continue
                
                # Query order status from broker
                orders = self.client.get_orders(status='all', limit=500)
                order = next((o for o in orders if o.id == order_id), None)
                
                if not order:
                    logger.warning(f"Order {order_id} not found in broker records")
                    stale_orders.append(order_id)
                    continue
                
                # Update status
                status = getattr(order, 'status', 'unknown')
                
                if status in ['filled', 'canceled', 'expired', 'rejected']:
                    # Order is complete, remove from tracking
                    logger.info(
                        f"Order {order_id} completed with status: {status} | "
                        f"{order_data['intent'].symbol}"
                    )
                    del self.pending_orders[order_id]
                    
                    if status == 'filled':
                        # Log successful fill
                        log_trade(
                            action="ORDER_FILLED",
                            symbol=order_data['intent'].symbol,
                            qty=order_data['intent'].qty,
                            price=float(getattr(order, 'filled_avg_price', 0)),
                            side=order_data['intent'].side.value,
                            strategy=order_data['intent'].strategy_name,
                            reason=order_data['intent'].reason,
                            order_id=order_id
                        )
                        
                        if self.prometheus:
                            self.prometheus.record_order_filled(
                                strategy=order_data['intent'].strategy_name or "unknown",
                                side=order_data['intent'].side.value
                            )
                
            except Exception as e:
                logger.error(f"Error reconciling order {order_id}: {e}")
        
        # Handle stale orders
        for order_id in stale_orders:
            self._handle_stale_order(order_id)
    
    def _handle_stale_order(self, order_id: str) -> None:
        """
        Handle a stale order - cancel and optionally retry.
        
        Args:
            order_id: Stale order ID
        """
        try:
            order_data = self.pending_orders.get(order_id)
            if not order_data:
                return
            
            intent = order_data['intent']
            age = (datetime.now() - order_data['timestamp']).total_seconds()
            
            logger.warning(
                f"Stale order detected | {order_id} | {intent.symbol} | "
                f"Age: {age:.0f}s"
            )
            
            # Cancel the stale order
            self.client.cancel_order(order_id)
            
            # Remove from tracking
            del self.pending_orders[order_id]
            
            # Alert about stale order
            if self.notifier:
                self.notifier.send_alert(
                    title="Stale Order Detected",
                    message=f"Order {order_id} for {intent.symbol} was stale ({age:.0f}s) and cancelled",
                    severity="warning",
                    metadata={
                        'order_id': order_id,
                        'symbol': intent.symbol,
                        'age_seconds': int(age)
                    }
                )
            
            # TODO: Implement retry logic based on strategy
            # For now, just log and alert
            logger.info(f"Stale order {order_id} cancelled (retry not implemented)")
            
        except Exception as e:
            logger.error(f"Error handling stale order {order_id}: {e}")
