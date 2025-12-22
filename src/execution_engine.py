"""
Order execution engine with position sizing and bracket order support.
"""
import math
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
        
        logger.info(
            f"ExecutionEngine initialized | "
            f"Order type: {self.default_order_type}, "
            f"Bracket orders: {self.use_bracket_orders}"
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
        
        # Check if we can afford it
        max_affordable = math.floor((equity * 0.95) / price)  # Use 95% to leave buffer
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
        
        logger.info(
            f"Order intent created | {intent.side.value.upper()} {intent.qty} {intent.symbol} "
            f"@ ${limit_price:.2f} | Strategy: {signal.strategy_name}"
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
    
    def execute_intents(self, intents: List[OrderIntent]) -> List[Optional[any]]:
        """
        Execute a list of order intents.
        
        Args:
            intents: List of OrderIntent objects
            
        Returns:
            List of order results (or None for failures)
        """
        results = []
        
        for intent in intents:
            try:
                result = self.execute_single_intent(intent)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing intent for {intent.symbol}: {e}")
                results.append(None)
        
        return results
    
    def execute_single_intent(self, intent: OrderIntent) -> Optional[any]:
        """
        Execute a single order intent.
        
        Args:
            intent: OrderIntent to execute
            
        Returns:
            Order object or None on failure
        """
        try:
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
            
            # Place order
            order = self.client.place_order(
                symbol=intent.symbol,
                qty=intent.qty,
                side=intent.side.value,
                order_type=intent.order_type.value,
                time_in_force=intent.time_in_force.value,
                limit_price=intent.limit_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order:
                # Log trade
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
                
                return order
            else:
                logger.error(f"Order placement failed for {intent.symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order for {intent.symbol}: {e}")
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
