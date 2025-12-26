"""
Rate-limit-aware batching with exponential backoff (Phase 3).
Manages API call batching and handles rate limit errors gracefully.
"""
import time
from typing import List, Callable, Any, Optional, Dict
from datetime import datetime, timedelta
from collections import deque
from loguru import logger


class RateLimitError(Exception):
    """Custom exception for rate limit errors."""
    pass


class RateLimiter:
    """
    Rate limiter with exponential backoff for API calls.
    Phase 3 implementation for execution sophistication.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize rate limiter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        execution_config = config.get('execution', {})
        
        # Rate limiting parameters
        self.max_calls_per_minute = execution_config.get('max_api_calls_per_minute', 200)
        self.max_calls_per_second = self.max_calls_per_minute / 60.0
        
        # Batching parameters
        self.batch_size = 50  # Max items per batch
        self.batch_delay_ms = 100  # Delay between batches (milliseconds)
        
        # Backoff parameters
        self.initial_backoff_seconds = 1.0
        self.max_backoff_seconds = 60.0
        self.backoff_multiplier = 2.0
        self.current_backoff = self.initial_backoff_seconds
        
        # Call tracking
        self.call_times: deque = deque(maxlen=1000)
        self.rate_limit_hits = 0
        self.last_rate_limit_time: Optional[datetime] = None
        
        # Throttle state
        self.is_throttled = False
        self.throttle_until: Optional[datetime] = None
        
        logger.info(
            f"RateLimiter initialized | "
            f"Max calls: {self.max_calls_per_minute}/min, "
            f"Batch size: {self.batch_size}"
        )
    
    def wait_if_needed(self) -> None:
        """
        Wait if rate limit is about to be exceeded.
        Implements sliding window rate limiting.
        """
        # Check if currently throttled
        if self.is_throttled and self.throttle_until:
            now = datetime.now()
            if now < self.throttle_until:
                wait_seconds = (self.throttle_until - now).total_seconds()
                logger.warning(
                    f"Rate limit throttle active, waiting {wait_seconds:.1f}s"
                )
                time.sleep(wait_seconds)
                self.is_throttled = False
                self.throttle_until = None
            else:
                self.is_throttled = False
                self.throttle_until = None
        
        # Remove calls older than 1 minute
        now = time.time()
        cutoff = now - 60.0
        
        while self.call_times and self.call_times[0] < cutoff:
            self.call_times.popleft()
        
        # Check if we're at the limit
        if len(self.call_times) >= self.max_calls_per_minute:
            # Calculate wait time until oldest call expires
            oldest_call = self.call_times[0]
            wait_time = 60.0 - (now - oldest_call) + 0.1  # Add small buffer
            
            if wait_time > 0:
                logger.debug(
                    f"Rate limit approaching ({len(self.call_times)}/{self.max_calls_per_minute}), "
                    f"waiting {wait_time:.2f}s"
                )
                time.sleep(wait_time)
        
        # Record this call
        self.call_times.append(time.time())
    
    def execute_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: int = 5,
        **kwargs
    ) -> Any:
        """
        Execute a function with exponential backoff on rate limit errors.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            max_retries: Maximum retry attempts
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries exhausted
        """
        for attempt in range(max_retries):
            try:
                # Wait if needed before making call
                self.wait_if_needed()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Success - reset backoff
                if self.current_backoff > self.initial_backoff_seconds:
                    logger.info("API call successful, resetting backoff")
                    self.current_backoff = self.initial_backoff_seconds
                
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                is_rate_limit_error = any(
                    phrase in error_str
                    for phrase in ['rate limit', 'too many requests', '429']
                )
                
                if is_rate_limit_error:
                    self.rate_limit_hits += 1
                    self.last_rate_limit_time = datetime.now()
                    
                    if attempt < max_retries - 1:
                        wait_time = self.current_backoff
                        logger.warning(
                            f"Rate limit error (attempt {attempt + 1}/{max_retries}), "
                            f"backing off for {wait_time:.1f}s"
                        )
                        
                        # Set throttle
                        self.is_throttled = True
                        self.throttle_until = datetime.now() + timedelta(seconds=wait_time)
                        
                        time.sleep(wait_time)
                        
                        # Increase backoff
                        self.current_backoff = min(
                            self.current_backoff * self.backoff_multiplier,
                            self.max_backoff_seconds
                        )
                    else:
                        logger.error(f"Rate limit error, max retries exhausted")
                        raise RateLimitError(f"Rate limit exceeded after {max_retries} retries")
                else:
                    # Not a rate limit error, re-raise immediately
                    raise
        
        raise RateLimitError(f"Failed after {max_retries} retries")
    
    def execute_batch(
        self,
        func: Callable,
        items: List[Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Execute function on items in batches with rate limiting.
        
        Args:
            func: Function to execute on each item
            items: List of items to process
            batch_size: Batch size (defaults to self.batch_size)
            
        Returns:
            List of results
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        results = []
        num_batches = (len(items) + batch_size - 1) // batch_size
        
        logger.info(
            f"Processing {len(items)} items in {num_batches} batches "
            f"(batch size: {batch_size})"
        )
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.debug(
                f"Processing batch {batch_num}/{num_batches} "
                f"({len(batch)} items)"
            )
            
            # Process batch items
            for item in batch:
                try:
                    result = self.execute_with_backoff(func, item)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item in batch: {e}")
                    results.append(None)
            
            # Delay between batches (except for last batch)
            if i + batch_size < len(items):
                time.sleep(self.batch_delay_ms / 1000.0)
        
        logger.info(
            f"Batch processing complete: {len([r for r in results if r is not None])}/{len(items)} successful"
        )
        
        return results
    
    def get_current_rate(self) -> float:
        """
        Get current API call rate (calls per minute).
        
        Returns:
            Current rate
        """
        now = time.time()
        cutoff = now - 60.0
        
        # Count calls in last minute
        recent_calls = sum(1 for t in self.call_times if t >= cutoff)
        
        return recent_calls
    
    def get_rate_limit_stats(self) -> Dict:
        """
        Get rate limiting statistics.
        
        Returns:
            Dictionary with stats
        """
        current_rate = self.get_current_rate()
        utilization_pct = (current_rate / self.max_calls_per_minute) * 100.0
        
        return {
            'current_rate': current_rate,
            'max_rate': self.max_calls_per_minute,
            'utilization_pct': utilization_pct,
            'rate_limit_hits': self.rate_limit_hits,
            'last_rate_limit': self.last_rate_limit_time.isoformat() if self.last_rate_limit_time else None,
            'is_throttled': self.is_throttled,
            'current_backoff_seconds': self.current_backoff,
            'total_calls': len(self.call_times)
        }
    
    def reset_backoff(self) -> None:
        """Reset backoff to initial value."""
        self.current_backoff = self.initial_backoff_seconds
        self.is_throttled = False
        self.throttle_until = None
        logger.info("Rate limiter backoff reset")


class BatchProcessor:
    """
    Batch processor for multiple API operations.
    Coordinates rate limiting across different operation types.
    """
    
    def __init__(self, rate_limiter: RateLimiter):
        """
        Initialize batch processor.
        
        Args:
            rate_limiter: RateLimiter instance
        """
        self.rate_limiter = rate_limiter
        
        # Operation queues
        self.order_queue: List[Dict] = []
        self.cancel_queue: List[str] = []
        self.query_queue: List[str] = []
        
        logger.info("BatchProcessor initialized")
    
    def add_order(self, order_params: Dict) -> None:
        """
        Add order to batch queue.
        
        Args:
            order_params: Order parameters dict
        """
        self.order_queue.append(order_params)
        logger.debug(f"Added order to queue (queue size: {len(self.order_queue)})")
    
    def add_cancel(self, order_id: str) -> None:
        """
        Add cancellation to batch queue.
        
        Args:
            order_id: Order ID to cancel
        """
        self.cancel_queue.append(order_id)
        logger.debug(f"Added cancel to queue (queue size: {len(self.cancel_queue)})")
    
    def add_query(self, symbol: str) -> None:
        """
        Add data query to batch queue.
        
        Args:
            symbol: Symbol to query
        """
        self.query_queue.append(symbol)
        logger.debug(f"Added query to queue (queue size: {len(self.query_queue)})")
    
    def process_orders(self, submit_func: Callable) -> List[Any]:
        """
        Process all queued orders.
        
        Args:
            submit_func: Function to submit orders
            
        Returns:
            List of order results
        """
        if not self.order_queue:
            return []
        
        logger.info(f"Processing {len(self.order_queue)} queued orders")
        
        results = self.rate_limiter.execute_batch(
            submit_func,
            self.order_queue
        )
        
        self.order_queue.clear()
        return results
    
    def process_cancels(self, cancel_func: Callable) -> List[bool]:
        """
        Process all queued cancellations.
        
        Args:
            cancel_func: Function to cancel orders
            
        Returns:
            List of cancellation results
        """
        if not self.cancel_queue:
            return []
        
        logger.info(f"Processing {len(self.cancel_queue)} queued cancellations")
        
        results = self.rate_limiter.execute_batch(
            cancel_func,
            self.cancel_queue
        )
        
        self.cancel_queue.clear()
        return results
    
    def process_queries(self, query_func: Callable) -> List[Any]:
        """
        Process all queued data queries.
        
        Args:
            query_func: Function to query data
            
        Returns:
            List of query results
        """
        if not self.query_queue:
            return []
        
        logger.info(f"Processing {len(self.query_queue)} queued queries")
        
        results = self.rate_limiter.execute_batch(
            query_func,
            self.query_queue
        )
        
        self.query_queue.clear()
        return results
    
    def get_queue_summary(self) -> Dict:
        """
        Get summary of queue states.
        
        Returns:
            Dictionary with queue sizes
        """
        return {
            'orders_queued': len(self.order_queue),
            'cancels_queued': len(self.cancel_queue),
            'queries_queued': len(self.query_queue),
            'total_queued': len(self.order_queue) + len(self.cancel_queue) + len(self.query_queue)
        }
