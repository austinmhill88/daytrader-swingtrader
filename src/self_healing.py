"""
Self-healing and health monitoring (Phase 3).
Watchdogs, automatic recovery, and system health checks.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from loguru import logger


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Optional[Dict] = None


class SelfHealingMonitor:
    """
    Self-healing monitor with automatic recovery.
    Phase 3 implementation for system reliability.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize self-healing monitor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.health_checks: Dict[str, HealthCheck] = {}
        self.watchdogs: Dict[str, 'Watchdog'] = {}
        self.recovery_actions: Dict[str, Callable] = {}
        
        self.is_running = False
        self.monitor_thread = None
        
        logger.info("SelfHealingMonitor initialized")
    
    def register_watchdog(
        self,
        name: str,
        check_interval: int = 60,
        timeout: int = 300,
        recovery_action: Optional[Callable] = None
    ) -> None:
        """
        Register a watchdog for a component.
        
        Args:
            name: Component name
            check_interval: Interval between checks (seconds)
            timeout: Timeout before considering unhealthy (seconds)
            recovery_action: Optional recovery function
        """
        watchdog = Watchdog(name, check_interval, timeout)
        self.watchdogs[name] = watchdog
        
        if recovery_action:
            self.recovery_actions[name] = recovery_action
        
        logger.info(f"Registered watchdog: {name} (interval: {check_interval}s)")
    
    def heartbeat(self, component: str) -> None:
        """
        Record heartbeat from a component.
        
        Args:
            component: Component name
        """
        if component in self.watchdogs:
            self.watchdogs[component].heartbeat()
    
    def check_health(self, component: str) -> HealthCheck:
        """
        Check health of a specific component.
        
        Args:
            component: Component name
            
        Returns:
            HealthCheck result
        """
        if component in self.watchdogs:
            watchdog = self.watchdogs[component]
            
            if watchdog.is_healthy():
                status = HealthStatus.HEALTHY
                message = "Component is healthy"
            elif watchdog.is_responsive():
                status = HealthStatus.DEGRADED
                message = "Component responding but slow"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Component not responding (timeout: {watchdog.timeout}s)"
            
            check = HealthCheck(
                component=component,
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    'last_heartbeat': watchdog.last_heartbeat,
                    'timeout': watchdog.timeout
                }
            )
        else:
            check = HealthCheck(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message="Component not monitored",
                timestamp=datetime.now()
            )
        
        self.health_checks[component] = check
        return check
    
    def check_all_health(self) -> List[HealthCheck]:
        """
        Check health of all registered components.
        
        Returns:
            List of HealthCheck results
        """
        checks = []
        for component in self.watchdogs.keys():
            check = self.check_health(component)
            checks.append(check)
        return checks
    
    def attempt_recovery(self, component: str) -> bool:
        """
        Attempt to recover an unhealthy component.
        
        Args:
            component: Component name
            
        Returns:
            True if recovery attempted
        """
        if component in self.recovery_actions:
            try:
                logger.warning(f"Attempting recovery for {component}")
                recovery_fn = self.recovery_actions[component]
                recovery_fn()
                logger.info(f"Recovery action completed for {component}")
                return True
            except Exception as e:
                logger.error(f"Recovery failed for {component}: {e}")
                return False
        else:
            logger.warning(f"No recovery action registered for {component}")
            return False
    
    def start_monitoring(self) -> None:
        """Start health monitoring loop."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring loop."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Internal monitoring loop."""
        while self.is_running:
            try:
                # Check all components
                checks = self.check_all_health()
                
                # Attempt recovery for unhealthy components
                for check in checks:
                    if check.status == HealthStatus.UNHEALTHY:
                        logger.warning(f"Unhealthy component: {check.component}")
                        self.attempt_recovery(check.component)
                
                # Sleep before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def get_system_health(self) -> Dict:
        """
        Get overall system health summary.
        
        Returns:
            Dictionary with health summary
        """
        checks = self.check_all_health()
        
        status_counts = {
            'healthy': 0,
            'degraded': 0,
            'unhealthy': 0,
            'critical': 0
        }
        
        for check in checks:
            status_counts[check.status.value] += 1
        
        # Overall status
        if status_counts['critical'] > 0 or status_counts['unhealthy'] > len(checks) // 2:
            overall = HealthStatus.CRITICAL
        elif status_counts['unhealthy'] > 0:
            overall = HealthStatus.UNHEALTHY
        elif status_counts['degraded'] > 0:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return {
            'overall_status': overall.value,
            'components': len(checks),
            'status_counts': status_counts,
            'checks': [
                {
                    'component': c.component,
                    'status': c.status.value,
                    'message': c.message
                }
                for c in checks
            ]
        }


class Watchdog:
    """
    Watchdog for monitoring component health.
    """
    
    def __init__(self, name: str, check_interval: int, timeout: int):
        """
        Initialize watchdog.
        
        Args:
            name: Component name
            check_interval: Check interval in seconds
            timeout: Timeout in seconds
        """
        self.name = name
        self.check_interval = check_interval
        self.timeout = timeout
        self.last_heartbeat: Optional[datetime] = None
        self.heartbeat_count = 0
    
    def heartbeat(self) -> None:
        """Record a heartbeat."""
        self.last_heartbeat = datetime.now()
        self.heartbeat_count += 1
    
    def is_healthy(self) -> bool:
        """
        Check if component is healthy.
        
        Returns:
            True if healthy
        """
        if self.last_heartbeat is None:
            return False
        
        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        return elapsed < self.check_interval * 2
    
    def is_responsive(self) -> bool:
        """
        Check if component is responsive (even if slow).
        
        Returns:
            True if responsive
        """
        if self.last_heartbeat is None:
            return False
        
        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        return elapsed < self.timeout


class ConnectionManager:
    """
    Manages connections with automatic reconnection.
    Phase 3 implementation for data stream reliability.
    """
    
    def __init__(self, max_retries: int = 5, backoff_factor: float = 2.0):
        """
        Initialize connection manager.
        
        Args:
            max_retries: Maximum reconnection attempts
            backoff_factor: Exponential backoff factor
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_count = 0
        self.last_attempt: Optional[datetime] = None
        
        logger.info("ConnectionManager initialized")
    
    def attempt_connect(self, connect_fn: Callable) -> bool:
        """
        Attempt connection with exponential backoff.
        
        Args:
            connect_fn: Function to establish connection
            
        Returns:
            True if successful
        """
        if self.retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) reached")
            return False
        
        # Calculate backoff delay
        if self.last_attempt:
            wait_time = min(300, self.backoff_factor ** self.retry_count)
            elapsed = (datetime.now() - self.last_attempt).total_seconds()
            
            if elapsed < wait_time:
                remaining = wait_time - elapsed
                logger.info(f"Waiting {remaining:.1f}s before retry")
                time.sleep(remaining)
        
        try:
            self.last_attempt = datetime.now()
            logger.info(f"Connection attempt {self.retry_count + 1}/{self.max_retries}")
            
            connect_fn()
            
            # Success - reset counter
            self.retry_count = 0
            logger.info("Connection established")
            return True
            
        except Exception as e:
            self.retry_count += 1
            logger.error(f"Connection failed: {e}")
            return False
    
    def reset(self) -> None:
        """Reset retry counter."""
        self.retry_count = 0
        self.last_attempt = None
