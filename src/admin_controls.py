"""
Admin control interface for trading bot (Phase 3).
Provides manual controls for pause/resume, flattening, and emergency actions.
"""
from enum import Enum
from typing import Dict, Optional
from datetime import datetime
from loguru import logger
import threading


class TradingState(Enum):
    """Trading system states."""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    EMERGENCY_HALT = "emergency_halt"


class AdminControls:
    """
    Administrative controls for trading system.
    Phase 3 implementation for operational safety.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize admin controls.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.state = TradingState.STOPPED
        self.state_lock = threading.Lock()
        self.state_history = []
        
        # Track manual interventions
        self.interventions = []
        
        logger.info("AdminControls initialized")
    
    def get_state(self) -> TradingState:
        """
        Get current trading state.
        
        Returns:
            Current TradingState
        """
        with self.state_lock:
            return self.state
    
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed.
        
        Returns:
            True if trading allowed
        """
        with self.state_lock:
            return self.state == TradingState.RUNNING
    
    def pause_trading(self, reason: str = "Manual pause") -> bool:
        """
        Pause trading system.
        
        Args:
            reason: Reason for pausing
            
        Returns:
            True if successful
        """
        with self.state_lock:
            if self.state == TradingState.RUNNING:
                self.state = TradingState.PAUSED
                self._record_state_change("PAUSED", reason)
                logger.warning(f"🟡 Trading PAUSED: {reason}")
                return True
            else:
                logger.warning(f"Cannot pause from state: {self.state.value}")
                return False
    
    def resume_trading(self, confirm: bool = False) -> bool:
        """
        Resume trading system.
        
        Args:
            confirm: Confirmation required for safety
            
        Returns:
            True if successful
        """
        if not confirm:
            logger.warning("Resume requires confirmation (confirm=True)")
            return False
        
        with self.state_lock:
            if self.state == TradingState.PAUSED:
                self.state = TradingState.RUNNING
                self._record_state_change("RUNNING", "Manual resume")
                logger.info("🟢 Trading RESUMED")
                return True
            else:
                logger.warning(f"Cannot resume from state: {self.state.value}")
                return False
    
    def emergency_halt(self, reason: str = "Emergency halt") -> bool:
        """
        Emergency halt - stops all trading immediately.
        Requires manual intervention to recover.
        
        Args:
            reason: Reason for emergency halt
            
        Returns:
            True if successful
        """
        with self.state_lock:
            previous_state = self.state
            self.state = TradingState.EMERGENCY_HALT
            self._record_state_change("EMERGENCY_HALT", reason)
            
            logger.critical(f"🔴 EMERGENCY HALT ACTIVATED: {reason}")
            logger.critical(f"Previous state: {previous_state.value}")
            logger.critical("Manual intervention required to recover")
            
            return True
    
    def start_trading(self) -> bool:
        """
        Start trading system.
        
        Returns:
            True if successful
        """
        with self.state_lock:
            if self.state in [TradingState.STOPPED, TradingState.PAUSED]:
                self.state = TradingState.RUNNING
                self._record_state_change("RUNNING", "System start")
                logger.info("🟢 Trading STARTED")
                return True
            else:
                logger.warning(f"Cannot start from state: {self.state.value}")
                return False
    
    def stop_trading(self, reason: str = "Manual stop") -> bool:
        """
        Stop trading system gracefully.
        
        Args:
            reason: Reason for stopping
            
        Returns:
            True if successful
        """
        with self.state_lock:
            self.state = TradingState.STOPPED
            self._record_state_change("STOPPED", reason)
            logger.info(f"🔴 Trading STOPPED: {reason}")
            return True
    
    def reset_from_emergency(self, confirm_code: str) -> bool:
        """
        Reset from emergency halt state.
        Requires confirmation code for safety.
        
        Args:
            confirm_code: Confirmation code (must be "RESET_EMERGENCY")
            
        Returns:
            True if successful
        """
        if confirm_code != "RESET_EMERGENCY":
            logger.error("Invalid confirmation code for emergency reset")
            return False
        
        with self.state_lock:
            if self.state == TradingState.EMERGENCY_HALT:
                self.state = TradingState.STOPPED
                self._record_state_change("STOPPED", "Emergency reset")
                logger.warning("Emergency halt cleared - system in STOPPED state")
                return True
            else:
                logger.warning("Not in emergency halt state")
                return False
    
    def _record_state_change(self, new_state: str, reason: str) -> None:
        """Record state change in history."""
        self.state_history.append({
            'timestamp': datetime.now(),
            'state': new_state,
            'reason': reason
        })
        
        # Keep only last 100 state changes
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def record_intervention(self, action: str, details: str) -> None:
        """
        Record manual intervention.
        
        Args:
            action: Action taken
            details: Intervention details
        """
        self.interventions.append({
            'timestamp': datetime.now(),
            'action': action,
            'details': details
        })
        
        logger.info(f"Intervention recorded: {action} - {details}")
    
    def get_state_history(self, limit: int = 10) -> list:
        """
        Get recent state change history.
        
        Args:
            limit: Number of recent changes to return
            
        Returns:
            List of state changes
        """
        return self.state_history[-limit:]
    
    def get_status_summary(self) -> Dict:
        """
        Get status summary for monitoring.
        
        Returns:
            Dictionary with status information
        """
        with self.state_lock:
            return {
                'state': self.state.value,
                'trading_allowed': self.is_trading_allowed(),
                'last_state_change': self.state_history[-1] if self.state_history else None,
                'num_interventions': len(self.interventions),
                'recent_interventions': self.interventions[-5:] if self.interventions else []
            }


class CommandInterface:
    """
    Command interface for external control.
    Phase 3 implementation for API/CLI access.
    """
    
    def __init__(self, admin_controls: AdminControls, execution_engine, portfolio):
        """
        Initialize command interface.
        
        Args:
            admin_controls: AdminControls instance
            execution_engine: ExecutionEngine instance
            portfolio: PortfolioState instance
        """
        self.admin = admin_controls
        self.execution = execution_engine
        self.portfolio = portfolio
        
        logger.info("CommandInterface initialized")
    
    def execute_command(self, command: str, **kwargs) -> Dict:
        """
        Execute a command.
        
        Args:
            command: Command name
            **kwargs: Command arguments
            
        Returns:
            Result dictionary
        """
        try:
            if command == "pause":
                reason = kwargs.get('reason', 'Manual pause')
                success = self.admin.pause_trading(reason)
                return {'success': success, 'message': f"Trading paused: {reason}"}
            
            elif command == "resume":
                confirm = kwargs.get('confirm', False)
                success = self.admin.resume_trading(confirm)
                return {'success': success, 'message': 'Trading resumed' if success else 'Resume failed'}
            
            elif command == "flatten_all":
                reason = kwargs.get('reason', 'Manual flatten')
                success = self.execution.close_all_positions(reason)
                return {'success': success, 'message': 'All positions flattened' if success else 'Flatten failed'}
            
            elif command == "cancel_orders":
                reason = kwargs.get('reason', 'Manual cancel')
                success = self.execution.cancel_all_orders(reason)
                return {'success': success, 'message': 'Orders cancelled' if success else 'Cancel failed'}
            
            elif command == "emergency_halt":
                reason = kwargs.get('reason', 'Emergency halt')
                success = self.admin.emergency_halt(reason)
                # Also flatten and cancel
                self.execution.close_all_positions("Emergency flatten")
                self.execution.cancel_all_orders("Emergency cancel")
                return {'success': success, 'message': f'Emergency halt activated: {reason}'}
            
            elif command == "status":
                status = {
                    'admin': self.admin.get_status_summary(),
                    'portfolio': {
                        'equity': self.portfolio.equity(),
                        'positions': self.portfolio.position_count(),
                        'daily_pnl': self.portfolio.daily_pnl(),
                        'daily_pnl_pct': self.portfolio.daily_pnl_pct()
                    }
                }
                return {'success': True, 'status': status}
            
            else:
                return {'success': False, 'message': f'Unknown command: {command}'}
                
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return {'success': False, 'message': str(e)}
