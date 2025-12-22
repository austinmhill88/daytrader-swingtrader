"""
Scheduler framework for automated tasks (Phase 1).
Handles pre-market prep, EOD processing, and nightly maintenance.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from typing import Dict, Any, Callable, List
from loguru import logger
import pytz


class TradingScheduler:
    """
    Manages scheduled tasks for autonomous trading operation.
    Phase 1 implementation with hooks for Phase 2-4 enhancements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scheduler.
        
        Args:
            config: Scheduler configuration from config.yaml
        """
        self.config = config
        scheduler_config = config.get('scheduler', {})
        
        self.enabled = scheduler_config.get('enabled', False)
        self.timezone = pytz.timezone(scheduler_config.get('timezone', 'America/New_York'))
        self.tasks_config = scheduler_config.get('tasks', {})
        
        self.scheduler = BackgroundScheduler(timezone=self.timezone)
        self.task_handlers: Dict[str, Callable] = {}
        
        # Alert notifier reference (will be set externally)
        self.notifier = None
        
        logger.info(f"TradingScheduler initialized | Enabled: {self.enabled}, TZ: {self.timezone}")
    
    def register_handler(self, action: str, handler: Callable) -> None:
        """
        Register a handler for a specific action.
        
        Args:
            action: Action name (e.g., 'data_sync', 'retrain_models')
            handler: Callable to execute for this action
        """
        self.task_handlers[action] = handler
        logger.info(f"Registered handler for action: {action}")
    
    def _execute_task(self, task_name: str, actions: List[str]) -> None:
        """
        Execute a scheduled task by running its actions.
        
        Args:
            task_name: Name of the task
            actions: List of action names to execute
        """
        logger.info(f"Executing scheduled task: {task_name}")
        
        failed_actions = []
        
        for action in actions:
            if action in self.task_handlers:
                try:
                    logger.info(f"Running action: {action}")
                    self.task_handlers[action]()
                    logger.info(f"Action completed: {action}")
                except Exception as e:
                    logger.error(f"Error in action {action}: {e}", exc_info=True)
                    failed_actions.append(action)
                    
                    # Send alert for failed action
                    if self.notifier:
                        self.notifier.send_alert(
                            title=f"Scheduled Task Failed: {action}",
                            message=f"Action '{action}' in task '{task_name}' failed: {str(e)}",
                            severity="warning",
                            metadata={'task': task_name, 'action': action, 'error': str(e)}
                        )
            else:
                logger.warning(f"No handler registered for action: {action}")
        
        logger.info(f"Task completed: {task_name}")
        
        # Send summary alert if any actions failed
        if failed_actions and self.notifier:
            self.notifier.send_alert(
                title=f"Task Completed with Failures: {task_name}",
                message=f"Task '{task_name}' completed with {len(failed_actions)} failed actions: {', '.join(failed_actions)}",
                severity="warning",
                metadata={'task': task_name, 'failed_actions': ', '.join(failed_actions)}
            )
    
    def start(self) -> None:
        """Start the scheduler with configured tasks."""
        if not self.enabled:
            logger.info("Scheduler is disabled")
            return
        
        # Schedule pre-market tasks
        pre_market = self.tasks_config.get('pre_market', {})
        if pre_market.get('enabled', False):
            time = pre_market.get('time', '08:30')
            actions = pre_market.get('actions', [])
            hour, minute = map(int, time.split(':'))
            
            self.scheduler.add_job(
                func=lambda: self._execute_task('pre_market', actions),
                trigger=CronTrigger(
                    hour=hour,
                    minute=minute,
                    day_of_week='mon-fri',
                    timezone=self.timezone
                ),
                id='pre_market',
                name='Pre-Market Preparation',
                replace_existing=True
            )
            logger.info(f"Scheduled pre-market tasks at {time}")
        
        # Schedule end-of-day tasks
        eod = self.tasks_config.get('end_of_day', {})
        if eod.get('enabled', False):
            time = eod.get('time', '16:30')
            actions = eod.get('actions', [])
            hour, minute = map(int, time.split(':'))
            
            self.scheduler.add_job(
                func=lambda: self._execute_task('end_of_day', actions),
                trigger=CronTrigger(
                    hour=hour,
                    minute=minute,
                    day_of_week='mon-fri',
                    timezone=self.timezone
                ),
                id='end_of_day',
                name='End-of-Day Processing',
                replace_existing=True
            )
            logger.info(f"Scheduled end-of-day tasks at {time}")
        
        # Schedule nightly tasks
        nightly = self.tasks_config.get('nightly', {})
        if nightly.get('enabled', False):
            time = nightly.get('time', '02:00')
            actions = nightly.get('actions', [])
            hour, minute = map(int, time.split(':'))
            
            self.scheduler.add_job(
                func=lambda: self._execute_task('nightly', actions),
                trigger=CronTrigger(
                    hour=hour,
                    minute=minute,
                    timezone=self.timezone
                ),
                id='nightly',
                name='Nightly Maintenance',
                replace_existing=True
            )
            logger.info(f"Scheduled nightly tasks at {time}")
        
        # Start scheduler
        self.scheduler.start()
        logger.info("Scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")
    
    def get_jobs(self) -> List[Dict]:
        """
        Get list of scheduled jobs.
        
        Returns:
            List of job information dicts
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        return jobs
    
    def pause_job(self, job_id: str) -> bool:
        """
        Pause a specific job.
        
        Args:
            job_id: Job ID to pause
            
        Returns:
            True if successful
        """
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Job paused: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Error pausing job {job_id}: {e}")
            return False
    
    def resume_job(self, job_id: str) -> bool:
        """
        Resume a paused job.
        
        Args:
            job_id: Job ID to resume
            
        Returns:
            True if successful
        """
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Job resumed: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Error resuming job {job_id}: {e}")
            return False


# Pre-defined action handlers (to be implemented in main.py or separate modules)

def data_sync_action():
    """Sync data from sources to local storage."""
    logger.info("ACTION: Data sync")
    # Implementation will be in main orchestrator

def universe_refresh_action():
    """Refresh trading universe based on liquidity criteria."""
    logger.info("ACTION: Universe refresh")
    # Implementation will be in universe module

def model_check_action():
    """Check model validity and performance."""
    logger.info("ACTION: Model check")
    # Phase 2: Check model metrics and drift

def flatten_intraday_action():
    """Flatten all intraday positions."""
    logger.info("ACTION: Flatten intraday positions")
    # Implementation will be in main orchestrator

def generate_reports_action():
    """Generate daily performance reports."""
    logger.info("ACTION: Generate reports")
    # Implementation will be in reporting module

def backup_data_action():
    """Backup critical data."""
    logger.info("ACTION: Backup data")
    # Implementation will backup database and logs

def retrain_models_action():
    """Retrain ML models with latest data."""
    logger.info("ACTION: Retrain models")
    # Phase 2: ML model retraining pipeline

def run_backtests_action():
    """Run backtests with latest parameters."""
    logger.info("ACTION: Run backtests")
    # Implementation will run walk-forward validation

def cleanup_action():
    """Cleanup old files and logs."""
    logger.info("ACTION: Cleanup")
    # Remove old logs, temp files, etc.
