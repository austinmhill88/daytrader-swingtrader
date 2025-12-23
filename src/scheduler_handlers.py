"""
Scheduler action handlers to enable autonomous operation:
- data_sync: optional daily data sync
- universe_refresh: rebuild universe pre-market
- retrain_models: nightly ML training & promotion
- run_backtests: optional batch backtests
- cleanup: housekeeping (logs, old artifacts)
"""
from loguru import logger
from typing import Dict, List
from datetime import datetime, timedelta

from src.data_feed import HistoricalDataFeed
from src.universe_analytics import UniverseAnalytics
from src.ml_pipeline import MLPipeline


def make_handlers(config: Dict, alpaca_client):
    """
    Create scheduler action handlers.
    
    Args:
        config: Application configuration
        alpaca_client: Alpaca client instance
        
    Returns:
        Dictionary of action name -> handler function
    """
    ua = UniverseAnalytics(config)
    feed = HistoricalDataFeed(alpaca_client)
    pipeline = MLPipeline(config, alpaca_client)
    
    def data_sync():
        """Optional daily data sync - can implement archival to parquet."""
        logger.info("ACTION: Data sync - optional noop or implement archival")
        # Optional: Store recent bars to parquet for future training
        # This would extend data_storage.py to append to parquet files
    
    def universe_refresh():
        """Rebuild trading universe pre-market based on liquidity criteria."""
        logger.info("ACTION: Universe refresh - rebuilding trading universe")
        
        try:
            default_symbols = config['universe'].get('default_symbols', [])
            
            # Fetch recent data for universe analysis
            symbol_data = feed.get_multi_symbol_bars(default_symbols, "1Day", limit=60)
            
            # Build universe with filters
            result = ua.build_universe_with_filters(
                symbol_data=symbol_data,
                tier='core',
                alpaca_client=alpaca_client,
                apply_earnings_filter=True,
                apply_shortability_filter=True
            )
            
            logger.info(
                f"Universe refreshed: {result['final_count']} symbols | "
                f"Earnings blackout: {result.get('earnings_blackout_count', 0)}, "
                f"Non-shortable: {result.get('non_shortable_count', 0)}"
            )
            
        except Exception as e:
            logger.error(f"Error refreshing universe: {e}")
    
    def model_check():
        """Check model validity and performance - Phase 2 enhancement."""
        logger.info("ACTION: Model check")
        # Optional: Check if models need retraining based on age or drift
    
    def retrain_models():
        """
        Retrain ML models with latest data.
        Runs nightly training pipeline with walk-forward validation and promotion gates.
        """
        logger.info("ACTION: Retrain models - starting nightly ML training")
        
        if not config.get('ml_training', {}).get('enabled', False):
            logger.info("ML training disabled in config, skipping")
            return
        
        try:
            # Get symbols for training
            core_symbols = config['universe'].get('default_symbols', [])
            
            # Train intraday mean reversion model
            if config.get('strategies', {}).get('intraday_mean_reversion', {}).get('ml_model', {}).get('enabled', False):
                logger.info("Training intraday mean reversion model...")
                res = pipeline.train_intraday_mean_reversion(core_symbols)
                logger.info(f"Intraday training completed: {res}")
            
            # Train swing trend following model (similar pattern)
            # Can be extended for swing model training
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def flatten_intraday():
        """Flatten all intraday positions at end of day."""
        logger.info("ACTION: Flatten intraday positions")
        # This would be called by execution engine to close all intraday positions
        # Implementation would be in main.py to call execution_engine methods
    
    def generate_reports():
        """Generate daily performance reports."""
        logger.info("ACTION: Generate reports")
        # Optional: Generate daily P&L, trade summary, etc.
    
    def backup_data():
        """Backup critical data."""
        logger.info("ACTION: Backup data")
        # Optional: Backup database, logs, model artifacts
    
    def run_backtests():
        """Run backtests with latest parameters."""
        logger.info("ACTION: Run backtests - optional walk-forward validation")
        # Optional: Run full backtest suite for validation
    
    def cleanup():
        """Cleanup old files and logs."""
        logger.info("ACTION: Cleanup - prune old logs or artifacts")
        # Optional: Remove old log files, temp artifacts, etc.
        try:
            # Example: clean up logs older than retention period
            logs_dir = config.get('storage', {}).get('logs_dir', './logs')
            retention_days = 30  # Could be from config
            logger.info(f"Cleanup: would remove logs older than {retention_days} days from {logs_dir}")
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
    
    return {
        "data_sync": data_sync,
        "universe_refresh": universe_refresh,
        "model_check": model_check,
        "retrain_models": retrain_models,
        "flatten_intraday": flatten_intraday,
        "generate_reports": generate_reports,
        "backup_data": backup_data,
        "run_backtests": run_backtests,
        "cleanup": cleanup
    }
