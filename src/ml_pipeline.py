"""
ML Training Pipeline Orchestrator (Phase 2).
Coordinates feature engineering, model training, validation, and deployment.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from src.feature_store import FeatureStore
from src.ml_trainer import MLModelTrainer
from backtest.walk_forward import WalkForwardValidator


class MLPipeline:
    """
    ML training pipeline orchestrator.
    Phase 2 implementation for enabling ML across strategies.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ML pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        ml_config = config.get('ml_training', {})
        
        # Components
        self.feature_store = FeatureStore(config)
        self.ml_trainer = MLModelTrainer(config)
        
        # Configuration
        self.enabled = ml_config.get('enabled', False)
        self.training_schedule = ml_config.get('training_schedule', 'weekly')
        self.validation_method = ml_config.get('validation_method', 'walk_forward')
        
        # Strategy-specific ML config
        self.strategy_configs = {}
        for strategy_name in ['intraday_mean_reversion', 'swing_trend_following']:
            strategy_config = config.get('strategies', {}).get(strategy_name, {})
            ml_model_config = strategy_config.get('ml_model', {})
            
            if ml_model_config.get('enabled', False):
                self.strategy_configs[strategy_name] = {
                    'model_type': ml_model_config.get('model_type', 'lightgbm'),
                    'features': ml_model_config.get('features', []),
                    'enabled': True
                }
        
        # Walk-forward validator
        self.walk_forward_validator = None
        if self.validation_method == 'walk_forward':
            try:
                self.walk_forward_validator = WalkForwardValidator(config)
            except Exception as e:
                logger.warning(f"Could not initialize walk-forward validator: {e}")
        
        logger.info(
            f"MLPipeline initialized | "
            f"Enabled: {self.enabled}, "
            f"Strategies with ML: {len(self.strategy_configs)}"
        )
    
    def train_strategy_model(
        self,
        strategy_name: str,
        historical_data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Train ML model for a specific strategy.
        
        Args:
            strategy_name: Strategy identifier
            historical_data: Dict of symbol -> OHLCV DataFrame
            start_date: Training start date (optional)
            end_date: Training end date (optional)
            
        Returns:
            Dictionary with training results or None
        """
        if not self.enabled:
            logger.warning("ML training is disabled in config")
            return None
        
        if strategy_name not in self.strategy_configs:
            logger.warning(f"No ML config found for strategy: {strategy_name}")
            return None
        
        strategy_config = self.strategy_configs[strategy_name]
        model_type = strategy_config['model_type']
        feature_list = strategy_config['features']
        
        logger.info(
            f"Training {model_type} model for {strategy_name} | "
            f"{len(historical_data)} symbols, {len(feature_list)} features"
        )
        
        try:
            # Step 1: Compute features for all symbols
            logger.info("Step 1: Computing features...")
            feature_dfs = {}
            
            for symbol, df in historical_data.items():
                if df is None or len(df) < 100:
                    logger.debug(f"Skipping {symbol} - insufficient data")
                    continue
                
                # Compute features
                df_with_features = self.feature_store.compute_features(
                    df.copy(),
                    feature_set='all'
                )
                
                # Filter to requested features
                available_features = [f for f in feature_list if f in df_with_features.columns]
                if len(available_features) < len(feature_list):
                    missing = set(feature_list) - set(available_features)
                    logger.debug(f"{symbol}: Missing features {missing}")
                
                if available_features:
                    feature_dfs[symbol] = df_with_features[available_features + ['close', 'returns']]
            
            logger.info(f"Features computed for {len(feature_dfs)} symbols")
            
            # Step 2: Create training labels
            logger.info("Step 2: Creating labels...")
            X_train, y_train = self._create_labels(
                feature_dfs,
                strategy_name,
                start_date,
                end_date
            )
            
            if X_train is None or len(X_train) == 0:
                logger.error("No training data generated")
                return None
            
            logger.info(
                f"Training data: {len(X_train)} samples, "
                f"{X_train.shape[1]} features, "
                f"Positive rate: {y_train.mean():.2%}"
            )
            
            # Step 3: Train model with cross-validation
            logger.info("Step 3: Training model...")
            model = self.ml_trainer.train_model(
                X_train,
                y_train,
                model_type=model_type
            )
            
            # Step 4: Cross-validation
            logger.info("Step 4: Cross-validation...")
            cv_results = self.ml_trainer.cross_validate(
                X_train,
                y_train,
                model_type=model_type,
                n_splits=5
            )
            
            # Step 5: Walk-forward validation (if enabled)
            backtest_metrics = None
            if self.walk_forward_validator:
                logger.info("Step 5: Walk-forward validation...")
                try:
                    backtest_metrics = self._run_walk_forward_validation(
                        strategy_name,
                        model,
                        feature_dfs
                    )
                except Exception as e:
                    logger.warning(f"Walk-forward validation failed: {e}")
            
            # Step 6: Check promotion gates
            logger.info("Step 6: Checking promotion gates...")
            metrics_for_gates = {
                'auc': cv_results.get('auc_mean', 0),
                'accuracy': cv_results.get('accuracy_mean', 0)
            }
            
            # Add backtest metrics if available
            if backtest_metrics:
                metrics_for_gates.update({
                    'sharpe_ratio': backtest_metrics.get('sharpe', 0),
                    'max_drawdown_pct': backtest_metrics.get('max_drawdown', 100),
                    'num_trades': backtest_metrics.get('total_trades', 0),
                    'win_rate': backtest_metrics.get('win_rate', 0)
                })
            
            passed_gates, failed_gates = self.ml_trainer.check_promotion_gates(metrics_for_gates)
            
            # Step 7: Save model to registry
            logger.info("Step 7: Saving model...")
            model_name = f"{strategy_name}_model"
            
            model_params = {
                'model_type': model_type,
                'features': feature_list,
                'n_samples': len(X_train),
                'training_date': datetime.now().isoformat()
            }
            
            run_id = self.ml_trainer.save_model_to_registry(
                model=model,
                model_name=model_name,
                strategy=strategy_name,
                metrics=metrics_for_gates,
                params=model_params,
                model_type=model_type
            )
            
            # Step 8: Promote if gates passed
            if passed_gates and run_id:
                logger.info(f"✓ Model passed promotion gates, promoting to production")
                self.ml_trainer.promote_model(model_name, run_id, stage="Production")
            else:
                logger.warning(
                    f"✗ Model failed promotion gates: {', '.join(failed_gates)}"
                )
            
            # Return results
            results = {
                'strategy': strategy_name,
                'model_type': model_type,
                'features': feature_list,
                'training_samples': len(X_train),
                'cv_results': cv_results,
                'backtest_metrics': backtest_metrics,
                'promotion_passed': passed_gates,
                'failed_gates': failed_gates,
                'model_id': run_id,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(
                f"Training complete for {strategy_name} | "
                f"CV AUC: {cv_results.get('auc_mean', 0):.3f}, "
                f"Promotion: {'PASSED' if passed_gates else 'FAILED'}"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model for {strategy_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_labels(
        self,
        feature_dfs: Dict[str, pd.DataFrame],
        strategy_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Create training labels based on strategy logic.
        
        Args:
            feature_dfs: Dict of symbol -> feature DataFrame
            strategy_name: Strategy name for label creation logic
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Tuple of (X, y) or (None, None)
        """
        try:
            X_list = []
            y_list = []
            
            for symbol, df in feature_dfs.items():
                df = df.copy()
                
                # Filter by date if specified
                if start_date and 'timestamp' in df.columns:
                    df = df[df['timestamp'] >= start_date]
                if end_date and 'timestamp' in df.columns:
                    df = df[df['timestamp'] <= end_date]
                
                if len(df) < 50:
                    continue
                
                # Create labels based on strategy
                if strategy_name == 'intraday_mean_reversion':
                    # Label = 1 if next period return is positive
                    df['label'] = (df['returns'].shift(-1) > 0).astype(int)
                
                elif strategy_name == 'swing_trend_following':
                    # Label = 1 if 5-period forward return is positive
                    df['forward_return'] = df['close'].shift(-5) / df['close'] - 1
                    df['label'] = (df['forward_return'] > 0.01).astype(int)  # >1% gain
                
                else:
                    logger.warning(f"Unknown strategy for labeling: {strategy_name}")
                    continue
                
                # Drop rows with NaN
                df = df.dropna()
                
                if len(df) == 0:
                    continue
                
                # Extract features and labels
                feature_cols = [col for col in df.columns if col not in ['label', 'close', 'returns', 'forward_return', 'timestamp', 'symbol']]
                
                X_symbol = df[feature_cols]
                y_symbol = df['label']
                
                X_list.append(X_symbol)
                y_list.append(y_symbol)
            
            if not X_list:
                return None, None
            
            # Combine all symbols
            X = pd.concat(X_list, ignore_index=True)
            y = pd.concat(y_list, ignore_index=True)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return None, None
    
    def _run_walk_forward_validation(
        self,
        strategy_name: str,
        model: Any,
        feature_dfs: Dict[str, pd.DataFrame]
    ) -> Optional[Dict]:
        """
        Run walk-forward validation.
        
        Args:
            strategy_name: Strategy name
            model: Trained model
            feature_dfs: Feature DataFrames
            
        Returns:
            Backtest metrics dictionary or None
        """
        if not self.walk_forward_validator:
            return None
        
        try:
            # Run walk-forward backtest
            results = self.walk_forward_validator.run_validation(
                strategy_name=strategy_name,
                model=model,
                feature_dfs=feature_dfs
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Walk-forward validation error: {e}")
            return None
    
    def should_retrain(
        self,
        strategy_name: str,
        model_age_days: int,
        recent_performance: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Determine if a strategy model should be retrained.
        
        Args:
            strategy_name: Strategy name
            model_age_days: Age of current model in days
            recent_performance: Recent performance metrics
            
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check schedule
        if self.training_schedule == 'daily' and model_age_days >= 1:
            return True, "Scheduled daily retrain"
        elif self.training_schedule == 'weekly' and model_age_days >= 7:
            return True, "Scheduled weekly retrain"
        elif self.training_schedule == 'monthly' and model_age_days >= 30:
            return True, "Scheduled monthly retrain"
        
        # Check performance degradation
        if recent_performance:
            sharpe = recent_performance.get('sharpe', 0)
            if sharpe < 0.5:
                return True, f"Performance degradation (Sharpe: {sharpe:.2f})"
        
        # Check model age limit
        if model_age_days > 90:
            return True, f"Model too old ({model_age_days} days)"
        
        return False, "No retrain needed"
    
    def get_pipeline_status(self) -> Dict:
        """
        Get status of ML pipeline.
        
        Returns:
            Status dictionary
        """
        return {
            'enabled': self.enabled,
            'training_schedule': self.training_schedule,
            'validation_method': self.validation_method,
            'strategies_with_ml': list(self.strategy_configs.keys()),
            'feature_store_ready': self.feature_store is not None,
            'ml_trainer_ready': self.ml_trainer is not None,
            'walk_forward_ready': self.walk_forward_validator is not None
        }
