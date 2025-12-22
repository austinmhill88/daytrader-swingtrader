"""
ML model training with overfitting prevention (Phase 2).
Implements purged k-fold CV, walk-forward validation, and promotion gates.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from loguru import logger

try:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not available")


class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purging and embargo.
    Prevents lookahead bias in time series data.
    """
    
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        """
        Initialize purged CV splitter.
        
        Args:
            n_splits: Number of CV folds
            embargo_pct: Percentage of data to embargo after each train set
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """
        Generate train/test splits with purging.
        
        Args:
            X: Feature DataFrame
            y: Target series (optional)
            groups: Group labels (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        test_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            
            # Train on all data before test (with embargo)
            train_end = test_start - embargo_size
            if train_end > 0:
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, test_end)
                
                yield train_idx, test_idx


class MLModelTrainer:
    """
    ML model training with overfitting prevention.
    Phase 2 implementation - extensible for Phase 3 production.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ML model trainer.
        
        Args:
            config: Application configuration
        """
        if not ML_AVAILABLE:
            raise ImportError("ML libraries not installed. Install lightgbm, xgboost, scikit-learn.")
        
        self.config = config
        ml_config = config.get('ml_training', {})
        
        self.enabled = ml_config.get('enabled', False)
        self.validation_method = ml_config.get('validation_method', 'walk_forward')
        self.model_dir = Path(config.get('storage', {}).get('model_dir', './data/models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Promotion gates
        gates = ml_config.get('promotion_gates', {})
        self.min_sharpe = gates.get('min_sharpe', 1.0)
        self.max_drawdown = gates.get('max_drawdown_pct', 10.0)
        self.min_trades = gates.get('min_trades', 100)
        self.min_win_rate = gates.get('min_win_rate', 0.45)
        
        # Purged CV config
        bt_config = config.get('backtesting', {})
        self.enable_purged_cv = bt_config.get('enable_purged_cv', False)
        self.embargo_pct = bt_config.get('embargo_pct', 0.01)
        
        logger.info(
            f"MLModelTrainer initialized | "
            f"Enabled: {self.enabled}, Method: {self.validation_method}"
        )
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = 'lightgbm',
        params: Optional[Dict] = None
    ) -> Any:
        """
        Train an ML model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: 'lightgbm', 'xgboost', or 'random_forest'
            params: Model hyperparameters
            
        Returns:
            Trained model
        """
        try:
            if model_type == 'lightgbm':
                default_params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }
                if params:
                    default_params.update(params)
                
                train_data = lgb.Dataset(X_train, label=y_train)
                model = lgb.train(default_params, train_data, num_boost_round=100)
                
            elif model_type == 'xgboost':
                default_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
                if params:
                    default_params.update(params)
                
                dtrain = xgb.DMatrix(X_train, label=y_train)
                model = xgb.train(default_params, dtrain, num_boost_round=100)
                
            elif model_type == 'random_forest':
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5
                }
                if params:
                    default_params.update(params)
                
                model = RandomForestClassifier(**default_params)
                model.fit(X_train, y_train)
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            logger.info(f"Trained {model_type} model on {len(X_train)} samples")
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'lightgbm',
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation with purging if enabled.
        
        Args:
            X: Features
            y: Labels
            model_type: Model type to train
            n_splits: Number of CV folds
            
        Returns:
            Dictionary of CV metrics
        """
        try:
            if self.enable_purged_cv:
                cv = PurgedTimeSeriesSplit(n_splits=n_splits, embargo_pct=self.embargo_pct)
                logger.info(f"Using purged time series CV (embargo: {self.embargo_pct*100}%)")
            else:
                cv = TimeSeriesSplit(n_splits=n_splits)
                logger.info("Using standard time series CV")
            
            scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'auc': []
            }
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                model = self.train_model(X_train, y_train, model_type)
                
                # Predict
                if model_type == 'lightgbm':
                    y_pred_proba = model.predict(X_test)
                elif model_type == 'xgboost':
                    dtest = xgb.DMatrix(X_test)
                    y_pred_proba = model.predict(dtest)
                else:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calculate metrics
                scores['accuracy'].append(accuracy_score(y_test, y_pred))
                scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
                scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
                scores['auc'].append(roc_auc_score(y_test, y_pred_proba))
                
                logger.info(
                    f"Fold {fold+1}/{n_splits} | "
                    f"Acc: {scores['accuracy'][-1]:.3f}, "
                    f"AUC: {scores['auc'][-1]:.3f}"
                )
            
            # Aggregate results
            cv_results = {
                'accuracy_mean': np.mean(scores['accuracy']),
                'accuracy_std': np.std(scores['accuracy']),
                'precision_mean': np.mean(scores['precision']),
                'recall_mean': np.mean(scores['recall']),
                'auc_mean': np.mean(scores['auc']),
                'auc_std': np.std(scores['auc'])
            }
            
            logger.info(
                f"CV Results | Acc: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}, "
                f"AUC: {cv_results['auc_mean']:.3f} ± {cv_results['auc_std']:.3f}"
            )
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save model to disk with metadata.
        
        Args:
            model: Trained model
            model_name: Name for saved model
            metadata: Optional metadata dict
            
        Returns:
            True if successful
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{model_name}_{timestamp}.pkl"
            filepath = self.model_dir / filename
            
            # Save model
            joblib.dump(model, filepath)
            
            # Save metadata
            if metadata:
                metadata['timestamp'] = timestamp
                metadata['model_name'] = model_name
                metadata['filepath'] = str(filepath)
                
                metadata_file = self.model_dir / f"{model_name}_{timestamp}_metadata.json"
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """
        Load model from disk.
        
        Args:
            model_name: Model name
            version: Optional specific version (latest if None)
            
        Returns:
            Loaded model or None
        """
        try:
            if version:
                pattern = f"{model_name}_{version}.pkl"
            else:
                pattern = f"{model_name}_*.pkl"
            
            files = list(self.model_dir.glob(pattern))
            if not files:
                logger.warning(f"No model files found matching {pattern}")
                return None
            
            # Use most recent
            filepath = sorted(files)[-1]
            model = joblib.load(filepath)
            
            logger.info(f"Loaded model from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def check_promotion_gates(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check if model passes promotion gates.
        
        Args:
            metrics: Model performance metrics
            
        Returns:
            Tuple of (passed, failed_gates)
        """
        failed_gates = []
        
        # Check Sharpe ratio
        if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] < self.min_sharpe:
            failed_gates.append(f"Sharpe {metrics['sharpe_ratio']:.2f} < {self.min_sharpe}")
        
        # Check max drawdown
        if 'max_drawdown_pct' in metrics and metrics['max_drawdown_pct'] > self.max_drawdown:
            failed_gates.append(f"Max DD {metrics['max_drawdown_pct']:.2f}% > {self.max_drawdown}%")
        
        # Check number of trades
        if 'num_trades' in metrics and metrics['num_trades'] < self.min_trades:
            failed_gates.append(f"Trades {metrics['num_trades']} < {self.min_trades}")
        
        # Check win rate
        if 'win_rate' in metrics and metrics['win_rate'] < self.min_win_rate * 100:
            failed_gates.append(f"Win rate {metrics['win_rate']:.1f}% < {self.min_win_rate*100}%")
        
        passed = len(failed_gates) == 0
        
        if passed:
            logger.info("✓ Model passed all promotion gates")
        else:
            logger.warning(f"✗ Model failed promotion gates: {', '.join(failed_gates)}")
        
        return passed, failed_gates
