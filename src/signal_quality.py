"""
Signal quality assessment and humility thresholds.
Implements adaptive signal thresholds, multi-factor agreement, and confidence calibration.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


class SignalQualityScorer:
    """
    Assesses signal quality and applies humility thresholds.
    Only trades when signal strength exceeds calibrated thresholds.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize signal quality scorer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        signal_config = config.get('signal_quality', {})
        
        # Base thresholds
        self.base_zscore_threshold = 2.0
        self.base_adx_threshold = 20.0
        self.base_confidence_threshold = 0.55
        
        # Adaptive thresholds
        self.enable_adaptive_thresholds = signal_config.get('enable_adaptive_thresholds', True)
        self.vol_scaling_factor = signal_config.get('vol_scaling_factor', 1.5)
        
        # Multi-factor agreement
        self.require_multi_factor = signal_config.get('require_multi_factor', True)
        self.min_factors_agree = signal_config.get('min_factors_agree', 2)
        
        # Confidence calibration
        self.enable_calibration = signal_config.get('enable_confidence_calibration', True)
        self.calibration_method = signal_config.get('calibration_method', 'isotonic')  # 'platt' or 'isotonic'
        
        # Calibrated models storage
        self.calibrated_models: Dict[str, any] = {}
        
        logger.info(
            f"SignalQualityScorer initialized | "
            f"Adaptive thresholds: {self.enable_adaptive_thresholds}, "
            f"Multi-factor: {self.require_multi_factor}, "
            f"Calibration: {self.enable_calibration}"
        )
    
    def calculate_adaptive_threshold(
        self,
        base_threshold: float,
        current_volatility: float,
        baseline_volatility: float = 0.15
    ) -> float:
        """
        Calculate adaptive threshold based on current volatility.
        In high volatility, require stronger signals.
        
        Args:
            base_threshold: Base threshold value
            current_volatility: Current realized volatility
            baseline_volatility: Baseline volatility (default 15%)
            
        Returns:
            Adjusted threshold
        """
        if not self.enable_adaptive_thresholds:
            return base_threshold
        
        # Scale threshold proportionally with volatility
        vol_ratio = current_volatility / baseline_volatility
        
        # Apply scaling factor (default 1.5x = increase threshold by 50% when vol doubles)
        adjusted_threshold = base_threshold * (1 + (vol_ratio - 1) * self.vol_scaling_factor)
        
        logger.debug(
            f"Adaptive threshold: {base_threshold:.2f} -> {adjusted_threshold:.2f} "
            f"(vol: {current_volatility:.2%}, baseline: {baseline_volatility:.2%})"
        )
        
        return adjusted_threshold
    
    def check_signal_strength(
        self,
        signal_value: float,
        threshold: float,
        volatility: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Check if signal exceeds adaptive threshold.
        
        Args:
            signal_value: Signal strength value
            threshold: Base threshold
            volatility: Current volatility for adaptive adjustment
            
        Returns:
            Tuple of (passes_threshold, adjusted_threshold)
        """
        # Calculate adaptive threshold if volatility provided
        if volatility is not None and self.enable_adaptive_thresholds:
            adjusted_threshold = self.calculate_adaptive_threshold(threshold, volatility)
        else:
            adjusted_threshold = threshold
        
        passes = abs(signal_value) >= adjusted_threshold
        
        return passes, adjusted_threshold
    
    def check_multi_factor_agreement(
        self,
        factors: Dict[str, bool],
        min_agree: Optional[int] = None
    ) -> Tuple[bool, int, List[str]]:
        """
        Check if minimum number of factors agree on signal direction.
        
        Args:
            factors: Dict of factor_name -> agrees (bool)
            min_agree: Minimum number of factors that must agree (default: self.min_factors_agree)
            
        Returns:
            Tuple of (passes, num_agree, agreeing_factors)
        """
        if not self.require_multi_factor:
            return True, len(factors), list(factors.keys())
        
        if min_agree is None:
            min_agree = self.min_factors_agree
        
        # Count agreeing factors
        agreeing = [name for name, agrees in factors.items() if agrees]
        num_agree = len(agreeing)
        
        passes = num_agree >= min_agree
        
        logger.debug(
            f"Multi-factor agreement: {num_agree}/{len(factors)} factors agree "
            f"(min: {min_agree}) - {'PASS' if passes else 'FAIL'}"
        )
        
        return passes, num_agree, agreeing
    
    def calibrate_model_confidence(
        self,
        model: any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        strategy_name: str
    ) -> any:
        """
        Calibrate model output probabilities using Platt scaling or Isotonic regression.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            strategy_name: Strategy name for storing calibrated model
            
        Returns:
            Calibrated model
        """
        if not self.enable_calibration:
            return model
        
        try:
            if self.calibration_method == 'platt':
                # Platt scaling (logistic regression on model outputs)
                calibrated_model = CalibratedClassifierCV(
                    model,
                    method='sigmoid',
                    cv='prefit'
                )
                calibrated_model.fit(X_val, y_val)
                
            elif self.calibration_method == 'isotonic':
                # Isotonic regression (non-parametric)
                calibrated_model = CalibratedClassifierCV(
                    model,
                    method='isotonic',
                    cv='prefit'
                )
                calibrated_model.fit(X_val, y_val)
                
            else:
                logger.warning(f"Unknown calibration method: {self.calibration_method}")
                return model
            
            # Store calibrated model
            self.calibrated_models[strategy_name] = calibrated_model
            
            logger.info(
                f"Confidence calibration complete for {strategy_name} "
                f"using {self.calibration_method} method"
            )
            
            return calibrated_model
            
        except Exception as e:
            logger.error(f"Error calibrating model: {e}")
            return model
    
    def get_calibrated_confidence(
        self,
        model: any,
        X: pd.DataFrame,
        strategy_name: str
    ) -> np.ndarray:
        """
        Get calibrated confidence scores from model.
        
        Args:
            model: Model (calibrated or uncalibrated)
            X: Features
            strategy_name: Strategy name
            
        Returns:
            Array of calibrated probabilities
        """
        # Check if we have a calibrated model for this strategy
        if strategy_name in self.calibrated_models:
            model = self.calibrated_models[strategy_name]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            # Return probability of positive class
            if proba.shape[1] > 1:
                return proba[:, 1]
            else:
                return proba[:, 0]
        else:
            logger.warning("Model does not support predict_proba")
            return np.zeros(len(X))
    
    def assess_signal_quality(
        self,
        signal_data: Dict,
        strategy_type: str,
        current_volatility: Optional[float] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Comprehensive signal quality assessment.
        
        Args:
            signal_data: Dictionary with signal components
                - zscore: Z-score signal (for mean reversion)
                - adx: ADX value (for trend following)
                - rsi: RSI value
                - confidence: Model confidence (if using ML)
                - spread: Bid-ask spread
                - liquidity: Liquidity score
            strategy_type: 'intraday_mean_reversion' or 'swing_trend_following'
            current_volatility: Current market volatility
            
        Returns:
            Tuple of (passes_quality_check, quality_score, details)
        """
        details = {
            'threshold_checks': {},
            'factor_agreement': {},
            'quality_score': 0.0
        }
        
        quality_score = 0.0
        max_score = 0.0
        
        # Strategy-specific checks
        if strategy_type == 'intraday_mean_reversion':
            # Check Z-score threshold
            if 'zscore' in signal_data:
                zscore = signal_data['zscore']
                threshold = self.base_zscore_threshold
                
                if current_volatility:
                    threshold = self.calculate_adaptive_threshold(threshold, current_volatility)
                
                passes_zscore = abs(zscore) >= threshold
                details['threshold_checks']['zscore'] = {
                    'value': zscore,
                    'threshold': threshold,
                    'passes': passes_zscore
                }
                
                if passes_zscore:
                    quality_score += abs(zscore) / threshold
                max_score += 1.0
            
            # Multi-factor agreement for mean reversion
            factors = {}
            if 'zscore' in signal_data:
                factors['zscore'] = abs(signal_data['zscore']) >= 1.5
            if 'rsi' in signal_data:
                rsi = signal_data['rsi']
                factors['rsi'] = rsi < 30 or rsi > 70
            if 'spread' in signal_data:
                factors['spread'] = signal_data['spread'] < 50  # Under 50 bps
            
            if factors:
                passes_factors, num_agree, agreeing = self.check_multi_factor_agreement(factors)
                details['factor_agreement'] = {
                    'passes': passes_factors,
                    'num_agree': num_agree,
                    'agreeing_factors': agreeing
                }
                if passes_factors:
                    quality_score += num_agree / len(factors)
                max_score += 1.0
        
        elif strategy_type == 'swing_trend_following':
            # Check ADX threshold
            if 'adx' in signal_data:
                adx = signal_data['adx']
                threshold = self.base_adx_threshold
                
                if current_volatility:
                    # In high vol, require stronger trend
                    threshold = self.calculate_adaptive_threshold(threshold, current_volatility)
                
                passes_adx = adx >= threshold
                details['threshold_checks']['adx'] = {
                    'value': adx,
                    'threshold': threshold,
                    'passes': passes_adx
                }
                
                if passes_adx:
                    quality_score += adx / threshold
                max_score += 1.0
            
            # Multi-factor agreement for trend following
            factors = {}
            if 'ema_cross' in signal_data:
                factors['ema_cross'] = signal_data['ema_cross']
            if 'adx' in signal_data:
                factors['adx'] = signal_data['adx'] >= 20
            if 'regime' in signal_data:
                # Trend strategy should align with trending regime
                regime = signal_data['regime']
                factors['regime'] = 'trending' in regime.lower()
            
            if factors:
                passes_factors, num_agree, agreeing = self.check_multi_factor_agreement(factors)
                details['factor_agreement'] = {
                    'passes': passes_factors,
                    'num_agree': num_agree,
                    'agreeing_factors': agreeing
                }
                if passes_factors:
                    quality_score += num_agree / len(factors)
                max_score += 1.0
        
        # Check ML confidence if available
        if 'confidence' in signal_data:
            confidence = signal_data['confidence']
            threshold = self.base_confidence_threshold
            
            passes_confidence = confidence >= threshold
            details['threshold_checks']['confidence'] = {
                'value': confidence,
                'threshold': threshold,
                'passes': passes_confidence
            }
            
            if passes_confidence:
                quality_score += confidence / threshold
            max_score += 1.0
        
        # Normalize quality score
        if max_score > 0:
            normalized_score = quality_score / max_score
        else:
            normalized_score = 0.0
        
        details['quality_score'] = normalized_score
        
        # Overall pass: score must be > 0.6 (60% of checks passed)
        passes = normalized_score >= 0.6
        
        logger.info(
            f"Signal quality for {strategy_type}: "
            f"{'PASS' if passes else 'FAIL'} "
            f"(score: {normalized_score:.2f})"
        )
        
        return passes, normalized_score, details
