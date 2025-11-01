"""
Comprehensive Validation and Error Handling

Production-ready validation for all pipeline components.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .base import BaseValidator


class DataValidator(BaseValidator):
    """
    Data validation utilities.
    """
    
    @staticmethod
    def validate_time_series(
        df: pd.DataFrame,
        required_cols: List[str] = ['datetime', 'current'],
        check_monotonic: bool = True,
        check_duplicates: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate time series data.
        
        Args:
            df: DataFrame to validate
            required_cols: Required column names
            check_monotonic: Check if datetime is monotonic
            check_duplicates: Check for duplicate timestamps
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("DataFrame is empty")
        
        # Check datetime monotonicity
        if check_monotonic and 'datetime' in df.columns:
            if not df['datetime'].is_monotonic_increasing:
                errors.append("Datetime column is not monotonically increasing")
        
        # Check duplicates
        if check_duplicates and 'datetime' in df.columns:
            duplicates = df['datetime'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate timestamps")
        
        # Check for excessive missing values
        if 'current' in df.columns:
            missing_pct = df['current'].isna().sum() / len(df) * 100
            if missing_pct > 50:
                errors.append(f"Excessive missing values: {missing_pct:.1f}%")
        
        # Check for invalid values
        if 'current' in df.columns:
            if (df['current'] < 0).any():
                errors.append("Found negative current values")
            if (df['current'] > 1000).any():
                errors.append("Found suspiciously high current values (>1000)")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required keys
        required_keys = ['sampling_sec', 'cycle_boundary', 'inrush_mask_sec']
        missing = [key for key in required_keys if key not in config]
        if missing:
            errors.append(f"Missing required config keys: {missing}")
        
        # Validate thresholds
        if 'spike' in config:
            spike_config = config['spike']
            if 'z_threshold' in spike_config:
                z_thr = spike_config['z_threshold']
                if not (1 <= z_thr <= 20):
                    errors.append(f"Spike z_threshold out of range: {z_thr}")
        
        if 'drift' in config:
            drift_config = config['drift']
            if 'ewma_alpha' in drift_config:
                alpha = drift_config['ewma_alpha']
                if not (0 < alpha < 1):
                    errors.append(f"Drift ewma_alpha out of range: {alpha}")
        
        # Validate risk weights sum to reasonable value
        if 'risk' in config and 'weights' in config['risk']:
            weights = config['risk']['weights']
            total_weight = sum(weights.values())
            if not (0 < total_weight <= 2):
                errors.append(f"Risk weights sum to invalid value: {total_weight}")
        
        return len(errors) == 0, errors


class ModelValidator(BaseValidator):
    """
    Model validation utilities.
    """
    
    @staticmethod
    def validate_features(features: np.ndarray, expected_n_features: Optional[int] = None) -> Tuple[bool, List[str]]:
        """
        Validate feature array.
        
        Args:
            features: Feature array
            expected_n_features: Expected number of features
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(features, np.ndarray):
            errors.append(f"Features must be numpy array, got {type(features)}")
            return False, errors
        
        if features.ndim != 2:
            errors.append(f"Features must be 2D array, got {features.ndim}D")
        
        if features.size == 0:
            errors.append("Features array is empty")
        
        if expected_n_features and features.shape[1] != expected_n_features:
            errors.append(f"Expected {expected_n_features} features, got {features.shape[1]}")
        
        # Check for NaN/Inf
        if np.isnan(features).any():
            errors.append("Features contain NaN values")
        
        if np.isinf(features).any():
            errors.append("Features contain Inf values")
        
        return len(errors) == 0, errors


class PipelineValidator(BaseValidator):
    """
    End-to-end pipeline validation.
    """
    
    def __init__(self):
        """Initialize validators."""
        self.data_validator = DataValidator()
        self.config_validator = DataValidator()
        self.model_validator = ModelValidator()
    
    def validate_pipeline(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        features: Optional[np.ndarray] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate entire pipeline.
        
        Args:
            df: Input data
            config: Configuration
            features: Extracted features (optional)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors = []
        
        # Validate data
        data_valid, data_errors = self.data_validator.validate_time_series(df)
        all_errors.extend(data_errors)
        
        # Validate config
        config_valid, config_errors = self.config_validator.validate_config(config)
        all_errors.extend(config_errors)
        
        # Validate features if provided
        if features is not None:
            features_valid, features_errors = self.model_validator.validate_features(features)
            all_errors.extend(features_errors)
        
        return len(all_errors) == 0, all_errors


# Convenience function
def validate_pipeline_inputs(df: pd.DataFrame, config: Dict[str, Any], features: Optional[np.ndarray] = None) -> None:
    """
    Validate pipeline inputs and raise exceptions on errors.
    
    Args:
        df: Input DataFrame
        config: Configuration
        features: Feature array (optional)
        
    Raises:
        ValueError: If validation fails
    """
    validator = PipelineValidator()
    is_valid, errors = validator.validate_pipeline(df, config, features)
    
    if not is_valid:
        error_msg = "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

