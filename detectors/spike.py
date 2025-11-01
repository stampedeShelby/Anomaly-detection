"""
Spike Detector - Strategy Pattern Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from ..core import BaseDetector, BaseValidator
from ..config import CONFIG


class SpikeDetector(BaseDetector):
    """
    Spike anomaly detector.
    
    Detects sudden current spikes using z-score and percent-over-baseline.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize spike detector."""
        cfg = config or CONFIG.get('spike', {})
        super().__init__('spike', cfg)
        self.z_threshold = cfg.get('z_threshold', 5.0)
        self.pct_threshold = cfg.get('pct_over_baseline', 0.20)
        self.min_duration_sec = cfg.get('min_duration_sec', 1.0)
        self.sampling_sec = CONFIG.get('sampling_sec', 0.5)
        self.inrush_mask_sec = CONFIG.get('inrush_mask_sec', 5.0)
        self.cycle_boundary = CONFIG.get('cycle_boundary', 'cycle_counter')
        
        self.validator = BaseValidator()
    
    def get_required_columns(self) -> List[str]:
        """Required columns."""
        return ['current', 'datetime']
    
    def _sec_to_samples(self, seconds: float) -> int:
        """Convert seconds to samples."""
        return int(np.ceil(seconds / self.sampling_sec))
    
    def _compute_inrush_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute inrush mask for each cycle.
        
        Args:
            df: DataFrame with datetime and cycle_boundary
            
        Returns:
            Boolean Series indicating inrush periods
        """
        # Detect cycle starts
        cycle_start = df[self.cycle_boundary].ne(df[self.cycle_boundary].shift(1))
        
        # Compute time within cycle
        cycle_time_sec = df.groupby(self.cycle_boundary)['datetime'].transform(
            lambda s: (s - s.iloc[0]).dt.total_seconds()
        )
        
        # Apply mask
        return cycle_time_sec <= self.inrush_mask_sec
    
    def _compute_rolling_z_score(self, df: pd.DataFrame, window_samples: int) -> tuple:
        """
        Compute rolling z-score for current values.
        
        Args:
            df: DataFrame with current column
            window_samples: Rolling window size in samples
            
        Returns:
            Tuple of (mean, std, z_score) Series
        """
        mu = df['current'].rolling(window_samples, min_periods=window_samples // 2).mean()
        sd = df['current'].rolling(window_samples, min_periods=window_samples // 2).std()
        z = (df['current'] - mu) / (sd + 1e-6)
        return mu, sd, z
    
    def _compute_cycle_baseline(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute expanding mean baseline per cycle.
        
        Args:
            df: DataFrame with current and cycle_boundary
            
        Returns:
            Series of baseline values
        """
        return df.groupby(self.cycle_boundary)['current'].transform(
            lambda x: x.expanding().mean()
        )
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect spike anomalies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with spike detection results
        """
        # Validate input
        self.validate(df)
        
        df = df.copy()
        
        # Compute inrush mask
        df['inrush_mask'] = self._compute_inrush_mask(df)
        
        # Rolling z-score
        win_samples = self._sec_to_samples(10.0)  # 10 second window
        _, _, df['z_score'] = self._compute_rolling_z_score(df, win_samples)
        
        # Cycle baseline
        df['cycle_baseline'] = self._compute_cycle_baseline(df)
        df['pct_over_baseline'] = (df['current'] - df['cycle_baseline']) / (df['cycle_baseline'] + 1e-6)
        
        # Spike conditions
        spike_persist_samples = self._sec_to_samples(self.min_duration_sec)
        spike_raw = (
            (df['z_score'] > self.z_threshold) &
            (df['pct_over_baseline'] > self.pct_threshold) &
            (~df['inrush_mask'])
        )
        
        # Persistence check
        df['spike_flag'] = spike_raw.rolling(
            spike_persist_samples, 
            min_periods=spike_persist_samples
        ).sum().ge(spike_persist_samples)
        
        return df
    
    def get_severity(self, df: pd.DataFrame) -> pd.Series:
        """
        Get severity scores (binary: 0 or 1).
        
        Args:
            df: DataFrame with spike_flag
            
        Returns:
            Series of severity scores
        """
        if 'spike_flag' not in df.columns:
            raise ValueError("spike_flag not found. Run detect() first.")
        
        return df['spike_flag'].astype(float)

