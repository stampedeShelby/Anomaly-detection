"""
Anomaly Detectors Module

Implements rule-based detectors for:
- Spike detection (z-score + baseline + inrush masking)
- Drift detection (EWMA + CUSUM)
- Shape deviation (cycle correlation to template)
"""

import numpy as np
import pandas as pd
from .config import CONFIG, sec_to_samples


def apply_inrush_mask(df, cfg=CONFIG):
    """
    Mask first N seconds of each cycle to exclude inrush currents.
    
    Args:
        df: DataFrame with datetime and cycle_boundary
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Added 'inrush_mask' boolean column
    """
    # Detect cycle starts
    df['cycle_start'] = df[cfg['cycle_boundary']].ne(df[cfg['cycle_boundary']].shift(1))
    
    # Compute time within cycle
    df['cycle_time_sec'] = df.groupby(cfg['cycle_boundary'])['datetime'].transform(
        lambda s: (s - s.iloc[0]).dt.total_seconds()
    )
    
    # Apply mask
    df['inrush_mask'] = df['cycle_time_sec'] <= cfg['inrush_mask_sec']
    
    return df


def compute_spike_flags(df, cfg=CONFIG):
    """
    Detect spikes using z-score and percent-over-baseline.
    
    Logic:
    - z-score > threshold
    - >20% above cycle baseline
    - Persists for min_duration_sec
    - NOT in inrush mask period
    
    Args:
        df: DataFrame with current and inrush_mask
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Added 'spike_flag' boolean column
    """
    # Rolling z-score (10s window)
    win = sec_to_samples(10, cfg.get('sampling_sec', 0.5))
    mu = df['current'].rolling(win, min_periods=win // 2).mean()
    sd = df['current'].rolling(win, min_periods=win // 2).std()
    df['z'] = (df['current'] - mu) / (sd + 1e-6)
    
    # Cycle baseline (expanding mean per cycle)
    df['cycle_baseline'] = df.groupby(cfg['cycle_boundary'])['current'].transform(
        lambda x: x.expanding().mean()
    )
    df['pct_over_baseline'] = (df['current'] - df['cycle_baseline']) / (df['cycle_baseline'] + 1e-6)
    
    # Spike conditions
    spike_persist = sec_to_samples(cfg['spike']['min_duration_sec'], cfg.get('sampling_sec', 0.5))
    spike_raw = (
        (df['z'] > cfg['spike']['z_threshold']) &
        (df['pct_over_baseline'] > cfg['spike']['pct_over_baseline']) &
        (~df['inrush_mask'])
    )
    
    # Persistence check
    df['spike_flag'] = spike_raw.rolling(spike_persist, min_periods=spike_persist).sum().ge(spike_persist)
    
    return df


def drift_ewma_cusum(df, cfg=CONFIG):
    """
    Detect drift using EWMA + CUSUM.
    
    Logic:
    - EWMA: exponential moving average to track baseline
    - CUSUM: cumulative sum of deviations above threshold k
    - Flag when CUSUM exceeds h
    - Reset CUSUM on shutdowns
    
    Args:
        df: DataFrame with current and shutdown_break
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Added 'ewma', 'cusum_pos', 'drift_flag' columns
    """
    alpha = cfg['drift']['ewma_alpha']
    k = cfg['drift']['cusum_k']
    h = cfg['drift']['cusum_h']
    
    # EWMA (exponential moving average)
    df['ewma'] = df['current'].ewm(alpha=alpha, adjust=False).mean()
    
    # Residual
    df['resid'] = df['current'] - df['ewma']
    
    # CUSUM (cumulative sum of positive deviations)
    df['cusum_pos'] = 0.0
    df['cusum_pos'] = np.maximum(0, (df['resid'] - k) + df['cusum_pos'].shift(fill_value=0))
    
    # Reset on shutdown
    if cfg['drift']['reset_on_shutdown'] and 'shutdown_break' in df:
        df.loc[df['shutdown_break'], 'cusum_pos'] = 0
    
    # Flag drift
    df['drift_flag'] = df['cusum_pos'] > h
    
    return df


def compute_gap_flags(df, cfg=CONFIG):
    """
    Flag medium gaps and shutdown breaks for event logging.
    
    Args:
        df: DataFrame with dt_diff
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Added 'medium_gap' and 'shutdown_break' columns
    """
    df['dt_diff'] = df['datetime'].diff().dt.total_seconds()
    lo, hi = cfg['medium_gap_range_sec']
    
    df['medium_gap'] = (df['dt_diff'] >= lo) & (df['dt_diff'] < hi)
    df['shutdown_break'] = df['dt_diff'] >= cfg['shutdown_threshold_sec']
    
    return df


def tune_drift_thresholds(df, cycle_summary, cfg=CONFIG):
    """
    Tune drift thresholds to reduce noise (e.g., 95th-99th percentile).
    
    Args:
        df: DataFrame with ewma and cusum_pos
        cycle_summary: DataFrame with cycle_counter
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Updated cycle_summary with drift_any_tuned flag
    """
    # Compute per-cycle EWMA and CUSUM summaries
    ewma_mean_by_cycle = df.groupby(cfg['cycle_boundary'])['ewma'].mean() if 'ewma' in df else pd.Series(dtype=float)
    cusum_max_by_cycle = df.groupby(cfg['cycle_boundary'])['cusum_pos'].max() if 'cusum_pos' in df else pd.Series(dtype=float)
    
    # Choose 95th percentile thresholds (more conservative)
    ewma_thresh = np.nan if ewma_mean_by_cycle.empty else np.percentile(ewma_mean_by_cycle.dropna(), 95)
    cusum_thresh = np.nan if cusum_max_by_cycle.empty else np.percentile(cusum_max_by_cycle.dropna(), 95)
    
    # Create tuned drift flags
    drift_tuned_map = pd.Series(False, index=cycle_summary['cycle_counter'])
    
    if not ewma_mean_by_cycle.empty and not cusum_max_by_cycle.empty:
        tuned_idx = (ewma_mean_by_cycle > ewma_thresh) & (cusum_max_by_cycle > cusum_thresh)
        drift_tuned_map.loc[tuned_idx.index] = tuned_idx.values
    elif not ewma_mean_by_cycle.empty:
        tuned_idx = ewma_mean_by_cycle > ewma_thresh
        drift_tuned_map.loc[tuned_idx.index] = tuned_idx.values
    elif not cusum_max_by_cycle.empty:
        tuned_idx = cusum_max_by_cycle > cusum_thresh
        drift_tuned_map.loc[tuned_idx.index] = tuned_idx.values
    
    return drift_tuned_map

