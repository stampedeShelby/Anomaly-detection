"""
Data Loading and Preprocessing Module

Handles:
- CSV loading with datetime parsing
- Gap analysis and classification
- Shutdown detection
- Cycle segmentation
- Missing value handling
"""

import numpy as np
import pandas as pd
from .config import CONFIG, sec_to_samples


def load_and_preprocess_data(filepath, cfg=CONFIG):
    """
    Load CSV data and perform initial preprocessing.
    
    Args:
        filepath: Path to CSV file
        cfg: Configuration dictionary
    
    Returns:
        df: Preprocessed DataFrame with datetime, gap flags, cycle segments
    """
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['dt_diff'] = df['datetime'].diff().dt.total_seconds()
    
    # Gap analysis
    gaps = df['dt_diff'].dropna()
    print("=" * 60)
    print("DATA AUDIT SUMMARY")
    print("=" * 60)
    print("Gap summary (seconds):")
    print(gaps.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
    
    for t in [5, 30, 60, 300, 600, 1800, 3600, cfg['shutdown_threshold_sec']]:
        print(f"Gaps > {t}s: {(gaps > t).sum()}")
    
    print(f"Zero readings: {(df['current'] == 0).sum()}")
    print(f"NaN readings: {df['current'].isna().sum()}")
    print("=" * 60)
    
    # Handle NaNs
    if cfg['nan_policy'] == 'log_then_drop':
        nan_count = df['current'].isna().sum()
        if nan_count > 0:
            print(f"WARNING: Dropping {nan_count} NaN readings (nan_policy='log_then_drop')")
        df = df[df['current'].notna()].copy()
    
    # Segment cycles
    df = _segment_cycles(df, cfg)
    
    return df


def _segment_cycles(df, cfg):
    """
    Segment data into cycles and detect shutdowns.
    
    Args:
        df: DataFrame with datetime and current
        cfg: Configuration dictionary
    
    Returns:
        df: DataFrame with cycle_segment_id
    """
    # Detect cycle boundaries
    cycle_change = df[cfg['cycle_boundary']].ne(df[cfg['cycle_boundary']].shift(1))
    
    # Detect shutdowns (gaps >= threshold)
    df['shutdown_break'] = df['dt_diff'].fillna(0) > cfg['shutdown_threshold_sec']
    
    # Create segment IDs
    df['cycle_segment_id'] = (cycle_change | df['shutdown_break']).cumsum()
    
    # Add cycle start markers
    df['cycle_start'] = cycle_change
    
    # Compute gap flags
    df = compute_gap_flags(df, cfg)
    
    print(f"Total cycles detected: {df['cycle_segment_id'].nunique()}")
    print(f"Total shutdown events: {df['shutdown_break'].sum()}")
    
    return df


def compute_gap_flags(df, cfg=CONFIG):
    """
    Flag medium gaps and shutdowns for logging.
    
    Args:
        df: DataFrame with dt_diff
        cfg: Configuration dictionary
    
    Returns:
        df: DataFrame with medium_gap and shutdown_break flags
    """
    df['dt_diff'] = df['datetime'].diff().dt.total_seconds()
    lo, hi = cfg['medium_gap_range_sec']
    
    df['medium_gap'] = (df['dt_diff'] >= lo) & (df['dt_diff'] < hi)
    df['shutdown_break'] = df['dt_diff'] >= cfg['shutdown_threshold_sec']
    
    return df


def get_cycle_trace(df, cycle_id, cfg=CONFIG):
    """
    Extract raw current trace for a specific cycle.
    
    Args:
        df: DataFrame
        cycle_id: cycle_counter value
        cfg: Configuration dictionary
    
    Returns:
        Series: Current values for the cycle
    """
    return df[df[cfg['cycle_boundary']] == cycle_id]['current'].reset_index(drop=True)


def per_cycle_stat(df, col, agg='mean', cfg=CONFIG):
    """
    Compute per-cycle aggregated statistics.
    
    Args:
        df: DataFrame
        col: Column name
        agg: Aggregation function ('mean', 'max', 'min', 'std', etc.)
        cfg: Configuration dictionary
    
    Returns:
        Series: Aggregated values per cycle
    """
    return df.groupby(cfg['cycle_boundary'])[col].agg(agg)

