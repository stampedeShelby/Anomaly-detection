"""
Risk Fusion Module

Combines multiple anomaly detectors into unified risk scores.
Implements multi-detector corroboration for higher precision.
"""

import numpy as np
import pandas as pd
from .config import CONFIG, sec_to_samples


def risk_fusion(df, shape_df=None, if_scores=None, cfg=CONFIG):
    """
    Compute unified risk score from multiple detectors.
    
    Logic:
    - Weighted combination of spike/drift/shape/model detectors
    - Spike override: if spike detected, set risk=1.0
    - Smoothing: rolling mean over smoothing_sec
    
    Args:
        df: DataFrame with spike_flag, drift_flag
        shape_df: DataFrame with shape_severity per cycle
        if_scores: Series or array of model scores (0-1 normalized)
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Added 'risk' and 'risk_smoothed' columns
    """
    df = df.copy()
    df['risk'] = 0.0
    w = cfg['risk']['weights']
    
    # Spikes
    if 'spike_flag' in df:
        df['risk'] += w['spike'] * df['spike_flag'].astype(float)
    
    # Drift
    if 'drift_flag' in df:
        df['risk'] += w['drift'] * df['drift_flag'].astype(float)
    
    # Shape
    if shape_df is not None and 'shape_severity' in shape_df:
        shape_map = shape_df.set_index('cycle_counter')['shape_severity'].to_dict()
        df['shape_sev'] = df['cycle_counter'].map(shape_map)
        df['risk'] += w['shape'] * df['shape_sev'].map({'warn': 0.5, 'critical': 1.0}).fillna(0.0)
    
    # Model
    if if_scores is not None:
        df['risk'] += w['model'] * if_scores
    
    # Spike override
    if cfg['risk']['spike_override'] and 'spike_flag' in df:
        df.loc[df['spike_flag'], 'risk'] = 1.0
    
    # Smoothing
    sm = sec_to_samples(cfg['risk']['smoothing_sec'], cfg.get('sampling_sec', 0.5))
    df['risk_smoothed'] = df['risk'].rolling(sm, min_periods=max(2, sm // 2)).mean()
    
    return df


def fused_severity(df, shape_df, cycle_summary, cfg=CONFIG):
    """
    Combine multiple detectors to assign per-cycle severity.
    
    Fusion logic:
    - Critical: ≥2 strong detectors (model/shape/spike)
    - Warn: 1 strong + 1 assist (drift)
    - Info: otherwise
    
    Args:
        df: DataFrame with spike_flag, drift_flag
        shape_df: DataFrame with shape_severity
        cycle_summary: DataFrame with cycle_counter and model_severity
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: cycle_summary with fused_severity column
    """
    flags = cycle_summary[['cycle_counter', 'model_severity']].copy()
    
    # Merge shape severity
    if 'shape_severity' in shape_df:
        flags = flags.merge(shape_df[['cycle_counter', 'shape_severity']], on='cycle_counter', how='left')
    else:
        flags['shape_severity'] = None
    
    # Spike aggregation
    if 'spike_flag' in df:
        spike_agg = df.groupby(cfg['cycle_boundary'])['spike_flag'].max()
        flags['spike_any'] = flags['cycle_counter'].map(spike_agg).fillna(0).astype(bool)
    else:
        flags['spike_any'] = False
    
    # Drift aggregation
    if 'drift_flag' in df:
        drift_agg = df.groupby(cfg['cycle_boundary'])['drift_flag'].max()
        flags['drift_any_raw'] = flags['cycle_counter'].map(drift_agg).fillna(0).astype(bool)
    else:
        flags['drift_any_raw'] = False
    
    # Fusion logic
    def compute_fused(row):
        votes_strong = 0
        votes_assist = 0
        
        # Strong votes
        if row.get('model_severity') in ['warn', 'critical']:
            votes_strong += 1
        if row.get('shape_severity') in ['warn', 'critical']:
            votes_strong += 1
        if row.get('spike_any', False):
            votes_strong += 1
        
        # Assist votes
        if row.get('drift_any_raw', False):
            votes_assist += 1
        
        # Decision
        if votes_strong >= 2:
            return 'critical'
        if votes_strong == 1 and votes_assist >= 1:
            return 'warn'
        return 'info'
    
    flags['fused_severity'] = flags.apply(compute_fused, axis=1)
    
    return flags

