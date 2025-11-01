"""
Feature Extraction Module

Computes statistical, temporal, and spectral features per cycle segment.
Includes cycle-shape normalization and correlation with rolling templates.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft
from sklearn.preprocessing import StandardScaler
from .config import CONFIG


def extract_segment_features(df, cfg=CONFIG):
    """
    Extract per-segment statistical, temporal, and spectral features.
    
    Args:
        df: DataFrame with current and cycle_segment_id
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: One row per segment with extracted features
    """
    feats = []
    
    for seg_id, g in df.groupby('cycle_segment_id'):
        cur = g['current'].values
        
        # Skip very short segments
        if len(cur) < 5:
            continue
        
        # Statistical features
        mean = np.mean(cur)
        std = np.std(cur)
        min_val = np.min(cur)
        max_val = np.max(cur)
        range_val = max_val - min_val
        
        # Temporal features
        duration_sec = (g['datetime'].iloc[-1] - g['datetime'].iloc[0]).total_seconds()
        
        # Rise time: time from start to peak
        rise_time = max(np.argmax(cur) - np.argmax(cur > 0), 0)
        
        # Shape features
        skewness = skew(cur)
        kurt = kurtosis(cur)
        
        # Area under curve
        auc = np.trapz(cur)
        
        # Spectral energy (FFT)
        fft_energy = np.sum(np.abs(rfft(cur)) ** 2)
        
        feats.append({
            'cycle_segment_id': seg_id,
            'cycle_counter': g['cycle_counter'].iloc[0],
            'length': len(cur),
            'duration_sec': duration_sec,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'rise_time': rise_time,
            'skew': skewness,
            'kurtosis': kurt,
            'auc': auc,
            'fft_energy': fft_energy
        })
    
    return pd.DataFrame(feats)


def normalize_cycles(df, cfg=CONFIG):
    """
    Normalize cycles to fixed length for shape comparison.
    
    Args:
        df: DataFrame with current and cycle_counter
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Normalized cycle vectors (cycle_counter x normalize_points)
    """
    N = cfg['shape']['normalize_points']
    
    def normalize_cycle(cycle_num, g):
        x = np.linspace(0, 1, len(g))
        xi = np.linspace(0, 1, N)
        yi = np.interp(xi, x, g.values)
        return pd.Series(yi, name=cycle_num)
    
    cycle_vectors_list = []
    for cycle_id, group in df.groupby(cfg['cycle_boundary'])['current']:
        norm_series = normalize_cycle(cycle_id, group)
        cycle_vectors_list.append(norm_series)
    
    cycle_vectors = pd.concat(cycle_vectors_list, axis=1).T
    return cycle_vectors.sort_index()


def row_corr(a, b):
    """
    Compute Pearson correlation between two vectors.
    
    Args:
        a, b: pd.Series or arrays
    
    Returns:
        float: Correlation coefficient
    """
    if a.isna().any() or b.isna().any():
        return np.nan
    
    a0 = (a - a.mean()) / (a.std() + 1e-6)
    b0 = (b - b.mean()) / (b.std() + 1e-6)
    
    return float(np.clip(np.dot(a0, b0) / len(a0), -1, 1))


def compute_shape_correlation(df, cfg=CONFIG):
    """
    Compute cycle-shape correlation to rolling template.
    
    For each cycle:
    - Normalize to 100 points
    - Compute rolling template (mean of last 20 cycles)
    - Compute correlation with template
    - Flag warn if corr <= p10, critical if corr <= p05
    
    Args:
        df: DataFrame with current and cycle_counter
        cfg: Configuration dictionary
    
    Returns:
        tuple: (shape_df, thresholds_dict)
            - shape_df: DataFrame with cycle_counter, shape_corr, shape_severity
            - thresholds: dict with 'p10' and 'p05' thresholds
    """
    N = cfg['shape']['normalize_points']
    T = cfg['shape']['template_window_cycles']
    
    # Normalize cycles
    cycle_mat = normalize_cycles(df, cfg)
    
    # Compute rolling template
    templates = cycle_mat.rolling(T, min_periods=5).mean()
    
    # Compute correlations
    corr_scores = []
    for idx in cycle_mat.index:
        try:
            tmpl = templates.loc[idx]
            vec = cycle_mat.loc[idx]
            corr = row_corr(vec, tmpl)
            
            corr_scores.append({
                'cycle_counter': idx,
                'shape_corr': corr
            })
        except (KeyError, IndexError):
            continue
    
    shape_df = pd.DataFrame(corr_scores).dropna()
    
    if len(shape_df) == 0:
        print("WARNING: No shape correlations computed!")
        return pd.DataFrame(), {}
    
    # Compute thresholds
    p10 = shape_df['shape_corr'].quantile(cfg['shape']['warn_quantile'])
    p05 = shape_df['shape_corr'].quantile(cfg['shape']['crit_quantile'])
    
    # Assign severities
    def sev(c):
        if pd.isna(c):
            return None
        if c <= p05:
            return 'critical'
        if c <= p10:
            return 'warn'
        return None
    
    shape_df['shape_severity'] = shape_df['shape_corr'].apply(sev)
    
    thresholds = {'p10': float(p10), 'p05': float(p05)}
    
    return shape_df, thresholds


def simple_shape_diagnostics(trace):
    """
    Quick shape diagnostics for manual inspection of cycles.
    
    Args:
        trace: Series or array of current values
    
    Returns:
        dict: Diagnostic statistics
    """
    if len(trace) == 0:
        return {
            'length': 0, 'mean': np.nan, 'std': np.nan, 'max': np.nan,
            'rise_slope': np.nan, 'fall_slope': np.nan,
            'early_mean': np.nan, 'late_mean': np.nan
        }
    
    n = len(trace)
    w10 = max(1, int(0.10 * n))
    w20 = max(1, int(0.20 * n))
    
    # Early vs late means (coarse drift/shape proxy)
    early_mean = trace.iloc[:w10].mean()
    late_mean = trace.iloc[-w10:].mean()
    
    # Rise/fall slopes using linear fit on first/last 20%
    x_early = np.arange(w20)
    x_late = np.arange(w20)
    y_early = trace.iloc[:w20].to_numpy()
    y_late = trace.iloc[-w20:].to_numpy()
    
    if len(y_early) >= 2 and len(y_late) >= 2:
        rise_slope = np.polyfit(x_early, y_early, 1)[0]
        fall_slope = np.polyfit(x_late, y_late, 1)[0]
    else:
        rise_slope = np.nan
        fall_slope = np.nan
    
    return {
        'length': n,
        'mean': float(trace.mean()),
        'std': float(trace.std()),
        'max': float(trace.max()),
        'early_mean': float(early_mean),
        'late_mean': float(late_mean),
        'rise_slope': float(rise_slope),
        'fall_slope': float(fall_slope)
    }

