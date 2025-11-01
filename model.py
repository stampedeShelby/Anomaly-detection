"""
Model-Based Anomaly Detection Module

Implements Isolation Forest for multivariate anomaly detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .config import CONFIG


def train_isolation_forest(segment_features, cfg=CONFIG):
    """
    Train Isolation Forest model on segment features.
    
    Args:
        segment_features: DataFrame with extracted features
        cfg: Configuration dictionary
    
    Returns:
        tuple: (segment_features, iforest, thresholds)
    """
    if_cfg = cfg['model_iforest']
    final_features = if_cfg['features_final']
    
    # Prepare feature matrix
    X_final = segment_features[final_features].copy()
    
    # Handle any NaN values
    if X_final.isna().any().any():
        print(f"WARNING: Found NaN in features. Columns: {X_final.columns[X_final.isna().any()].tolist()}")
        X_final = X_final.fillna(X_final.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # Fit Isolation Forest
    iforest = IsolationForest(
        n_estimators=200,
        contamination=if_cfg['contamination'],
        random_state=42
    )
    iforest.fit(X_scaled)
    
    # Compute scores (higher = more normal, lower = more anomalous)
    segment_features['if_score'] = iforest.decision_function(X_scaled)
    
    # Thresholds
    warn_thr = np.quantile(segment_features['if_score'], if_cfg['warn_quantile'])
    crit_thr = np.quantile(segment_features['if_score'], if_cfg['crit_quantile'])
    
    print("=" * 60)
    print("ISOLATION FOREST RESULTS")
    print("=" * 60)
    print(f"IF thresholds: warn < {warn_thr:.4f}, critical < {crit_thr:.4f}")
    
    # Assign severities
    segment_features['if_severity'] = np.where(
        segment_features['if_score'] <= crit_thr, 'critical',
        np.where(segment_features['if_score'] <= warn_thr, 'warn', None)
    )
    
    # Diagnostics
    stats = segment_features['if_score'].describe()
    print("if_score stats:", stats.to_dict())
    
    label_counts = segment_features['if_severity'].value_counts(dropna=False).to_dict()
    print("Label counts:", label_counts)
    
    quantiles = segment_features['if_score'].quantile([0.10, 0.50]).to_dict()
    print("if_score q10/q50:", quantiles)
    print("=" * 60)
    
    return segment_features, iforest, {'warn_thr': warn_thr, 'crit_thr': crit_thr}


def create_cycle_summary(segment_features, cfg=CONFIG):
    """
    Aggregate segment-level model scores to cycle-level summary.
    
    Args:
        segment_features: DataFrame with if_score and cycle_counter
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Cycle-level summary with worst/mean/std scores and severity
    """
    cycle_summary = segment_features.groupby('cycle_counter').agg(
        worst_model_score=('if_score', 'min'),
        mean_model_score=('if_score', 'mean'),
        std_model_score=('if_score', 'std'),
        frac_anomalous_segments=('if_severity', lambda x: (x.notna()).mean()),
        model_severity=('if_severity', lambda x: 'critical' if (x == 'critical').any()
                        else ('warn' if (x == 'warn').any() else None))
    ).reset_index()
    
    return cycle_summary

