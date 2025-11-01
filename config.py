"""
Motor Current Anomaly Detection - Configuration Module

Central configuration dictionary consolidating all thresholds, parameters, and policies
for the anomaly detection pipeline.

Based on validation from October 31, 2025 analysis.
"""

import numpy as np

CONFIG = {
    # Context
    'data_schema': ['datetime', 'mc', 'cycle_counter', 'date', 'current'],
    'notes': {
        'body_build_id': 'Not available in testing; multiple machines feed one motor stream',
        'machine_id': 'Optional; use mc if present, else None',
        'cycle_duration_sec': 'Typical cycles ~50–61 s; shutdowns produce very long segments',
        'shape_corr_baseline': 'Mean ~0.30; warn threshold ~0.19 (p10), critical ~0.16 (p05)'
    },

    # Cadence and segmentation
    'sampling_sec': 0.5,                 # observed median dt_diff ≈ 0.64 s
    'cycle_boundary': 'cycle_counter',   # new cycle when cycle_counter changes
    'shutdown_threshold_sec': 24*3600,   # ≥ 1 day = shutdown; reset drift baselines
    'medium_gap_range_sec': [30, 300],   # log as info; not anomalies

    # Missingness and zeros
    'nan_policy': 'log_then_drop',       # do not impute; keep datetime for audit
    'zeros_policy': 'expected_idle',     # not anomalies; use zero-run for dropout if needed

    # Inrush masking (spikes)
    'inrush_mask_sec': 5.0,              # inrush occurs 3–5 s; mask first 5 s from spike detection
    'spike': {
        'z_threshold': 5.0,              # z-score > 5
        'pct_over_baseline': 0.20,       # >20% above cycle baseline mean
        'min_duration_sec': 1.0,         # persistence: ≥ 1 s continuous
    },

    # Dropout (low priority)
    'dropout': {
        'enabled': False,                # tracked minimally but excluded from risk
        'near_zero_threshold_amp': 0.5,  # define "near zero" if enabled
        'min_duration_sec': 2.0,         # persistence: ≥ 2 s
        'risk_weight': 0.0,
    },

    # Drift detection (EWMA + CUSUM)
    'drift': {
        'ewma_alpha': 0.1,
        'cusum_k': 0.5,
        'cusum_h': 5.0,
        'window_sec': 60.0,              # reference horizon for context
        'reset_on_shutdown': True,
        'risk_weight': 0.3,
    },

    # Shape deviation (cycle-normalized correlation to rolling template)
    'shape': {
        'normalize_points': 100,
        'template_window_cycles': 20,     # rolling mean template
        'warn_quantile': 0.10,            # corr ≤ p10 → warn (~0.19 in this dataset)
        'crit_quantile': 0.05,            # corr ≤ p05 → critical (~0.16 in this dataset)
        'risk_weight': 0.2,
    },

    # Model-based detection (Isolation Forest)
    'model_iforest': {
        'features_final': [
            'mean', 'std', 'rise_time', 'skew', 'kurtosis',
            'fft_energy', 'duration_sec', 'max'
        ],
        'scale': 'StandardScaler',
        'contamination': 0.01,
        'warn_quantile': 0.50,  # median for warn
        'crit_quantile': 0.10,  # 10th percentile for critical
    },

    # Risk fusion
    'risk': {
        'weights': {
            'spike': 0.6,
            'drift': 0.3,
            'shape': 0.2,
            'model': 0.1,
            'dropout': 0.0
        },
        'smoothing_sec': 5.0,
        'alert_threshold': 0.3,
        'spike_override': True  # spikes always set risk=1.0
    },

    # Alert budget
    'alert_budget': {
        'max_per_hour': 5,
        'aggregate_on_exceed': True,  # emit main alert when exceeded
        'notify_supervisors': True
    },

    # Logging schema
    'logging': {
        'event_log_columns': [
            'datetime', 'date', 'machine_id', 'cycle_counter', 'segment_id',
            'anomaly_type', 'severity', 'persistence_sec', 'notes'
        ],
        'cycle_summary_columns': [
            'cycle_counter', 'overall_severity', 'worst_model_score',
            'detectors_triggered', 'shape_corr', 'notes'
        ]
    }
}

# Convenience function to convert seconds to samples
def sec_to_samples(seconds, sampling_sec=None):
    """Convert seconds to number of samples based on sampling interval."""
    if sampling_sec is None:
        sampling_sec = CONFIG['sampling_sec']
    return int(np.ceil(seconds / sampling_sec))

