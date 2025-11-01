"""
Motor Current Anomaly Detection Pipeline

Production-ready unsupervised anomaly detection system for conveyor motor fault detection.

Phases:
- Phase 0-3: Data prep, features, baseline detectors
- Phase 4: Isolation Forest model
- Phase 5: Risk fusion
- Phase 6: Evaluation (requires labels)
- Phase 7: Visualization
- Phase 8-9: Deployment, MLOps
"""

__version__ = "0.1.0"

from .config import CONFIG, sec_to_samples
from .data_loader import load_and_preprocess_data, get_cycle_trace, per_cycle_stat
from .features import extract_segment_features, compute_shape_correlation, normalize_cycles, simple_shape_diagnostics
from .detectors import (
    apply_inrush_mask,
    compute_spike_flags,
    drift_ewma_cusum,
    compute_gap_flags,
    tune_drift_thresholds
)
from .model import train_isolation_forest, create_cycle_summary
from .fusion import risk_fusion, fused_severity
from .logging import log_events, generate_anomaly_report, overall_cycle_severity, save_outputs
from .pipeline import run_pipeline, inspect_cycle, summarize_cycles

__all__ = [
    'CONFIG',
    'sec_to_samples',
    'load_and_preprocess_data',
    'compute_gap_flags',
    'get_cycle_trace',
    'per_cycle_stat',
    'extract_segment_features',
    'compute_shape_correlation',
    'normalize_cycles',
    'simple_shape_diagnostics',
    'apply_inrush_mask',
    'compute_spike_flags',
    'drift_ewma_cusum',
    'tune_drift_thresholds',
    'train_isolation_forest',
    'create_cycle_summary',
    'risk_fusion',
    'fused_severity',
    'log_events',
    'generate_anomaly_report',
    'overall_cycle_severity',
    'save_outputs',
    'run_pipeline',
    'inspect_cycle',
    'summarize_cycles'
]

