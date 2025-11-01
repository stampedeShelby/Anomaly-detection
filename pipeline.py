"""
Main Anomaly Detection Pipeline

End-to-end pipeline orchestrating all modules.
"""

import numpy as np
import pandas as pd
from .config import CONFIG
from .data_loader import load_and_preprocess_data, compute_gap_flags, get_cycle_trace
from .features import extract_segment_features, compute_shape_correlation, simple_shape_diagnostics
from .detectors import (
    apply_inrush_mask,
    compute_spike_flags,
    drift_ewma_cusum,
    tune_drift_thresholds
)
from .model import train_isolation_forest, create_cycle_summary
from .fusion import risk_fusion, fused_severity
from .logging import log_events, generate_anomaly_report, save_outputs


def run_pipeline(filepath, cfg=CONFIG, output_dir='outputs', verbose=True):
    """
    Run the complete anomaly detection pipeline.
    
    Pipeline steps:
    1. Load and preprocess data
    2. Extract segment features
    3. Compute shape correlation
    4. Apply detectors (spike, drift)
    5. Train Isolation Forest model
    6. Risk fusion
    7. Generate event log and reports
    
    Args:
        filepath: Path to input CSV
        cfg: Configuration dictionary
        output_dir: Directory for outputs
        verbose: Print progress messages
    
    Returns:
        dict: Results dictionary with all DataFrames and metrics
    """
    results = {}
    
    # === Phase 1: Data Loading ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 1: DATA LOADING & PREPROCESSING")
        print("=" * 80)
    
    df = load_and_preprocess_data(filepath, cfg)
    results['df'] = df
    
    # === Phase 2: Feature Extraction ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 2: FEATURE EXTRACTION")
        print("=" * 80)
    
    segment_features = extract_segment_features(df, cfg)
    results['segment_features'] = segment_features
    
    # === Phase 3: Shape Correlation ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 3: SHAPE CORRELATION DETECTOR")
        print("=" * 80)
    
    shape_df, shape_thresholds = compute_shape_correlation(df, cfg)
    results['shape_df'] = shape_df
    results['shape_thresholds'] = shape_thresholds
    
    if verbose:
        print("Shape correlation summary:", shape_df['shape_corr'].describe())
        print("Thresholds:", shape_thresholds)
        flagged = shape_df[shape_df['shape_severity'].notna()]
        print(f"Flagged cycles: {len(flagged)}")
        if len(flagged) > 0:
            print(flagged.head())
    
    # === Phase 4: Rule-Based Detectors ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 4: RULE-BASED DETECTORS")
        print("=" * 80)
    
    df = apply_inrush_mask(df, cfg)
    df = compute_spike_flags(df, cfg)
    df = compute_gap_flags(df, cfg)
    df = drift_ewma_cusum(df, cfg)
    results['df'] = df
    
    if verbose:
        if 'spike_flag' in df:
            print(f"Spikes detected: {df['spike_flag'].sum()}")
        if 'drift_flag' in df:
            print(f"Drift flags: {df['drift_flag'].sum()}")
    
    # === Phase 5: Isolation Forest Model ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 5: ISOLATION FOREST MODEL")
        print("=" * 80)
    
    segment_features, iforest, if_thresholds = train_isolation_forest(segment_features, cfg)
    results['iforest'] = iforest
    results['if_thresholds'] = if_thresholds
    
    cycle_summary = create_cycle_summary(segment_features, cfg)
    results['cycle_summary'] = cycle_summary
    
    # Map model scores back to df
    seg_score_map = segment_features.set_index('cycle_segment_id')['if_score']
    df['if_score'] = df['cycle_segment_id'].map(seg_score_map)
    
    # Convert to weights for risk fusion (0-1 normalized)
    warn_thr = if_thresholds['warn_thr']
    crit_thr = if_thresholds['crit_thr']
    df['if_weight'] = np.where(
        df['if_score'] <= crit_thr, 1.0,
        np.where(df['if_score'] <= warn_thr, 0.5, 0.0)
    )
    
    # === Phase 6: Risk Fusion ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 6: RISK FUSION")
        print("=" * 80)
    
    df = risk_fusion(df, shape_df=shape_df, if_scores=df['if_weight'], cfg=cfg)
    results['df'] = df
    
    if verbose:
        print("Risk distribution:")
        print(df[['risk', 'risk_smoothed']].describe())
    
    # === Phase 7: Drift Tuning (Optional) ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 7: DRIFT TUNING")
        print("=" * 80)
    
    drift_tuned_map = tune_drift_thresholds(df, cycle_summary, cfg)
    results['drift_tuned_map'] = drift_tuned_map
    
    # === Phase 8: Multi-Detector Fusion ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 8: MULTI-DETECTOR FUSION")
        print("=" * 80)
    
    flags = fused_severity(df, shape_df, cycle_summary, cfg)
    results['flags'] = flags
    
    # Merge back into cycle_summary
    cycle_summary = cycle_summary.merge(flags[['cycle_counter', 'fused_severity']], on='cycle_counter', how='left')
    results['cycle_summary'] = cycle_summary
    
    if verbose:
        print("Severity distribution (original model):")
        print(flags['model_severity'].value_counts())
        print("\nSeverity distribution (fused):")
        print(flags['fused_severity'].value_counts())
    
    # === Phase 9: Event Logging ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 9: EVENT LOGGING & REPORTING")
        print("=" * 80)
    
    events = log_events(df, shape_df=shape_df, if_summary=cycle_summary, cfg=cfg)
    results['events'] = events
    
    report = generate_anomaly_report(cycle_summary, top_n=20)
    results['report'] = report
    
    if verbose:
        print(f"Total events logged: {len(events)}")
        print("Event breakdown:")
        print(events['anomaly_type'].value_counts())
        
        print(f"\nTop 10 most anomalous critical cycles:")
        print(report[['cycle_counter', 'worst_model_score', 'notes']].head(10))
    
    # === Phase 10: Save Outputs ===
    if verbose:
        print("\n" + "=" * 80)
        print("PHASE 10: SAVE OUTPUTS")
        print("=" * 80)
    
    event_path, cycle_path, report_path = save_outputs(df, events, cycle_summary, output_dir, cfg)
    results['paths'] = {
        'event_log': event_path,
        'cycle_summary': cycle_path,
        'anomaly_report': report_path
    }
    
    # === Final Summary ===
    if verbose:
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Total cycles analyzed: {len(cycle_summary)}")
        print(f"Critical cycles: {(cycle_summary['fused_severity'] == 'critical').sum()}")
        print(f"Warn cycles: {(cycle_summary['fused_severity'] == 'warn').sum()}")
        print(f"Info cycles: {(cycle_summary['fused_severity'] == 'info').sum()}")
        print(f"Normal cycles: {(cycle_summary['fused_severity'].isna()).sum()}")
        print("=" * 80)
    
    return results


def inspect_cycle(df, cycle_id, cfg=CONFIG, verbose=True):
    """
    Inspect a specific cycle with full diagnostics.
    
    Args:
        df: Processed DataFrame
        cycle_id: cycle_counter value
        cfg: Configuration dictionary
        verbose: Print diagnostics
    
    Returns:
        dict: Diagnostic results
    """
    trace = get_cycle_trace(df, cycle_id, cfg)
    diag = simple_shape_diagnostics(trace)
    
    if verbose:
        print(f"\nCycle {cycle_id} Diagnostics:")
        for k, v in diag.items():
            if pd.isna(v):
                print(f"  {k}: None")
            elif isinstance(v, (float, np.floating)):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
    
    return diag


def summarize_cycles(df, cycle_ids, label, cfg=CONFIG):
    """
    Summarize multiple cycles for comparison.
    
    Args:
        df: Processed DataFrame
        cycle_ids: List of cycle_counter values
        label: Label for these cycles ('critical', 'warn', 'info', etc.)
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Summary statistics
    """
    summaries = []
    for cid in cycle_ids:
        trace = get_cycle_trace(df, cid, cfg)
        summaries.append({
            'cycle': cid,
            'label': label,
            'length': len(trace),
            'mean': float(trace.mean()),
            'std': float(trace.std()),
            'min': float(trace.min()),
            'max': float(trace.max()),
        })
    return pd.DataFrame(summaries)

