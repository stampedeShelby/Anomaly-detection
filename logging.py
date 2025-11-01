"""
Event Logging and Reporting Module

Generates event logs and anomaly reports for operators.
"""

import pandas as pd
from .config import CONFIG


def log_events(df, shape_df=None, if_summary=None, cfg=CONFIG):
    """
    Generate event log CSV with all detected anomalies.
    
    Event types:
    - medium_gap: info-level gaps (30s-5m)
    - shutdown_break: info-level shutdowns (≥1 day)
    - spike: critical spikes (z>5, >20% baseline, inrush-masked)
    - drift: warn/critical drift (EWMA/CUSUM)
    - shape_deviation: warn/critical cycle shape anomalies
    - iforest: warn/critical model-based anomalies
    
    Args:
        df: DataFrame with datetime, cycle_counter, and detector flags
        shape_df: DataFrame with shape_corr and shape_severity
        if_summary: DataFrame with worst_model_score and model_severity
        cfg: Configuration dictionary
    
    Returns:
        DataFrame: Event log with datetime, severity, anomaly_type, etc.
    """
    events = []
    
    def emit(row, atype, sev, note=''):
        """Helper to emit an event."""
        events.append({
            'datetime': row['datetime'],
            'date': row['datetime'].date() if hasattr(row['datetime'], 'date') else None,
            'machine_id': row.get('mc', None),
            'cycle_counter': row['cycle_counter'],
            'segment_id': row.get('cycle_segment_id', None),
            'anomaly_type': atype,
            'severity': sev,
            'persistence_sec': None,
            'notes': note,
            'log_level': sev
        })
    
    # Medium gaps and shutdowns
    if 'medium_gap' in df:
        for _, row in df[df['medium_gap']].iterrows():
            emit(row, 'medium_gap_30s_5m', 'info', '')
    
    if 'shutdown_break' in df:
        for _, row in df[df['shutdown_break']].iterrows():
            emit(row, 'shutdown_break', 'info', 'baseline reset')
    
    # Spikes and drift
    if 'spike_flag' in df:
        for _, row in df[df['spike_flag']].iterrows():
            emit(row, 'spike', 'critical', 'z>5 & >20% over baseline; inrush masked')
    
    if 'drift_flag' in df:
        for _, row in df[df['drift_flag']].iterrows():
            emit(row, 'drift', 'warn', 'EWMA/CUSUM')
    
    # Shape (cycle-level events)
    if shape_df is not None and 'shape_severity' in shape_df:
        for _, r in shape_df[shape_df['shape_severity'].notna()].iterrows():
            sev = r['shape_severity']
            cycle_id = r['cycle_counter']
            
            # Find first datetime for this cycle
            cycle_rows = df[df['cycle_counter'] == cycle_id]
            if len(cycle_rows) > 0:
                ts = cycle_rows['datetime'].iloc[0]
                events.append({
                    'datetime': ts,
                    'date': ts.date() if hasattr(ts, 'date') else None,
                    'machine_id': None,
                    'cycle_counter': cycle_id,
                    'segment_id': None,
                    'anomaly_type': 'shape_deviation',
                    'severity': sev,
                    'persistence_sec': None,
                    'notes': f"corr={r['shape_corr']:.3f}" if 'shape_corr' in r else '',
                    'log_level': sev
                })
    
    # Isolation Forest (cycle-level events)
    if if_summary is not None and 'model_severity' in if_summary:
        for _, r in if_summary[if_summary['model_severity'].notna()].iterrows():
            sev = r['model_severity']
            cycle_id = r['cycle_counter']
            
            # Find first datetime for this cycle
            cycle_rows = df[df['cycle_counter'] == cycle_id]
            if len(cycle_rows) > 0:
                ts = cycle_rows['datetime'].iloc[0]
                events.append({
                    'datetime': ts,
                    'date': ts.date() if hasattr(ts, 'date') else None,
                    'machine_id': None,
                    'cycle_counter': cycle_id,
                    'segment_id': None,
                    'anomaly_type': 'iforest',
                    'severity': sev,
                    'persistence_sec': None,
                    'notes': f"worst_score={r['worst_model_score']:.3f}" if 'worst_model_score' in r else '',
                    'log_level': sev
                })
    
    return pd.DataFrame(events)


def generate_anomaly_report(cycle_summary, top_n=20):
    """
    Generate concise anomaly report with top N critical cycles.
    
    Args:
        cycle_summary: DataFrame with cycle_counter, fused_severity, worst_model_score
        top_n: Number of top anomalies to include
    
    Returns:
        DataFrame: Ranked anomaly report
    """
    # Filter for fused critical cycles
    crit_cycles = cycle_summary.query("fused_severity == 'critical'").copy()
    
    # Sort by worst_model_score (ascending = more anomalous)
    crit_ranked = crit_cycles.sort_values('worst_model_score').head(top_n).reset_index(drop=True)
    
    # Add notes
    def make_note(row):
        if row['fused_severity'] == 'critical':
            return f"Critical anomaly detected (worst score={row['worst_model_score']:.3f})"
        elif row['fused_severity'] == 'warn':
            return f"Warning anomaly (borderline score={row['worst_model_score']:.3f})"
        else:
            return "No anomalies"
    
    crit_ranked['notes'] = crit_ranked.apply(make_note, axis=1)
    
    return crit_ranked


def overall_cycle_severity(events):
    """
    Compute overall per-cycle severity from event log.
    
    Args:
        events: DataFrame from log_events()
    
    Returns:
        DataFrame: cycle_counter and overall_severity
    """
    sev_map = {'critical': 2, 'warn': 1, 'info': 0, None: -1}
    
    # Take worst severity per cycle
    cycle_sev = (
        events.groupby('cycle_counter')['severity']
        .apply(lambda x: max(x, key=lambda s: sev_map.get(s, -1)))
        .reset_index()
        .rename(columns={'severity': 'overall_severity'})
    )
    
    return cycle_sev


def save_outputs(df, events, cycle_summary, output_dir='outputs', cfg=CONFIG):
    """
    Save event log and cycle summary to CSV files.
    
    Args:
        df: Main processed DataFrame
        events: Event log DataFrame
        cycle_summary: Cycle summary DataFrame
        output_dir: Output directory
        cfg: Configuration dictionary
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save event log
    if len(events) > 0:
        event_path = os.path.join(output_dir, 'event_log.csv')
        events.to_csv(event_path, index=False)
        print(f"Saved event log: {event_path} ({len(events)} events)")
    
    # Save cycle summary
    cycle_path = os.path.join(output_dir, 'cycle_summary.csv')
    cycle_summary.to_csv(cycle_path, index=False)
    print(f"Saved cycle summary: {cycle_path} ({len(cycle_summary)} cycles)")
    
    # Save anomaly report
    report = generate_anomaly_report(cycle_summary, top_n=20)
    report_path = os.path.join(output_dir, 'anomaly_report.csv')
    report.to_csv(report_path, index=False)
    print(f"Saved anomaly report: {report_path} ({len(report)} critical cycles)")
    
    return event_path, cycle_path, report_path

