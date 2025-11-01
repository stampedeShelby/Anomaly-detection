"""
Example Usage Script

Demonstrates how to use the Motor Anomaly Detection pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor_anomaly_detection import run_pipeline, inspect_cycle, summarize_cycles
import pandas as pd

# ===================================================================
# Example 1: Run complete pipeline on your CSV file
# ===================================================================

if __name__ == '__main__':
    # Path to your data file
    # NOTE: Update this to point to your actual CSV file
    # For demo, using synthetic data:
    data_file = str(Path(__file__).parent / 'demo_motor_data.csv')
    # Original: r'C:\Users\274005\Downloads\phase 0 check\DL1_Current Data.csv'
    
    try:
        # Run the complete pipeline
        results = run_pipeline(
            filepath=data_file,
            output_dir='outputs',
            verbose=True
        )
        
        # Access results
        df = results['df']
        events = results['events']
        cycle_summary = results['cycle_summary']
        report = results['report']
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nTotal cycles analyzed: {len(cycle_summary)}")
        print(f"Critical cycles: {(cycle_summary['fused_severity'] == 'critical').sum()}")
        print(f"Warn cycles: {(cycle_summary['fused_severity'] == 'warn').sum()}")
        print(f"Info cycles: {(cycle_summary['fused_severity'] == 'info').sum()}")
        print(f"Normal cycles: {(cycle_summary['fused_severity'].isna()).sum()}")
        
        # Inspect top critical cycles
        print("\n" + "=" * 80)
        print("TOP 10 CRITICAL CYCLES")
        print("=" * 80)
        print(report[['cycle_counter', 'worst_model_score', 'notes']].head(10))
        
        # ===================================================================
        # Example 2: Inspect specific cycles
        # ===================================================================
        
        print("\n" + "=" * 80)
        print("CYCLE INSPECTION EXAMPLE")
        print("=" * 80)
        
        # Pick cycles to inspect
        critical_cycles = [1060.0, 696.0, 1227.0]
        normal_cycles = [14.0, 15.0, 200.0]
        
        # Inspect critical cycles
        for cid in critical_cycles[:2]:  # First 2 only
            inspect_cycle(df, cycle_id=cid, verbose=True)
        
        # ===================================================================
        # Example 3: Compare cycles
        # ===================================================================
        
        print("\n" + "=" * 80)
        print("CYCLE COMPARISON")
        print("=" * 80)
        
        comparison = pd.concat([
            summarize_cycles(df, critical_cycles[:3], 'critical'),
            summarize_cycles(df, normal_cycles[:3], 'normal')
        ])
        
        print(comparison)
        
        # ===================================================================
        # Example 4: Event log analysis
        # ===================================================================
        
        print("\n" + "=" * 80)
        print("EVENT LOG BREAKDOWN")
        print("=" * 80)
        print(f"Total events: {len(events)}")
        print("\nBy anomaly type:")
        print(events['anomaly_type'].value_counts())
        print("\nBy severity:")
        print(events['severity'].value_counts())
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 80)
        print("\nOutput files saved to 'outputs/' directory:")
        print("  - event_log.csv")
        print("  - cycle_summary.csv")
        print("  - anomaly_report.csv")
        print("=" * 80)
        
    except FileNotFoundError:
        print("\n" + "=" * 80)
        print("ERROR: Data file not found")
        print("=" * 80)
        print(f"Expected file: {data_file}")
        print("\nPlease update the 'data_file' variable in this script")
        print("to point to your actual CSV file.")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: Pipeline execution failed")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()

