"""
Create synthetic demo data for testing the pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Generate 1 day of synthetic motor data
print("Creating synthetic motor current data...")

# Parameters
n_samples = 2000  # ~16 minutes at 0.5s sampling
n_cycles = 30

# Timestamps
date_rng = pd.date_range(start='2025-10-01 08:00:00', periods=n_samples, freq='500ms')

# Generate synthetic cycles
current_values = []
cycle_ids = []

for cycle in range(n_cycles):
    cycle_length = n_samples // n_cycles
    start_idx = cycle * cycle_length
    
    # Normal cycle: sine wave with noise
    t = np.linspace(0, 4*np.pi, cycle_length)
    cycle_current = 10 + 5 * np.sin(t) + np.random.normal(0, 0.5, cycle_length)
    
    # Add some anomalies
    if cycle == 5:  # Cycle 5: spike
        cycle_current[cycle_length//2] = 50  # Huge spike
        
    if cycle == 10:  # Cycle 10: drift
        cycle_current = cycle_current + np.linspace(0, 3, cycle_length)
        
    if cycle == 15:  # Cycle 15: different shape
        cycle_current = 10 + 3 * np.sin(2*t) + np.random.normal(0, 1, cycle_length)
    
    current_values.extend(cycle_current)
    cycle_ids.extend([cycle] * cycle_length)

# Create DataFrame
df = pd.DataFrame({
    'datetime': date_rng[:len(current_values)],
    'mc': 'DL1',
    'cycle_counter': cycle_ids,
    'date': date_rng[:len(current_values)].date,
    'current': current_values
})

# Save
output_file = 'demo_motor_data.csv'
df.to_csv(output_file, index=False)

print(f"Created {output_file}")
print(f"  - {len(df)} rows")
print(f"  - {df['cycle_counter'].nunique()} cycles")
print(f"  - Time range: {df['datetime'].min()} to {df['datetime'].max()}")
print()
print("Anomalies injected:")
print("  - Cycle 5: Large spike")
print("  - Cycle 10: Drift upward")
print("  - Cycle 15: Different shape")
print()
print(f"Run: py example_usage.py (update path to: {output_file})")

