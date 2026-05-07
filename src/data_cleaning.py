import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def preprocess_alibaba_data(input_file, output_folder):
    # Use Path for Mac/Windows compatibility
    input_path = Path(input_file)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Rigorous Cleaning: Sorting and Gap Detection 
    # Ensure data is sorted by machine and time to detect gaps accurately
    df = df.sort_values(by=['machine_id', 'time_stamp']).reset_index(drop=True)

    # Calculate the gap between consecutive timestamps for each machine
    # Note: If your timestamp is in seconds, a 2-minute gap is 120
    df['time_diff'] = df.groupby('machine_id')['time_stamp'].diff()

    # 2. Logical Segmentation 
    # Mark a new segment if the gap > 120 seconds (2 mins) or it's a new machine_id
    df['is_new_segment'] = (df['time_diff'] > 120) | (df['time_diff'].isna())
    
    # Create unique IDs for these segments (e.g., m_1932_seg_1, m_1932_seg_2)
    df['segment_count'] = df.groupby('machine_id')['is_new_segment'].cumsum().astype(int)
    df['logical_id'] = df['machine_id'].astype(str) + '_seg_' + df['segment_count'].astype(str)

    # Filter out segments that are too short to be useful for Path B modeling
    # Example: Discard segments with fewer than 60 consecutive data points
    min_length = 60
    counts = df['logical_id'].value_counts()
    keep_ids = counts[counts >= min_length].index
    df = df[df['logical_id'].isin(keep_ids)].copy()

    # 3. Feature Scaling: Standardize CPU to 0-1 
    scaler = MinMaxScaler()
    # Using 'cpu_util_percent' from your dataset preview
    df['cpu_scaled'] = scaler.fit_transform(df[['cpu_util_percent']])

    # 4. Strict Temporal Split (Preventing Data Leakage) 
    # Determine the global 80% cutoff point across the entire dataset's time range
    time_cutoff = df['time_stamp'].quantile(0.8)

    train_df = df[df['time_stamp'] <= time_cutoff].copy()
    test_df = df[df['time_stamp'] > time_cutoff].copy()

    # Make new dataframes with only the necessary columns for modeling
    train_df = train_df[['logical_id', 'time_stamp', 'cpu_scaled']]
    test_df = test_df[['logical_id', 'time_stamp', 'cpu_scaled']]

    # Save to CSV for the modeling phase
    train_df.to_csv(output_dir / "train_segments.csv", index=False)
    test_df.to_csv(output_dir / "test_segments.csv", index=False)

    print(f"Successfully created {df['logical_id'].nunique()} logical segments.")
    print(f"Saved training and test sets to {output_dir}.")

if __name__ == "__main__":
    # Update this to your local filename
    preprocess_alibaba_data("data/processed/sample_machine_usage.csv", "data/processed")