import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# Define columns based on the documentation

def load_full_data_optimized(file_path):

    
    # We use engine='c' which is the default but just being explicit
    df = pd.read_csv(file_path, header=0)

    return df


if __name__ == "__main__":
    raw_data_path = Path("data") / "processed" / "sample_machine_usage.csv"
    
    df_full = load_full_data_optimized(raw_data_path)

    # 1. Sort by machine_id AND time_stamp to ensure diffs are calculated chronologically per machine
    df_full = df_full.sort_values(by=["machine_id", "time_stamp"])

    # 2. Calculate the difference directly on the grouped column and assign it to the main DataFrame
    df_full["time_gap"] = df_full.groupby("machine_id")["time_stamp"].diff().fillna(0)
    
    # NOTE: If your time_stamp is a datetime object, fillna(0) will fail. 
    # Use fillna(pd.Timedelta(seconds=0)) instead.

    # 3. Now loop through to print the summary statistics
    for machine_id, group in df_full.groupby("machine_id"):
        total_gaps = len(group)
        
        # If time_gap is in seconds (numeric):
        gaps_over_1_minute = (group["time_gap"] > 60).sum()
        
        # IF time_gap is a timedelta object, use this instead:
        # gaps_over_1_minute = (group["time_gap"] > pd.Timedelta(seconds=60)).sum()
        
        percentage = (gaps_over_1_minute / total_gaps) * 100
        print(f"Machine ID: {machine_id} - {percentage:.2f}% of time gaps are greater than 1 minute")