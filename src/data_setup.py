import pandas as pd
import os
import matplotlib.pyplot as plt


# Define columns based on the documentation
COLUMNS = [
    "machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
    "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"
]

def load_full_data_optimized(file_path):
    """
    Given the 9GB file size and 24GB of RAM, we must optimize memory usage.
    By specifying exact datatypes (downcasting), we can dramatically reduce the 
    memory footprint to safely fit into RAM.
    """
    # Optimized data types to reduce memory usage
    dtypes = {
        "machine_id": "category",        # category saves a lot of memory for repeated strings
        "time_stamp": "int64",           # epoch seconds
        "cpu_util_percent": "Int8",      # Int8 supports NaN and covers [0, 100]
        "mem_util_percent": "Int8",      # Int8 supports NaN and covers [0, 100]
        "mem_gps": "float32",            # float32 is sufficient
        "mkpi": "Int32",                 # Int32 supports NaN for integers
        "net_in": "float32",
        "net_out": "float32",
        "disk_io_percent": "float32"     # float32 is sufficient
    }
    
    print(f"Loading full dataset from {file_path} with optimized datatypes...")
    print("This might take a minute or two...")
    
    # We use engine='c' which is the default but just being explicit
    df = pd.read_csv(file_path, header=None, names=COLUMNS, dtype=dtypes)
    
    print("Successfully loaded full dataset.")
    print(f"Full Dataframe Shape: {df.shape}")
    print("Memory Usage:")
    df.info(memory_usage="deep")
    return df

if __name__ == "__main__":
    raw_data_path = "../data/raw/machine_usage.csv"
    
    # Adjust path if script is run from project root
    if not os.path.exists(raw_data_path):
        raw_data_path = "data/raw/machine_usage.csv"
    
    df_full = load_full_data_optimized(raw_data_path)

    # Get unique machine IDs
    unique_machine_ids = df_full["machine_id"].unique()
    print(f"Unique Machine IDs: {len(unique_machine_ids)}")
    print(f"Sample Machine IDs: {unique_machine_ids[:5]}")

    # Get summary statistics for # of cpu_util_percent records per machine
    cpu_util_counts = df_full.groupby("machine_id")["cpu_util_percent"].count()
    print("Summary Statistics for cpu_util_percent records per machine:")
    print(cpu_util_counts.describe())

    # Select 50 random machine IDs for the sample dataset
    sample_machine_ids = pd.Series(unique_machine_ids).sample(n=50, random_state=42).tolist()
    print(f"Selected Sample Machine IDs: {sample_machine_ids}")

    # Filter the full dataset to create the sample dataset with only the selected machine IDs, timestamps, and cpu_util_percent
    df_sample = df_full[df_full["machine_id"].isin(sample_machine_ids)][["machine_id", "time_stamp", "cpu_util_percent"]]
    print(f"Sample Dataframe Shape: {df_sample.shape}")
    print("Sample Dataframe Head:")
    print(df_sample.head())

    # Scale the cpu_util_percent values to be between 0 and 1
    df_sample["cpu_util_percent"] = df_sample["cpu_util_percent"] / 100.0

    # Save the sample dataset to a new CSV file
    sample_data_path = "data/processed/sample_machine_usage.csv"
    df_sample.to_csv(sample_data_path, index=False)
    print(f"Sample dataset saved to {sample_data_path}")

    # Plot the time series of cpu_util_percent for the first machine in the sample dataset
    first_machine_id = sample_machine_ids[0]
    df_first_machine = df_sample[df_sample["machine_id"] == first_machine_id]
    # first sort by time_stamp
    df_first_machine = df_first_machine.sort_values(by="time_stamp")
    plt.figure(figsize=(12, 6))
    plt.plot(df_first_machine["time_stamp"], df_first_machine["cpu_util_percent"], marker='o', linestyle='-', markersize=2)
    plt.title(f"CPU Utilization Over Time for Machine ID: {first_machine_id}")
    plt.xlabel("Time Stamp (Epoch Seconds)")
    plt.ylabel("CPU Utilization (Scaled 0-1)")
    plt.grid()
    plt.show()