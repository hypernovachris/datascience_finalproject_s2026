import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class AlibabaSegmentDataset(Dataset):
    """
    Custom PyTorch Dataset for segmented Alibaba CPU telemetry.
    Generates sequence windows for the Consistency Model without crossing segment boundaries.
    """
    def __init__(self, csv_file, sequence_length=25, feature_col='cpu_scaled'):
        """
        Args:
            csv_file (str or Path): Path to the processed train/test CSV.
            sequence_length (int): How many minutes of data make up one sample window.
            feature_col (str): The name of the column containing the scaled CPU load.
        """
        super().__init__()
        self.sequence_length = sequence_length
        
        # Load data using pathlib for cross-platform safety
        print(f"Loading dataset from {csv_file}...")
        df = pd.read_csv(Path(csv_file))
        
        # Ensure data is sorted temporally within each segment
        df = df.sort_values(by=['logical_id', 'time_stamp']).reset_index(drop=True)
        
        self.samples = []
        
        # Group by logical_id to strictly isolate segments
        grouped = df.groupby('logical_id')
        
        for segment_id, group in grouped:
            # Extract the raw values as a numpy array for faster slicing
            values = group[feature_col].values
            
            # If a segment is shorter than our required window, we must skip it
            if len(values) < sequence_length:
                continue
                
            # Create valid sliding windows for this specific segment
            # e.g., for sequence_length=25, valid starts are 0 to (len - 5)
            for i in range(len(values) - sequence_length + 1):
                window = values[i : i + sequence_length]
                self.samples.append(window)
                
        # Convert the list of arrays into a single, contiguous PyTorch tensor
        # Shape will be (Total_Valid_Windows, Sequence_Length)
        self.data_tensor = torch.tensor(np.array(self.samples), dtype=torch.float32)
        
        print(f"Dataset initialized: {len(self.data_tensor)} total windows of length {sequence_length}.")

    def __len__(self):
        """Returns the total number of valid sequence windows."""
        return len(self.data_tensor)

    def __getitem__(self, idx):
        """
        Retrieves a single window of data.
        Returns shape: (Features, Sequence_Length) -> (1, sequence_length)
        The 1D Convolutional network requires the feature channel dimension.
        """
        window = self.data_tensor[idx]
        
        # Add the channel dimension (Features = 1 for univariate CPU load)
        # Shape changes from (sequence_length,) to (1, sequence_length)
        x_0 = window.unsqueeze(0) 
        
        # For a Consistency Model mapping noise back to clean data, 
        # the target 'y' is the exact same clean window 'x_0'.
        # The training loop will dynamically add noise to x_0 to create the input.
        return x_0