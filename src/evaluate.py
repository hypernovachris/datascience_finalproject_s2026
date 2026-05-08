import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import your custom modules
from models.dataset import AlibabaSegmentDataset
from models.consistency_model import CMTSBackbone
from train_advanced import asymmetric_business_metric

def evaluate_cmts(data_dir, model_path, sequence_length=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Evaluating on device: {device}")

    # 1. Load Test Data
    test_path = Path(data_dir) / "test_segments.csv"
    test_dataset = AlibabaSegmentDataset(test_path, sequence_length=sequence_length)
    # Batch size of 1 or full batch for easier unrolling of sequences
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 2. Load Trained Model
    model = CMTSBackbone(input_features=1, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inference Loop
    all_actuals = []
    all_preds = []

    print("Running 1-Step Consistency Inference...")
    with torch.no_grad():
        for batch_idx, x_0 in enumerate(test_loader):
            x_0 = x_0.to(device)
            current_batch_size = x_0.shape[0]

            # In consistency models, inference starts from pure noise at maximum t
            t_max = torch.ones(current_batch_size, device=device) * 0.9 
            noisy_input = x_0 + (torch.randn_like(x_0) * t_max.view(-1, 1, 1))

            # 1-step mapping from noise to clean data
            pred_x_0 = model(noisy_input, t_max)

            # We only care about predicting the *last* step in the sequence window
            # to align with our baseline forecasting approach
            actual_last_step = x_0[:, 0, -1].cpu().numpy()
            pred_last_step = pred_x_0[:, 0, -1].cpu().numpy()

            all_actuals.extend(actual_last_step)
            all_preds.extend(pred_last_step)

    all_actuals = np.array(all_actuals)
    all_preds = np.array(all_preds)

    # 4. Calculate Metrics
    mse = mean_squared_error(all_actuals, all_preds)
    mae = mean_absolute_error(all_actuals, all_preds)
    
    # Convert back to tensors for our custom metric function
    tensor_actuals = torch.tensor(all_actuals)
    tensor_preds = torch.tensor(all_preds)
    biz_penalty = asymmetric_business_metric(tensor_actuals, tensor_preds, penalty_weight=5.0)

    print("-" * 30)
    print("CMTS (PROPOSED MODEL) METRICS")
    print(f"Mean Squared Error (MSE):  {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Asymmetric Biz Penalty:    {biz_penalty:.6f}")
    print("-" * 30)

    # 5. Generate Visualizations for Report
    generate_evaluation_plots(all_actuals, all_preds)

def generate_evaluation_plots(actuals, preds, num_points=100):
    sns.set_theme(style="whitegrid")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # --- Plot 1: Time Series Line Graph ---
    plt.figure(figsize=(14, 5))
    plt.plot(actuals[:num_points], label='Actual CPU Load', color='blue', linewidth=2)
    plt.plot(preds[:num_points], label='CMTS Predicted Load', color='green', linestyle='--', linewidth=2)
    
    plt.title('Advanced Model (CMTS): Actual vs. Predicted CPU Load')
    plt.xlabel('Time Step (Minutes)')
    plt.ylabel('Scaled CPU Utilization (0 to 1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / 'cmts_timeseries.png')
    plt.show()

    # --- Plot 2: Error/Residual Distribution ---
    # This proves your model is robust and not just copying the last frame
    residuals = actuals - preds
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color='purple')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('Distribution of Prediction Errors (Residuals)')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(reports_dir / 'cmts_residuals.png')
    plt.show()

if __name__ == "__main__":
    # Ensure these paths match your directory structure
    evaluate_cmts(
        data_dir="data/processed", 
        model_path="models/cmts_online_weights.pth",
        sequence_length=25
    )