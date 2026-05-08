import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

# Import your custom modules
from models.dataset import AlibabaSegmentDataset
from models.consistency_model import CMTSBackbone, consistency_loss

# ---------------------------------------------------------
# 1. Exponential Moving Average (EMA) Helper
# ---------------------------------------------------------
def update_ema_target(online_model, target_model, decay=0.99):
    """
    Updates the target network weights using an exponential moving average.
    """
    with torch.no_grad():
        for online_param, target_param in zip(online_model.parameters(), target_model.parameters()):
            # target = decay * target + (1 - decay) * online
            target_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)

def initialize_target_model(online_model, target_model):
    """Copies exact weights from online to target at the start of training."""
    target_model.load_state_dict(online_model.state_dict())
    # Target model should not compute gradients
    for param in target_model.parameters():
        param.requires_grad = False

# ---------------------------------------------------------
# 2. Custom Business-Value Metric
# ---------------------------------------------------------
def asymmetric_business_metric(y_true, y_pred, penalty_weight=5.0):
    """
    Heavily penalizes under-predicting CPU spikes (which causes server crashes).
    Standard MSE is applied to over-predictions (which just wastes electricity).
    """
    diff = y_true - y_pred
    
    # If diff > 0, true > pred (under-prediction -> Apply penalty)
    # If diff <= 0, true <= pred (over-prediction -> Standard weight)
    weights = torch.where(diff > 0, penalty_weight, 1.0)
    
    loss = weights * (diff ** 2)
    return loss.mean().item()

# ---------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------
def train_consistency_model(data_dir, epochs=10, batch_size=64, sequence_length=25):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps") # For your Mac M-series chip
    print(f"Training on device: {device}")

    # 1. Initialize Data
    train_path = Path(data_dir) / "train_segments.csv"
    train_dataset = AlibabaSegmentDataset(train_path, sequence_length=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # 2. Initialize Models
    # Features = 1 (CPU load)
    online_model = CMTSBackbone(input_features=1, hidden_dim=64).to(device)
    target_model = CMTSBackbone(input_features=1, hidden_dim=64).to(device)
    
    initialize_target_model(online_model, target_model)
    target_model.eval() # Target model is always in eval mode

    # 3. Optimizer
    # AdamW is robust for diffusion/consistency models
    optimizer = optim.AdamW(online_model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Initialize the AMP GradScaler
    # We conditionally enable it only if we are using an NVIDIA GPU (CUDA)
    # This ensures the code won't crash when you test it on your Mac!
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)

    # 4. Training Loop
    print("\nStarting Training...")
    # 4. Training Loop
    print("\nStarting Training...")
    for epoch in range(epochs):
        online_model.train()
        total_c_loss = 0.0
        total_biz_loss = 0.0
        
        for batch_idx, x_0 in enumerate(train_loader):
            x_0 = x_0.to(device)
            current_batch_size = x_0.shape[0]

            # Generate random noise levels for consistency training
            # We sample t_n from [0.1, 0.9] and t_n_plus_1 as slightly more noisy
            t_n = torch.rand(current_batch_size, device=device) * 0.8 + 0.1
            t_n_plus_1 = t_n + 0.05 
            
            # Generate random gaussian noise
            noise_n = (torch.randn_like(x_0) * 0.5) * t_n.view(-1, 1, 1)
            noise_n_plus_1 = (torch.randn_like(x_0) * 0.5) * t_n_plus_1.view(-1, 1, 1)

            # Forward Pass & Consistency Loss Calculation
            optimizer.zero_grad()
            
            # Wrap the forward pass in autocast
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                c_loss = consistency_loss(
                    online_model, target_model, 
                    x_0, t_n, t_n_plus_1, noise_n, noise_n_plus_1
                )

            # Backpropagation using the Scaler
            scaler.scale(c_loss).backward()
            
            # CRITICAL: You must unscale the gradients BEFORE clipping them!
            # Otherwise, you are clipping the artificially scaled-up gradients.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(online_model.parameters(), max_norm=1.0)
            
            # Step the optimizer and update the scaler for the next batch
            scaler.step(optimizer)
            scaler.update()

            # Update Target Model via EMA
            update_ema_target(online_model, target_model)

            # Calculate our custom metric for monitoring
            # For logging, we see how well the model reconstructs the clean data
            with torch.no_grad():
                pred_x_0 = online_model(x_0 + noise_n, t_n)
                biz_loss = asymmetric_business_metric(x_0, pred_x_0, penalty_weight=5.0)

            total_c_loss += c_loss.item()
            total_biz_loss += biz_loss
            
        # Epoch Summary
        avg_c_loss = total_c_loss / len(train_loader)
        avg_biz_loss = total_biz_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Consistency Loss: {avg_c_loss:.6f} | Asymmetric Biz Penalty: {avg_biz_loss:.6f}")

    # Save the trained model
    save_path = Path("models")
    save_path.mkdir(exist_ok=True)
    torch.save(online_model.state_dict(), save_path / "cmts_online_weights.pth")
    print(f"\nTraining complete. Model saved to {save_path / 'cmts_online_weights.pth'}")

if __name__ == "__main__":
    # Ensure this points to the directory containing train_segments.csv
    train_consistency_model(data_dir="data/processed", epochs=10, batch_size=64, sequence_length=25)