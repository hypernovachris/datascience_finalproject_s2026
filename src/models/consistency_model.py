import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------------
# 1. Time Embedding (Crucial for Consistency/Diffusion)
# ---------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Injects information about the current 'noise level' (t) into the model.
    The model needs to know how much noise it is trying to resolve.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ---------------------------------------------------------
# 2. The Dilated ResNet Block
# ---------------------------------------------------------
class ResidualConv1DBlock(nn.Module):
    """
    A 1D Convolutional block with a residual (skip) connection.
    Uses dilation to expand the receptive field over the time-series window.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, dilation=1):
        super().__init__()
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # First convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, out_channels)

        # dropout layer
        self.dropout = nn.Dropout(0.1)
        
        # Second convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Skip connection adjustment if channel dimensions change
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        # Initial transformation
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h) # Swish activation performs very well in diffusion/consistency models
        
        # Inject time embeddings
        # t_emb shape: (batch_size, time_emb_dim) -> unsqueeze to match 1D conv shape
        time_emb = self.time_mlp(t_emb).unsqueeze(-1)
        h = h + time_emb
        
        # Second transformation
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Add residual skip connection
        return h + self.shortcut(x)

# ---------------------------------------------------------
# 3. The Main CMTS Architecture
# ---------------------------------------------------------
class CMTSBackbone(nn.Module):
    """
    The core Consistency Model for Time Series.
    Maps noisy CPU telemetry (x_t) directly to clean telemetry (x_0).
    """
    def __init__(self, input_features=1, hidden_dim=64, time_emb_dim=128):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Project raw input to hidden dimensions
        self.init_conv = nn.Conv1d(input_features, hidden_dim, kernel_size=1)
        
        # Stack of dilated residual blocks
        # Dilations of 1, 2, 4 allow the model to see long-term dependencies in the CPU load
        self.res_blocks = nn.ModuleList([
            ResidualConv1DBlock(hidden_dim, hidden_dim, time_emb_dim, dilation=1),
            ResidualConv1DBlock(hidden_dim, hidden_dim, time_emb_dim, dilation=2),
            ResidualConv1DBlock(hidden_dim, hidden_dim, time_emb_dim, dilation=4)
        ])
        
        # Final projection back to the predicted CPU load
        self.final_conv = nn.Conv1d(hidden_dim, input_features, kernel_size=1)

    def forward(self, x, time_steps):
        """
        x shape: (Batch_Size, Features, Sequence_Length)
        time_steps shape: (Batch_Size,)
        """
        # Get time embeddings
        t_emb = self.time_mlp(time_steps)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Pass through ResNet blocks
        for block in self.res_blocks:
            h = block(h, t_emb)
            
        # Raw network output
        f_theta = self.final_conv(h)
        
        # Enforce consistency model boundary condition: 
        # Output = c_skip(t) * x + c_out(t) * f_theta(x, t)
        sigma_data = 1
        c_skip = (sigma_data ** 2) / (time_steps ** 2 + sigma_data ** 2)
        c_out = (time_steps * sigma_data) / torch.sqrt(time_steps ** 2 + sigma_data ** 2)
        
        c_skip = c_skip.view(-1, 1, 1)
        c_out = c_out.view(-1, 1, 1)
        
        return c_skip * x + c_out * f_theta

# ---------------------------------------------------------
# 4. Consistency Loss Function Helper
# ---------------------------------------------------------
def consistency_loss(model_active, model_target, x_batch, t_n, t_n_plus_1, noise_n, noise_n_plus_1):
    """
    Calculates the Consistency Loss.
    Enforces that predictions from adjacent noise steps point to the same origin.
    
    In your report, cite this as:
    L(theta) = E[ || f_theta(x_{t_{n+1}}, t_{n+1}) - f_theta_target(x_{t_n}, t_n) ||^2 ]
    """
    # Create noisy inputs based on the two different time steps
    x_t_n = x_batch + noise_n
    x_t_n_plus_1 = x_batch + noise_n_plus_1
    
    # Active model predicts origin from t_{n+1}
    pred_active = model_active(x_t_n_plus_1, t_n_plus_1)
    
    # Target model (EMA weights) predicts origin from t_n
    with torch.no_grad(): # Target model is not updated via backprop
        pred_target = model_target(x_t_n, t_n)
        
    # Mean Squared Error between the two predictions
    loss = F.mse_loss(pred_active, pred_target)
    return loss