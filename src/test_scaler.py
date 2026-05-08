import torch
import torch.nn as nn
from torch.amp import GradScaler

model = nn.Linear(10, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler(device='cuda', enabled=False)

x = torch.randn(10)
y = torch.randn(10)
loss = nn.functional.mse_loss(model(x), y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
print("Success")
