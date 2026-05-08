import time
import torch
from models.dataset import AlibabaSegmentDataset
from torch.utils.data import DataLoader

print("Loading dataset...")
t0 = time.time()
dataset = AlibabaSegmentDataset("../data/processed/train_segments.csv", sequence_length=25)
print(f"Dataset loaded in {time.time()-t0:.2f}s")

loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

print("Testing dataloader...")
t1 = time.time()
for i, batch in enumerate(loader):
    if i == 10:
        break
print(f"Dataloader 10 batches in {time.time()-t1:.2f}s")

loader_fast = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
t2 = time.time()
for i, batch in enumerate(loader_fast):
    if i == 10:
        break
print(f"Dataloader_fast 10 batches in {time.time()-t2:.2f}s")
