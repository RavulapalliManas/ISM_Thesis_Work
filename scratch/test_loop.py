import torch
import torch.nn as nn
import torch.optim as optim
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Testing on {device}")

model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for i in range(100):
    start = time.time()
    x = torch.randn(32, 10).to(device)
    y = torch.randn(32, 10).to(device)
    
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Step {i} | Loss: {loss.item():.4f} | Time: {time.time() - start:.4f}s")
