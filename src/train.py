from torch_geometric.loader import DataLoader
from gnn_model import GNNDelayRegressor
from graph_dataset import NetlistGraphDataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Dataset and loader
dataset = NetlistGraphDataset(num_graphs=200)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model setup
model = GNNDelayRegressor(in_channels=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(1, 101):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch.to('cpu')  # or 'cuda' if using GPU
        optimizer.zero_grad()
        pred = model(batch).view(-1)           # (batch_size,)
        target = batch.y.view(-1)              # (batch_size,)
        loss = loss_fn(pred, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")

# ====================
# Evaluation + Plotting
# ====================
model.eval()
true_vals = []
pred_vals = []

for batch in train_loader:
    with torch.no_grad():
        pred = model(batch)
        pred = pred.view(-1)
        target = batch.y.view(-1)

    pred_vals.extend(pred.tolist())
    true_vals.extend(target.tolist())
# Plot: True vs Predicted Delay
plt.figure(figsize=(7, 5))
plt.scatter(true_vals, pred_vals, alpha=0.7)
plt.xlabel("True Delay")
plt.ylabel("Predicted Delay")
plt.title("Predicted vs True Total Delay")
plt.grid(True)
plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')  # y=x line
plt.tight_layout()
plt.savefig("pred_vs_true.png")
plt.show()
