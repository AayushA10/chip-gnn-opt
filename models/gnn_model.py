import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool

class GNNDelayRegressor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 32, heads=2, concat=True)  # 32 * 2 = 64 output
        self.bn1 = BatchNorm1d(64)
        self.dropout1 = Dropout(0.3)

        self.conv2 = GATConv(64, 32, heads=1)
        self.bn2 = BatchNorm1d(32)
        self.dropout2 = Dropout(0.3)

        self.linear = Linear(32, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = global_mean_pool(x, batch)  # Graph-level pooling
        return self.linear(x)
