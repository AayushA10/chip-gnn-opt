import torch
from torch_geometric.utils import from_networkx
from netlist_graph_gen import generate_netlist_graph

# Step 1: Generate the netlist graph
G = generate_netlist_graph()

# Step 2: Convert to PyTorch Geometric format
data = from_networkx(G)

# Step 3: Add random node features (e.g., 10 features per component)
num_nodes = G.number_of_nodes()
data.x = torch.randn((num_nodes, 10))

# Step 4: Set the label (target) as total delay (sum of all edge weights)
total_delay = sum([attr['weight'] for _, _, attr in G.edges(data=True)])
data.y = torch.tensor([total_delay], dtype=torch.float)

# ✅ Check it worked
print("Edge Index:", data.edge_index)
print("Node Features Shape:", data.x.shape)
print("Label (Total Delay):", data.y.item())
