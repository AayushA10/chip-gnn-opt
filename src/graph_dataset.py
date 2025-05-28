from torch_geometric.data import Dataset, Data
from netlist_graph_gen import generate_netlist_graph
from torch_geometric.utils import from_networkx
import torch

class NetlistGraphDataset(Dataset):
    def __init__(self, num_graphs=100):
        super().__init__()
        self.num_graphs = num_graphs
        self.graphs = []

        for _ in range(num_graphs):
            G = generate_netlist_graph()
            data = from_networkx(G)
            num_nodes = G.number_of_nodes()
            data.x = torch.randn((num_nodes, 10))
            total_delay = sum(attr['weight'] for _, _, attr in G.edges(data=True))
            data.y = torch.tensor([total_delay], dtype=torch.float)
            self.graphs.append(data)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]
