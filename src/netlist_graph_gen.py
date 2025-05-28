import networkx as nx
import random

def generate_netlist_graph(num_nodes=20, num_edges=40):
    G = nx.gnm_random_graph(num_nodes, num_edges)
    
    # Add delay to each wire (edge)
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 5.0)  # simulate wire delay in ns

    return G
