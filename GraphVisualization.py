import networkx as nx
from matplotlib import pyplot as plt

def Show_Graph(edges,nodes):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.draw_networkx(G)
    plt.show()