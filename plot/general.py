import networkx as nx
import matplotlib.pyplot as plt

from networks.construct_net import construct_network
import plot.config as pcfg


def draw_spring_layout(g):

    # Compute the spring layout, considering edge weights
    pos = nx.spring_layout(g, weight='weight')

    color_map = {'KR': pcfg.colors_bio[1], 'US': pcfg.colors_cs[1], 'CN': 'lightgrey'}
    node_colors = [color_map.get(g.nodes[node]['nation'], 'grey') for node in g.nodes]

    # Draw nodes
    nx.draw_networkx_nodes(g, pos, node_size=40, node_color=node_colors)

    # Draw edges with widths corresponding to weights
    edges = g.edges(data=True)
    # edge_weights = [edge[2]['weight'] for edge in edges]
    nx.draw_networkx_edges(g, pos, edgelist=edges, width=0.8)

    # Draw labels
    nx.draw_networkx_labels(g, pos, font_size=7, font_family='sans-serif')

    # Show plot
    plt.title(f"Spring Layout of {g.name}")
    plt.show()  


nets = construct_network(net_type='domestic')

for g in nets.values():
    draw_spring_layout(g)
