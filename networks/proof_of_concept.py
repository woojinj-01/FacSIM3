import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import rankdata
from copy import deepcopy

import networks.SpringRank as sp
from networks.construct_net import construct_network
from parse.queries_integration import get_abbrev

max_wapman_ranks = {"Biology": 201,
                    "ComputerScience": 216,
                    "Physics": 214}

alpha_for_sprank = 2


def calc_rank_to_dict(g, alpha=alpha_for_sprank, reverse=False):
    
    mat = nx.to_numpy_array(g, nodelist=sorted(g.nodes()))

    scores = sp.SpringRank(mat, alpha=alpha)
    # inv_temp = sp.get_inverse_temperature(mat, scores)

    ranks = score_to_rank(scores)

    if not reverse:
        return {rank: id for id, rank in zip(sorted(g.nodes()), ranks)}
    
    else:
        return {id: rank for id, rank in zip(sorted(g.nodes()), ranks)}


def score_to_rank(values):

    sorted_values = sorted([(val, idx) for idx, val in enumerate(values)],
                           reverse=True)
    
    ranks = [0] * len(values)
    current_rank = 1
    num_of_ties = 0
    
    for i in range(len(sorted_values)):
        if i > 0 and sorted_values[i][0] == sorted_values[i-1][0]:
            ranks[sorted_values[i][1]] = ranks[sorted_values[i-1][1]]
            num_of_ties += 1
        else:

            current_rank += num_of_ties
            num_of_ties = 0

            ranks[sorted_values[i][1]] = current_rank
            current_rank += 1
    
    return ranks


def generate_graph(num_nodes, num_edges):

    g = nx.MultiDiGraph()
    g.add_nodes_from(range(1000, 1000 + num_nodes))

    nodes = list(g.nodes(data=False))

    for _ in range(num_edges):

        src = random.choice(nodes)
        dst = random.choice(nodes)

        g.add_edge(src, dst)

    return g


def generate_kr_graph(iden):

    nets = construct_network(net_type='domestic')

    return nets[iden]


def generate_us_graph(iden):

    num_nodes = max_wapman_ranks[iden.split('_')[1]]
    num_edges = 3000

    g = generate_graph(num_nodes, num_edges)

    return g


def generate_combined_graph(iden, trial=5):

    g_kr = generate_kr_graph(iden)
    g_ref = construct_network(net_type='global')[iden]

    results = []

    for _ in range(trial):
        results.append(tuple(_generate_combined_graph(iden, g_kr, g_ref)))

    return results


def _generate_combined_graph(iden, g_kr, g_ref):

    abbrev = get_abbrev(iden)

    g_us = generate_us_graph(iden)

    ranks_us = calc_rank_to_dict(g_us)

    g_comb = nx.MultiDiGraph()

    g_comb.add_nodes_from(g_kr.nodes(data=True))
    g_comb.add_edges_from(g_kr.edges(data=True))
    g_comb.add_nodes_from(g_us.nodes(data=True))
    g_comb.add_edges_from(g_us.edges(data=True))

    for src_id, dst_id, data in g_ref.edges(data=True):

        src_nation = g_ref.nodes[src_id]['nation']
        dst_nation = g_ref.nodes[dst_id]['nation']

        nations = (src_nation, dst_nation)

        match nations:
            
            case ('KR', 'US'):

                dst_rank = g_ref.nodes[dst_id][f"{abbrev}_rank_wapman"]

                if dst_rank is None:
                    continue

                dst_id_converted = ranks_us[dst_rank]

                g_comb.add_edge(src_id, dst_id_converted)

            case ('US', 'KR'):

                src_rank = g_ref.nodes[src_id][f"{abbrev}_rank_wapman"]

                if src_rank is None:
                    continue

                src_id_converted = ranks_us[src_rank]

                g_comb.add_edge(src_id_converted, dst_id)

            case _:
                continue

    g_comb_wo_us = deepcopy(g_comb)

    for u, v, data in g_us.edges(data=True):
        if g_comb_wo_us.has_edge(u, v):
            g_comb_wo_us.remove_edge(u, v)

    # Remove nodes with no connections
    nodes_to_remove = [node for node, degree in g_comb_wo_us.degree() if degree == 0]
    g_comb_wo_us.remove_nodes_from(nodes_to_remove)

    return g_comb, g_comb_wo_us, g_us


from config.config import identifiers
import numpy as np

trial = 500

for iden in identifiers:

    xco = None
    ycos = []

    for _ in range(trial):
        xco, yco = generate_combined_graph(iden)

        ycos.append(yco)

    mean = np.mean(ycos, axis=0)
    std = np.std(ycos, axis=0)

    plt.xlabel('Rank (Global)')
    plt.ylabel('Rank (Global, full records)')
    plt.errorbar(xco, mean, yerr=std, fmt='o', linestyle='none', capsize=5)
    plt.show()
    plt.clf()
