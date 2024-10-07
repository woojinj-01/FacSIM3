import networkx as nx
import sqlite3 as sql
import pandas as pd

import networks.SpringRank as sp
import parse.queries_integration as qsi
from config.config import net_db_path, csv_data_path
from parse.text_processing import normalize_inst_name

alpha_for_sprank = 2


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


def calc_rank(g, alpha=alpha_for_sprank):
    
    mat = nx.to_numpy_array(g, nodelist=sorted(g.nodes()))

    scores = sp.SpringRank(mat, alpha=alpha)
    # inv_temp = sp.get_inverse_temperature(mat, scores)

    ranks = score_to_rank(scores)

    return zip(sorted(g.nodes()), ranks)


def calc_rank_to_dict(g, alpha=alpha_for_sprank):
    
    mat = nx.to_numpy_array(g, nodelist=sorted(g.nodes()))

    scores = sp.SpringRank(mat, alpha=alpha)
    # inv_temp = sp.get_inverse_temperature(mat, scores)

    ranks = score_to_rank(scores)

    return {id: rank for id, rank in zip(sorted(g.nodes()), ranks)}


def grant_ranks_to_db(g, net_type='global'):

    conn = sql.connect(net_db_path("KR_Integrated"))
    cursor = conn.cursor()

    iden = g.name

    ranks = calc_rank(g)

    if net_type == 'global':
        grant_rank = qsi.GRANT_GRANK_BY_NODEID(iden)
    elif net_type == 'domestic':
        grant_rank = qsi.GRANT_DRANK_BY_NODEID(iden)

    for id, rank in ranks:
        cursor.execute(grant_rank, (rank, id))

    conn.commit()

    cursor.close()
    conn.close()


def grant_all_ranks():

    from networks.construct_net import construct_network

    nets = construct_network(data=False)

    for g in nets.values():
        grant_ranks_to_db(g)

    nets = construct_network(data=False, net_type='domestic')

    for g in nets.values():
        grant_ranks_to_db(g, net_type='domestic')


def grant_wapman_ranks(g):

    conn = sql.connect(net_db_path('KR_Integrated'))
    cursor = conn.cursor()

    iden = g.name
    iden_us = '_'.join(["US", iden.split('_')[1]])

    us_path = csv_data_path(iden_us)

    df = pd.read_csv(us_path, header=None)

    ranks = df.iloc[1:, 0]
    names = [normalize_inst_name(name) for name in df.iloc[1:, 2]]

    name_to_rank = {}

    for rank, name in zip(ranks, names):
        name_to_rank[name] = rank

    # print(name_to_rank)
    print(iden)

    for id in sorted(g.nodes()):

        full_name = g.nodes[id]["name"]
        name = full_name.split(', ')[0]
        nation = g.nodes[id]["nation"]

        if nation == "US":
            if name in name_to_rank.keys():
                cursor.execute(qsi.GRANT_WRANK_BY_NODEID(iden),
                               (name_to_rank[name], id))

    df = pd.read_csv(f"Wapman_fail_{iden}_filtered.csv")

    names = df.iloc[:, 0]
    ranks = df.iloc[:, 1]

    for name, rank in zip(names, ranks):
        cursor.execute(qsi.GRANT_WRANK_BY_NAME(iden), (rank, name))
    
    conn.commit()
    cursor.close()
    conn.close()


def grant_all_wapman_ranks():

    from networks.construct_net import construct_network

    nets = construct_network()

    for g in nets.values():
        grant_wapman_ranks(g, add_col='True')


def _immigrate_wapman_ranks(g, cursor_dst):
    
    iden_src = g.name

    conn_src = sql.connect(net_db_path(iden_src))
    cursor_src = conn_src.cursor()

    cursor_src.execute(qsi.SELECT_TABLE_NODES)
    rows = cursor_src.fetchall()

    name_to_rank = {}

    for row in rows:

        name = normalize_inst_name(row[1])
        rank = row[10]

        if rank is not None:
            name_to_rank[name] = rank

    cursor_dst.execute(qsi.SELECT_TABLE_NODES)
    rows = cursor_dst.fetchall()

    for row in rows:

        id = row[0]
        name = normalize_inst_name(row[1])

        if name in name_to_rank:
            cursor_dst.execute(qsi.GRANT_WRANK_BY_NODEID(iden_src), (name_to_rank[name], id))   

    conn_src.commit()
    cursor_src.close()
    conn_src.close()


def immigrate_wapman_ranks():
     
    from networks.construct_net import construct_network

    conn_dst = sql.connect(net_db_path('KR_Integrated'))
    cursor_dst = conn_dst.cursor()

    nets = construct_network()

    for g in nets.values():
        _immigrate_wapman_ranks(g, cursor_dst)

    conn_dst.commit()
    cursor_dst.close()
    conn_dst.close()


if __name__ == "__main__":
    # immigrate_wapman_ranks()
    grant_all_ranks()
