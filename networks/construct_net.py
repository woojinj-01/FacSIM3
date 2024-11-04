import networkx as nx
import sqlite3 as sql
import re

from config.config import net_db_path, identifiers
import parse.queries_integration as qsi


def add_space_before_uppercase(text):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)


def db_to_nx(iden, net_type='global', data=True):

    phd_id_index = 18
    src_nation_index = 19
    dst_nation_index = 0

    db_path = net_db_path("KR_Integrated")
    conn = sql.connect(db_path)
    cursor = conn.cursor()

    g = nx.MultiDiGraph()

    field_name = add_space_before_uppercase(iden.split('_')[1])

    cursor.execute(qsi.SELECT_EDGES_FOR_GNET, (field_name, ))
    edges = cursor.fetchall()

    columns_edges = [description[0] for description in cursor.description]

    ap_pos_index = columns_edges.index('which_job_is_ap')

    for edge in edges:

        ap_pos = edge[ap_pos_index]

        if ap_pos is not None:
            dst_id_index = columns_edges.index(f"job{ap_pos}_inst_id")
            dst_nation_index = dst_id_index + 1
        else:
            continue

        src_nation = edge[src_nation_index]
        dst_nation = edge[dst_nation_index]
        
        if net_type == 'domestic':
            if src_nation != 'KR' or dst_nation != 'KR':
                continue
            
        src_id = edge[phd_id_index]
        dst_id = edge[dst_id_index]

        if data:
            edge_data = {col: edge[i] for i, col in enumerate(columns_edges)}
            g.add_edge(src_id, dst_id, **edge_data)

        else:
            g.add_edge(src_id, dst_id)

    g.name = iden
    cursor.execute(qsi.SELECT_TABLE_NODES)
    nodes_data = cursor.fetchall()

    columns_nodes = [description[0] for description in cursor.description]

    num = 0
    num_hit = 0

    for node_data in nodes_data:
        node_id = node_data[0]  # Assuming the first column is the node ID
        node_attrs = {col: node_data[i] for i, col in enumerate(columns_nodes)}
        
        if node_id in g:
            g.nodes[node_id].update(node_attrs)
            num_hit += 1

        num += 1

    conn.close()

    return g


def construct_network(net_type='global', data=True):

    net_dict = {}

    for iden in identifiers:
        net = db_to_nx(iden, net_type=net_type, data=data)

        net_dict[iden] = net

    return net_dict
    

if __name__ == "__main__":
    net_dict = construct_network(data=True, net_type='domestic')

    print('=== Domestic networks ===')

    for iden, net in net_dict.items():

        print(iden)
        print(f"Nodes: {net.number_of_nodes()}")
        print(f"Edges: {net.number_of_edges()}")

        # print(net.edges[5, 5, 0])

    net_dict = construct_network(data=True)

    print('=== Global networks ===')

    for iden, net in net_dict.items():

        print(iden)
        print(f"Nodes: {net.number_of_nodes()}")
        print(f"Edges: {net.number_of_edges()}")

        # print(net.edges[5, 5, 0])


