import pandas as pd

from networks.construct_net import construct_network
from config.config import csv_data_path, identifiers
from parse.queries_integration import get_abbrev


def extract_node_edge_list_overall(net_type='global', data=True):

    nets = construct_network(net_type)

    for g in nets.values():
        extract_node_edge_list(g, net_type, data=data)


def extract_node_edge_list(g, net_type, data=True):

    extract_node_list(g, net_type, data=data)
    extract_edge_list(g, net_type, data=data)


def extract_rank_list_overall(net_type='global'):

    nets = construct_network(net_type)

    for g in nets.values():
        extract_rank_list(g, net_type)


def extract_node_list(g, net_type, data=True, mode='extract'):

    assert (mode in ['extract', 'return'])

    node_list = g.nodes(data=data)
    col_list = ['Node', 'Attributes'] if data is True else ['Node']

    node_df = pd.DataFrame(node_list, columns=col_list)

    if data:
        if len(node_df) > 0 and 'Attributes' in node_df.columns:
            node_attributes_df = pd.json_normalize(node_df['Attributes'])
            node_df = node_df.drop(columns=['Attributes']).join(node_attributes_df)

    file_name = f'Nodes_{g.name}_{net_type}_data({data})'

    if mode == 'extract':
        return node_df.to_csv(csv_data_path(file_name), sep='\t')
    
    else:
        return node_df


def extract_edge_list(g, net_type, data=True, mode='extract'):

    assert (mode in ['extract', 'return'])

    edge_list = list(g.edges(data=data))
    col_list = ['Src', 'Target', 'Attributes'] if data is True else ['Src', 'Target']

    edge_df = pd.DataFrame(edge_list, columns=col_list)

    # Normalize the attributes dictionary to separate columns
    if data:
        if len(edge_df) > 0 and 'Attributes' in edge_df.columns:
            edge_attributes_df = pd.json_normalize(edge_df['Attributes'])
            edge_df = edge_df.drop(columns=['Attributes']).join(edge_attributes_df)

    file_name = f'Edges_{g.name}_{net_type}_data({data})'

    if mode == 'extract':
        return edge_df.to_csv(csv_data_path(file_name), sep='\t')

    else:
        return edge_df


def extract_rank_list(g, net_type, mode='extract'):

    iden = g.name
    abbrev = get_abbrev(iden)

    assert (net_type in ['global', 'domestic'])
    assert (iden in identifiers)
    assert (mode in ['extract', 'return'])

    rank_key = f'{abbrev}_rank_{net_type}'

    selected = ['id', 'name', rank_key]

    node_df = extract_node_list(g, net_type, data=True, mode='return')

    rank_df = node_df[selected]

    file_name = f'Ranks_{g.name}_{net_type}'

    if mode == 'extract':
        return rank_df.to_csv(csv_data_path(file_name), sep='\t')

    else:
        return rank_df


if __name__ == '__main__':

    # extract_node_edge_list_overall(data=False)
    extract_node_edge_list_overall(data=True)
    extract_rank_list_overall()

    # extract_node_edge_list_overall(data=False, net_type='domestic')
    extract_node_edge_list_overall(data=True, net_type='domestic')
    extract_rank_list_overall(net_type='domestic')





