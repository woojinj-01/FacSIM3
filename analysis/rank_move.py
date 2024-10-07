max_wapman_ranks = {"Biology": 201,
                    "ComputerScience": 216,
                    "Physics": 214}

max_domestic_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}

max_global_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}


def calc_rank_move(g, net_type='domestic', group=None):

    from parse.queries_integration import get_abbrev
    from analysis.doc_type import divide_by_groups

    iden = g.name
    field = iden.split('_')[1]

    moves = []

    if net_type == 'global':
        max_rank = max_global_ranks[field]
    else:
        max_rank = max_domestic_ranks[field]

    if group is not None:
        index = group[0]
        num_group = group[1]

        range_by_group = divide_by_groups(max_rank, num_group)
        correct_range = range_by_group[index - 1]

        lower_bound = correct_range[0]
        upper_bound = correct_range[1]

    rank_column = f"{get_abbrev(iden)}_rank_{net_type}"

    for src_id, dst_id, data in g.edges(data=True):

        src_rank = g.nodes[src_id][rank_column]
        dst_rank = g.nodes[dst_id][rank_column]

        if src_rank is None or dst_rank is None:
            continue
        elif group is not None and not lower_bound <=src_rank < upper_bound:
            continue

        move = (src_rank - dst_rank) / max_rank

        moves.append(move)

    return moves