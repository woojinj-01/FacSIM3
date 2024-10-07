import sqlite3 as sql

from config.config import net_db_path
import parse.queries_integration as qsi

max_wapman_ranks = {"Biology": 201,
                    "ComputerScience": 216,
                    "Physics": 214}

max_domestic_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}


def calc_mobility(g):

    types = ["KR2KR", "US2KR", "KR2US"]

    results = {}

    for t in types:
        results[t] = tuple(_calc_mobility(g, type=t))

    return results


def _calc_mobility(g, type="KR2KR"):

    assert (type in ["KR2KR", "US2KR", "KR2US"])

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    max_domestic_rank = max_domestic_ranks[iden.split('_')[1]]

    match type:
        case "KR2KR":
            src_rank_to_use = f"{abbrev}_rank_domestic"
            dst_rank_to_use = f"{abbrev}_rank_domestic"

            src_max_rank = max_domestic_rank
            dst_max_rank = max_domestic_rank

        case "US2KR":
            src_rank_to_use = f"{abbrev}_rank_wapman"
            dst_rank_to_use = f"{abbrev}_rank_domestic"

            src_max_rank = max_wapman_ranks[iden.split('_')[1]]
            dst_max_rank = max_domestic_rank

        case "KR2US":
            src_rank_to_use = f"{abbrev}_rank_domestic"
            dst_rank_to_use = f"{abbrev}_rank_wapman"

            src_max_rank = max_domestic_rank
            dst_max_rank = max_wapman_ranks[iden.split('_')[1]]

    up_h = 0
    self_h = 0
    down_h = 0

    def correct_type(src, dst, type):

        match type:
            case "KR2KR":
                return src == 'KR' and dst == 'KR'
            case "US2KR":
                return src == 'US' and dst == 'KR'
            case "KR2US":
                return src == 'KR' and dst == 'US'

    # There should be some deliberations
    quantization = max(1 / src_max_rank, 1 / dst_max_rank) if type != 'KR2KR' else 0

    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if not correct_type(src_nation, dst_nation, type):
            continue

        src_rank = g.nodes[src_id][src_rank_to_use]
        dst_rank = g.nodes[dst_id][dst_rank_to_use]

        if src_rank is None or dst_rank is None:
            continue

        dev = src_rank / src_max_rank - dst_rank / dst_max_rank

        if -quantization <= dev <= quantization:
            self_h += 1
        elif dev > 0:
            up_h += 1
        else:
            down_h += 1 

    total_h = up_h + self_h + down_h

    return up_h, self_h, down_h, total_h


if __name__ == "__main__":

    from networks.construct_net import construct_network

    nets = construct_network()
    
    for g in nets.values():
        print(calc_mobility(g))





