from networks.construct_net import construct_network
from config.config import net_db_path, fig_path, identifiers
import parse.queries_integration as qsi

max_domestic_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}


def divide_by_groups(max_rank, num_groups):

    quo = int(max_rank / num_groups)
    red = max_rank % num_groups

    num_per_groups = [quo] * num_groups
    rank_tuple_per_groups = [0] * num_groups

    if red < int(num_groups / 2):
        num_per_groups[num_groups - 1] += red
    else:
        
        i = 0

        while red > 0:
            num_per_groups[i] += 1

            red -= 1    
            i += 1

    disp = 1

    for i, quota in enumerate(num_per_groups):
        
        start = disp
        end = disp + quota

        rank_tuple_per_groups[i] = (start, end)

        disp = end

    return rank_tuple_per_groups[::-1]


def calc_records_total():

    total_records = {"Biology": 933,
                    "ComputerScience": 1322,
                    "Physics": 714}
    
    results = {}
    
    nets = construct_network()

    for iden, g in nets.items():

        num_total = total_records[iden.split('_')[1]]
        num_used = len(g.edges)
        num_male = 0
        num_female = 0
        num_unknown_gender = 0

        for _, _, data in g.edges(data=True):

            gender = data['gender']

            if gender == 'Male':
                num_male += 1

            elif gender == 'Female':
                num_female += 1

            else:
                num_unknown_gender += 1

        stat = {'Male': num_male,
                'Female': num_female,
                'Used': num_used,
                'Total': num_total}

        results[iden] = stat

    return results


def calc_doc_type_total():
    
    nets = construct_network()
    results = {}

    for iden in identifiers:

        g = nets[iden] 
        results[iden] = _calc_doc_type_total(g)

    return results


def _calc_doc_type_total(g):

    iden = g.name

    stats = {"US": 0, "KR": 0, "Others": 0, "Total": 0}

    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if dst_nation != 'KR':
            continue

        dst_rank = g.nodes[dst_id][f"{qsi.get_abbrev(iden)}_rank_domestic"]

        if dst_rank is None:
            continue

        if src_nation in stats:
            stats[src_nation] += 1
        else:
            stats["Others"] += 1

        stats["Total"] += 1

    return stats


def calc_doc_type():

    nets = construct_network()

    results = {}

    for iden in identifiers:

        g = nets[iden] 
        results[iden] = _calc_doc_type(g)

    return results


def _calc_doc_type(g):

    num_groups = 10

    iden = g.name

    max_rank = max_domestic_ranks[iden.split('_')[1]]

    range_by_group = divide_by_groups(max_rank, num_groups)

    stats = [{"US": 0, "KR": 0, "Others": 0, "Total": 0} for i in range(num_groups)]

    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if dst_nation != 'KR':
            continue

        dst_rank = g.nodes[dst_id][f"{qsi.get_abbrev(iden)}_rank_domestic"]

        if dst_rank is None:
            continue

        group = 0

        for i in range(num_groups):
            start = range_by_group[i][0]
            end = range_by_group[i][1]

            if start <= dst_rank < end:
                group = i
                break
        
        if src_nation in stats[group]:
            stats[group][src_nation] += 1
        else:
            stats[group]["Others"] += 1

        stats[group]["Total"] += 1

    return stats


def _calc_doc_type_per_inst(g):

    iden = g.name

    stats = {}

    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if dst_nation != 'KR':
            continue

        dst_rank = g.nodes[dst_id][f"{qsi.get_abbrev(iden)}_rank_domestic"]

        if dst_rank is None:
            continue

        if dst_id not in stats:
            stats[dst_id] = {"US": 0, "KR": 0, "Others": 0, "Total": 0}
        
        if src_nation in stats[dst_id].keys():
            stats[dst_id][src_nation] += 1
        else:
            stats[dst_id]["Others"] += 1

        stats[dst_id]["Total"] += 1

    return stats


print(calc_records_total())

        



    

