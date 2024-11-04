import numpy as np

from networks.construct_net import construct_network
from config.config import identifiers


def calc_hires(net_type='domestic'):

    nets = construct_network(net_type=net_type)

    results = {}

    for iden in identifiers:

        g = nets[iden] 
        results[iden] = _calc_hires(g, net_type)

    return results


def _calc_hires(g, net_type):

    from parse.queries_integration import get_abbrev

    iden = g.name

    rank_column = f"{get_abbrev(iden)}_rank_{net_type}"

    stats = {"Up": 0, "Down": 0, "Self": 0, "Total": 0}

    for src_id, dst_id, data in g.edges(data=True):

        src_rank = g.nodes[src_id][rank_column]
        dst_rank = g.nodes[dst_id][rank_column]

        if src_rank is None or dst_rank is None:
            continue

        if src_rank == dst_rank:
            stats["Self"] += 1

        elif src_rank < dst_rank:
            stats["Down"] += 1
        
        else:
            stats["Up"] += 1

        stats["Total"] += 1

    return stats


def calc_hires_z(net_type='domestic'):

    nets = construct_network(net_type=net_type)

    results = {iden: {"Up": None, "Self": None, "Down": None} for iden in identifiers}

    for iden in identifiers:

        g = nets[iden]

        stats = _calc_hires(g, net_type=net_type) 
        mean_and_std = _calc_hires_z(g, net_type)

        for type in ["Up", "Self", "Down"]:

            stat_norm = stats[type] / stats["Total"]

            results[iden][type] = (stat_norm - mean_and_std[type][0]) / mean_and_std[type][1]

    return results


def _calc_hires_z(g, net_type):

    from parse.queries_integration import get_abbrev
    from networks.randomize import randomize

    trial = 1000

    iden = g.name
    randoms = randomize(g, times=trial)

    rank_column = f"{get_abbrev(iden)}_rank_{net_type}"

    stats_rand = []

    for g_rand in randoms:

        stat = {"Up": 0, "Down": 0, "Self": 0, "Total": 0}

        for src_id, dst_id, _ in g_rand.edges(data=True):

            src_rank = g.nodes[src_id][rank_column]
            dst_rank = g.nodes[dst_id][rank_column]

            if src_rank is None or dst_rank is None:
                continue

            if src_rank == dst_rank:
                stat["Self"] += 1

            elif src_rank < dst_rank:
                stat["Down"] += 1
            
            else:
                stat["Up"] += 1

            stat["Total"] += 1

        stats_rand.append(stat)

    up = []
    se = []
    do = []

    for stat in stats_rand:

        up.append(stat["Up"] / stat["Total"])
        se.append(stat["Self"] / stat["Total"])
        do.append(stat["Down"] / stat["Total"])

    mean_and_std = {}

    mean_and_std["Up"] = (np.mean(up), np.std(up))
    mean_and_std["Self"] = (np.mean(se), np.std(se))
    mean_and_std["Down"] = (np.mean(do), np.std(do))

    return mean_and_std


def calc_duration(net_type='domestic'):

    nets = construct_network(net_type=net_type)

    results = {}

    for iden in identifiers:

        g = nets[iden] 
        results[iden] = _calc_duration(g, net_type)

    return results


def _calc_duration(g, net_type):

    x = []
    y = []

    for src_id, dst_id, data in g.edges(data=True):

        i = data['which_job_is_ap']

        year_phd = data["phd_end_year"]
        year_hired = data[f"job{i}_start_year"]

        if any(not isinstance(y, int) or y == 0 for y in [year_phd, year_hired]):
            continue

        duration = year_hired - year_phd

        x.append(year_phd)
        y.append(duration)

    return x, y


def calc_duration_by_dec(net_type='domestic'):

    nets = construct_network(net_type=net_type)

    results = {}

    for iden in identifiers:

        g = nets[iden] 
        results[iden] = _calc_duration_by_dec(g, net_type)

    return results


def _calc_duration_by_dec(g, net_type):

    data = {'Total': [],
            '-1999': [],
            '2000s': [],
            '2010s': []}

    for _, _, d in g.edges(data=True):

        i = d['which_job_is_ap']

        year_phd = d["phd_end_year"]
        year_hired = d[f"job{i}_start_year"]

        if any(not isinstance(y, int) or y == 0 for y in [year_phd, year_hired]):
            continue

        duration = year_hired - year_phd

        if year_phd <= 1999:
            data['-1999'].append(duration)
        elif 2000 <= year_phd < 2010:
            data['2000s'].append(duration)
        else:
            data['2010s'].append(duration)

        data['Total'].append(duration)

    for k, v in data.items():
        data[k] = np.mean(v)

    print(data)
    
    return data


if __name__ == '__main__':
    calc_duration_by_dec()



    




