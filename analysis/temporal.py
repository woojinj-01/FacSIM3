import networkx as nx

from plot.trained_rank import _classify_career_year


def split_g_into_decades(g):

    decs = [f"{(i + 7) % 10}0s" for i in range(6)]

    g_by_dec = {dec: nx.MultiDiGraph() for dec in decs}

    for src_id, dst_id, data in g.edges(data=True):

        phd_year = data['phd_end_year']
        dec = _classify_career_year(phd_year)

        if dec is None:
            continue

        g_by_dec[dec].add_edge(src_id, dst_id)


    return g_by_dec








