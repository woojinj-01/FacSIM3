from copy import deepcopy
from random import shuffle


def random(g):
    g_rand = randomize(g)[0]
    g_rand.name = f'Random {g.name}'

    return g_rand


def randomize(g, times=1):

    randoms = []

    for i in range(times):

        g_rand = _randomize(g)
        g_rand.name = f"Random {g.name} {i}"

        randoms.append(g_rand)

    return randoms

    
def _randomize(g):

    g_rand = deepcopy(g)

    nodes_u = []
    nodes_v = []

    for name, deg in g.out_degree:
        
        for _ in range(deg):
            nodes_u.append(name)

    for name, deg in g.in_degree:

        for _ in range(deg):
            nodes_v.append(name)

    g_rand.clear_edges()

    shuffle(nodes_u)
    shuffle(nodes_v)

    for inst_u, inst_v in zip(nodes_u, nodes_v):
        
        g_rand.add_edge(inst_u, inst_v)

    return g_rand