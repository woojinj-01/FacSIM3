import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from networks.construct_net import construct_network
from config.config import identifiers, fig_path
import parse.queries_integration as qsi
import plot.config as pcfg

max_domestic_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}


def rank_v_dist():

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    ylabel = r"$\pi_{KR}$"
    xlabel = 'Distance to Seoul (km)'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        if i == 0:
            handles, labels = _rank_v_dist(ax, g, return_legend=True)

        else:
            _rank_v_dist(ax, g)

    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               fontsize=pcfg.legend_size, bbox_to_anchor=(0.5, 1.05), frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path("rank_v_dist"),
                bbox_inches='tight')


def _rank_v_dist(ax, g, return_legend=False):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)
    max_rank = max_domestic_ranks[iden.split('_')[1]]

    ranks = []
    distances = []

    ranks_ist = []
    dist_ist = []

    ranks_postech = []
    dist_postech = []

    for id, data in g.nodes(data=True):

        rank = data[f"{abbrev}_rank_domestic"]
        dist = data["distance_to_seoul"]

        if id in [2, 12, 229, 282, 48]:

            ranks_ist.append(rank)
            dist_ist.append(dist)

        elif id in [15]:
            ranks_postech.append(rank)
            dist_postech.append(dist)

        else:
            ranks.append(rank)
            distances.append(dist)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlim(0, 500)
    ax.set_ylim(0, max_rank + 1)

    ax.set_xticks(range(0, 501, 100))

    # ax.invert_xaxis()
    ax.invert_yaxis()

    ax.scatter(distances, ranks, marker='o',
                alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
                edgecolor='black')
    
    ax.scatter(dist_ist, ranks_ist, marker='s',
                alpha=pcfg.alpha, s=150, c='yellow', label='-IST',
                edgecolor='black')
    
    ax.scatter(dist_postech, ranks_postech, marker='v',
                alpha=pcfg.alpha, s=150, c='yellow', label='POSTECH',
                edgecolor='black')
    
    # ax.legend()

    if return_legend:
        return ax.get_legend_handles_labels()
    
    else:
        return None, None
    

if __name__ == "__main__":
    rank_v_dist()