import matplotlib.pyplot as plt
import sqlite3 as sql
import scipy.stats
import googlemaps
import numpy as np

from networks.construct_net import construct_network
from networks.rank import calc_rank_to_dict
from config.config import net_db_path, fig_path, identifiers
import parse.queries_integration as qsi
import plot.config as pcfg
from plot.trained_rank import _classify_career_year
from analysis.temporal import split_g_into_decades


def sprank_by_dec(net_type='global'):

    nets = construct_network(net_type=net_type)

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    fig.supxlabel('Decades', fontsize=pcfg.xlabel_size)
    fig.supylabel(r"$\pi_{normalized}$", fontsize=pcfg.ylabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        if i == 0:
            handles, labels = _sprank_by_dec(ax, g, return_legend=True)

        else:
            _sprank_by_dec(ax, g)

    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               fontsize=pcfg.legend_size, bbox_to_anchor=(0.5, 1.06), frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"sprank_by_dec_{net_type}"),
                bbox_inches='tight')


def _sprank_by_dec(ax, arg_g, return_legend=False):

    decs = [f"{(i + 7) % 10}0s" for i in range(6)]
    decs_to_plot = ["90s", "00s", "10s"]

    g_by_dec = split_g_into_decades(arg_g)
    ranks_by_id = {}

    for dec, g in g_by_dec.items():

        max_rank = len(g.nodes())

        ranks = calc_rank_to_dict(g)

        ranks_norm = {id: r / max_rank for id, r in ranks.items()}

        for id, r_norm in ranks_norm.items():

            if id not in ranks_by_id:
                ranks_by_id[id] = {d: None for d in decs} 

            ranks_by_id[id][dec] = r_norm

    for id, ranks in ranks_by_id.items():

        ranks_to_plot = []

        for d in decs_to_plot:

            if ranks[d] is None:
                continue
            else:
                ranks_to_plot.append(ranks[d])
        
        if len(ranks_to_plot) != 3:
            continue

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        match id:
            case 3:     #Korea Univ.: Hexagon, Red
                marker = 'h'    
                color = colors[3]
                label = 'Korea Univ.'

            case 5:     #SNU: Pentagon, Purple
                marker = 'p'    
                color = colors[4]
                label = 'SNU'

            case 12:    #KAIST: Diamond, Blue
                marker = 'D'    
                color = colors[0]
                label = 'KAIST'

            case 15:    #POSTECH: Square, Orange
                marker = 's'    
                color = colors[1]
                label = "POSTECH"
                
            case 26:    #Yonsei Univ.: Triangle down, Green
                marker = 'v'    
                color = colors[2]
                label = 'Yonsei Univ.'

            case _:
                marker = ''

        if id in [3, 5, 12, 15, 26]:
            ax.plot(decs_to_plot, ranks_to_plot, linewidth=3,
                    color=color, marker=marker, markersize=15, label=label)

        else:
            ax.plot(decs_to_plot, ranks_to_plot, linewidth=6, color='grey',
                    alpha=pcfg.alpha - 0.1, marker=marker, markersize=10)
    
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    # ax.legend()

    if return_legend:
        return ax.get_legend_handles_labels()
    

def sprank_by_dec_hist(net_type='global'):

    nets = construct_network(net_type=net_type)

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    fig.supxlabel('Rank variation', fontsize=pcfg.xlabel_size)
    fig.supylabel(r"#", fontsize=pcfg.ylabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _sprank_by_dec_hist(ax, g)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"sprank_by_dec_hist_{net_type}"),
                bbox_inches='tight')


def _sprank_by_dec_hist(ax, arg_g):

    decs = [f"{(i + 7) % 10}0s" for i in range(6)]
    decs_to_plot = ["90s", "00s", "10s"]

    g_by_dec = split_g_into_decades(arg_g)
    ranks_by_id = {}

    for dec, g in g_by_dec.items():

        max_rank = len(g.nodes())

        ranks = calc_rank_to_dict(g)

        ranks_norm = {id: r / max_rank for id, r in ranks.items()}

        for id, r_norm in ranks_norm.items():

            if id not in ranks_by_id:
                ranks_by_id[id] = {d: None for d in decs} 

            ranks_by_id[id][dec] = r_norm

    variation_90_00 = []
    variation_00_10 = []
    variation_90_10 = []

    variation_90_10_dict = {}

    for id, ranks in ranks_by_id.items():

        ranks_to_plot = []

        for d in decs_to_plot:

            if ranks[d] is None:
                continue
            else:
                ranks_to_plot.append(ranks[d])
        
        if len(ranks_to_plot) != 3:
            continue

        variation_90_00.append(ranks_to_plot[0] - ranks_to_plot[1])
        variation_00_10.append(ranks_to_plot[1] - ranks_to_plot[2])
        variation_90_10.append(ranks_to_plot[0] - ranks_to_plot[2])

        variation_90_10_dict[arg_g.nodes[id]['name']] = ranks_to_plot[0] - ranks_to_plot[2]

    bins = np.arange(-1, 1.01, 0.05)

    ax.hist(variation_90_00, bins=bins, color='blue',
            alpha=pcfg.alpha, edgecolor='black')
    ax.hist(variation_00_10, bins=bins, color='green',
            alpha=pcfg.alpha, edgecolor='black')
    ax.hist(variation_90_10, bins=bins, color='orange',
            alpha=pcfg.alpha, edgecolor='black')
    
    print(f"=== {arg_g.name} ===")

    sorted_keys = sorted(variation_90_10_dict, key=variation_90_10_dict.get, reverse=True)

    # Print the sorted keys
    for key in sorted_keys:
        print(key, variation_90_10_dict[key])

    print('\n')

    ax.set_xlim(-1, 1)
    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    # ax.legend()


if __name__ == "__main__":
    # sprank_by_dec(net_type='global')
    # sprank_by_dec(net_type='domestic')
    sprank_by_dec_hist(net_type='global')
    sprank_by_dec_hist(net_type='domestic')

