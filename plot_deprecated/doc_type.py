import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from networks.construct_net import construct_network
from config.config import net_db_path, fig_path, identifiers
import parse.queries_integration as qsi
import plot.config as pcfg

max_wapman_ranks = {"Biology": 201,
                    "ComputerScience": 216,
                    "Physics": 214}

max_domestic_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}


def records_total(normalize=True):

    from analysis.doc_type import calc_records_total

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 1, figsize=(pcfg.fig_xsize, pcfg.fig_ysize),
                            dpi=pcfg.dpi, sharey=True)

    ylabel = 'Fraction of records' if normalize else 'Number of records'

    plt.ylabel(ylabel, fontsize=pcfg.xlabel_size)

    plt.xticks(fontsize=pcfg.small_tick_size - 5)
    plt.yticks(fontsize=pcfg.tick_size)

    stats = calc_records_total()

    for i, iden in enumerate(identifiers):

        stat = stats[iden]
        field = iden.split('_')[1]

        if field == 'ComputerScience':
            field = 'Computer\nScience'

        female = stat['Female']
        male = stat['Male']
        unpercieved = stat['Used'] - male - female
        unused = stat['Total'] - stat['Used']

        if normalize:
            female = female / stat['Total']
            male = male / stat['Total']
            unpercieved = unpercieved / stat['Total']
            unused = unused / stat['Total']

        if i == 0:
            plt.bar(field, female, color=pcfg.colors['KR_ComputerScience'][1],
                    alpha=pcfg.alpha, edgecolor='black', linewidth=3,
                    label='Perceived female', width=0.7)
            plt.bar(field, unpercieved, color=pcfg.colors['KR_Physics'][0],
                    alpha=pcfg.alpha, edgecolor='black', linewidth=3, bottom=female+male,
                    label='Unknown gender', width=0.7)
            plt.bar(field, male, color=pcfg.colors['KR_Biology'][0],
                    alpha=pcfg.alpha, edgecolor='black', linewidth=3, bottom=female,
                    label='Perceived male', width=0.7)
            plt.bar(field, unused, color='lightgrey',
                    alpha=pcfg.alpha, edgecolor='black', linewidth=3, bottom=female+male+unpercieved,
                    label='Unused records', width=0.7)
            
        else:
            plt.bar(field, female, color=pcfg.colors['KR_ComputerScience'][1],
                    alpha=pcfg.alpha, edgecolor='black', linewidth=3, width=0.7)
            plt.bar(field, male, color=pcfg.colors['KR_Biology'][0],
                    alpha=pcfg.alpha, edgecolor='black', linewidth=3, bottom=female, width=0.7)
            plt.bar(field, unpercieved, color=pcfg.colors['KR_Physics'][0],
                    alpha=pcfg.alpha, edgecolor='black', linewidth=3, bottom=female+male, width=0.7)
            plt.bar(field, unused, color='lightgrey',
                    alpha=pcfg.alpha, edgecolor='black', linewidth=3, bottom=female+male+unpercieved,
                    width=0.7)
            
    if normalize:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.25))

    else:
        plt.ylim(0, 1500)
        plt.yticks(range(0, 1501, 500))
        
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False, fontsize=25)
    # plt.legend()
    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"records_total_normalize({normalize})"), bbox_inches='tight')
    plt.clf()


def doc_type_total():

    from analysis.doc_type import calc_doc_type_total

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 1, figsize=(pcfg.fig_xsize, pcfg.fig_ysize),
                            dpi=pcfg.dpi, sharey=True)

    plt.ylabel(r"$Fraction$", fontsize=pcfg.xlabel_size)

    plt.xticks(fontsize=pcfg.small_tick_size - 5)
    plt.yticks(fontsize=pcfg.tick_size)

    plt.ylim(0, 1)

    stats = calc_doc_type_total()

    for iden in identifiers:

        stat = stats[iden]
        field = iden.split('_')[1]

        if field == 'ComputerScience':
            field = 'Computer\nScience'

        kr = stat["KR"] / stat["Total"]
        us = stat["US"] / stat["Total"]
        others = stat["Others"] / stat["Total"]

        plt.bar(field, us, color=pcfg.colors[iden][0], alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        plt.bar(field, others, color=pcfg.colors[iden][4], bottom=us, alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        plt.bar(field, kr, color=pcfg.colors[iden][2], bottom=us+others, alpha=pcfg.alpha, edgecolor='black', linewidth=2)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path("doc_type_total"), bbox_inches='tight')
    plt.clf()


def doc_type_us_kr():

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize),
                            dpi=pcfg.dpi, sharey=True)

    axs[1].set_xlabel("Fraction of faculty", fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel("Rank decile", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _doc_type_us_kr(ax, g)

    handles = []
    labels = []
    
    for i in [0, 4, 2]:
        for iden in identifiers:

            handle = Patch(facecolor=pcfg.colors[iden][i], alpha=pcfg.alpha, edgecolor='black', linewidth=3)
            handles.append(handle)

    for nation in ['United States', 'Others', 'South Korea']:
        for field in ['Biology', 'Computer Science', 'Physics']:

            label = f'{field} ({nation})'
            labels.append(label)

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2),
               ncol=3, frameon=False, fontsize=pcfg.legend_size)
    
    plt.yticks(range(1, 11), fontsize=pcfg.tick_size)
    plt.tight_layout(pad=1)
    plt.savefig(fig_path("doc_type_group"),
                bbox_inches='tight')
    

def _doc_type_us_kr(ax, g):

    from analysis.doc_type import _calc_doc_type

    iden = g.name
    stats = _calc_doc_type(g)

    x_co = [i + 1 for i in range(10)]
    y_co_us = [stat["US"] / stat["Total"] for stat in stats]
    y_co_kr = [stat["KR"] / stat["Total"] for stat in stats]
    y_co_others = [stat["Others"] / stat["Total"] for stat in stats]

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.25))

    # ax.invert_yaxis()

    bars = ax.barh(x_co, y_co_us, color=pcfg.colors[iden][0],
            alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_others, color=pcfg.colors[iden][4], left=y_co_us,
            alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_kr, color=pcfg.colors[iden][2],
            left=[us + others for us, others in zip(y_co_us, y_co_others)],
            alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    
    for bar in bars[-3:]:
        width = bar.get_width()
        ax.text(width / 2, bar.get_y() + bar.get_height() / 2 - 0.05,
                f'{width: .2f}', ha='center', va='center')
    

def doc_type_dist():

    nets = construct_network(net_type='global')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    xlabel = 'Fraction of US-trained faculty'
    ylabel = 'Distance to Seoul (km)'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _doc_type_dist(ax, g)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path("doc_type_v_dist"),
                bbox_inches='tight')
    

def _doc_type_dist(ax, g):

    stats = {}
    iden = g.name

    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if dst_nation != 'KR':
            continue

        if dst_id not in stats:
            stats[dst_id] = {"US": 0, "KR": 0, "Others": 0, "Total": 0}
        
        if src_nation in stats[dst_id]:
            stats[dst_id][src_nation] += 1
        else:
            stats[dst_id]["Others"] += 1

        stats[dst_id]["Total"] += 1

    frac = []
    dists = []

    for id, stat in stats.items():

        dist = g.nodes[id]['distance_to_seoul']

        if dist is None or stat["Total"] == 0:
            continue

        frac.append(stat["US"] / stat["Total"])
        dists.append(dist)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 500)

    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.set_yticks(range(0, 501, 100))

    ax.invert_yaxis()

    ax.scatter(frac, dists, marker='o',
                alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
                edgecolor='black')
    

if __name__ == "__main__":

    # records_total()
    # records_total(normalize=False)

    doc_type_us_kr()

