import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from networks.construct_net import construct_network
from config.config import net_db_path, fig_path, identifiers
import plot.config as pcfg
import parse.queries_integration as qsi

max_domestic_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}


def lcurve_deg(net_type='domestic'):

    nets = construct_network(net_type=net_type)

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = 'Cumulative proportion of institutions'
    ylabel = 'Cumulative proportion of degrees'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _lcurve_deg(ax, g)

    handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=4),
               Line2D([0], [0], color='black', linestyle='--', linewidth=4)]

    labels = ["Out-degrees", "In-degrees"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
    
    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"lcurve_deg_{net_type}"),
                bbox_inches='tight')
    

def _lcurve_deg(ax, g):

    iden = g.name
    
    indegs = [deg[1] for deg in g.in_degree()]
    outdegs = [deg[1] for deg in g.out_degree()]

    indegs.sort(reverse=True)
    outdegs.sort(reverse=True)

    xco = []
    yco_in = []
    yco_out = []

    x = 0
    y_in = 0
    y_out = 0

    xco.append(x)
    yco_in.append(y_in)
    yco_out.append(y_out)

    for deg_in, deg_out in zip(indegs, outdegs):

        x += 1
        y_in += deg_in
        y_out += deg_out

        xco.append(x)
        yco_in.append(y_in)
        yco_out.append(y_out)

    xco = np.array(xco) / max(xco)
    yco_in = np.array(yco_in) / max(yco_in)
    yco_out = np.array(yco_out) / max(yco_out)

    area_in = np.trapz(yco_in, xco)
    gini_in = (area_in - 0.5) / 0.5

    area_out = np.trapz(yco_out, xco)
    gini_out = (area_out - 0.5) / 0.5

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    ax.plot(xco, yco_out, alpha=pcfg.alpha, c=pcfg.colors[iden][1], linewidth=6)
    ax.plot(xco, yco_in, alpha=pcfg.alpha, c=pcfg.colors[iden][1], linewidth=6, linestyle='--')
    ax.plot([0, 1], [0, 1], c='black', alpha=pcfg.alpha)

    ax.text(0.95, 0.05,
                f'Gini (out-deg): {gini_out:.3f}\nGini (in-deg): {gini_in:.3f}',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.small_tick_size, bbox=dict(facecolor='white', alpha=0.8))
    

def lcurve_deg_overall():

    nets_g = construct_network(net_type='global')
    nets_d = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = 'Cumulative proportion of institutions'
    ylabel = 'Cumulative proportion of degrees'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets_g[iden] 
        d = nets_d[iden] 

        _lcurve_deg_overall(ax, g, d)

    handles = [Patch(facecolor=pcfg.color_ut, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.color_kt, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Line2D([0], [0], color='black', linestyle='-', linewidth=4),
               Line2D([0], [0], color='black', linestyle='--', linewidth=4)]

    labels = ["Global", "Domestic", "Out-degrees", "In-degrees"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False)
    
    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"lcurve_deg_overall"),
                bbox_inches='tight')
    

def _lcurve_deg_overall(ax, g, d):

    iden = g.name

    def draw_lcurve(g_to_draw, c):
    
        indegs = [deg[1] for deg in g_to_draw.in_degree()]
        outdegs = [deg[1] for deg in g_to_draw.out_degree()]

        indegs.sort(reverse=True)
        outdegs.sort(reverse=True)

        xco = []
        yco_in = []
        yco_out = []

        x = 0
        y_in = 0
        y_out = 0

        xco.append(x)
        yco_in.append(y_in)
        yco_out.append(y_out)

        for deg_in, deg_out in zip(indegs, outdegs):

            x += 1
            y_in += deg_in
            y_out += deg_out

            xco.append(x)
            yco_in.append(y_in)
            yco_out.append(y_out)

        xco = np.array(xco) / max(xco)
        yco_in = np.array(yco_in) / max(yco_in)
        yco_out = np.array(yco_out) / max(yco_out)

        area_in = np.trapz(yco_in, xco)
        gini_in = (area_in - 0.5) / 0.5

        area_out = np.trapz(yco_out, xco)
        gini_out = (area_out - 0.5) / 0.5

        ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

        ax.plot(xco, yco_out, alpha=pcfg.alpha, c=c, linewidth=6)
        ax.plot(xco, yco_in, alpha=pcfg.alpha, c=c, linewidth=6, linestyle='--')

        return gini_in, gini_out
    
    gini_g_in, gini_g_out = draw_lcurve(g, pcfg.color_ut)
    gini_d_in, gini_d_out = draw_lcurve(d, pcfg.color_kt)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    ax.plot([0, 1], [0, 1], c='black', alpha=pcfg.alpha)

    label_g = f'<Global>\nGini (out-deg): {gini_g_out:.2f}\nGini (in-deg): {gini_g_in:.2f}'
    label_d = f'<Domestic>\nGini (out-deg): {gini_d_out:.2f}\nGini (in-deg): {gini_d_in:.2f}'

    label_to_put = '\n'.join([label_g, label_d])

    ax.text(0.95, 0.05,
                label_to_put,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.small_tick_size, bbox=dict(facecolor='white', alpha=0.8))


def deg_v_rank(net_type='domestic'):

    nets = construct_network(net_type=net_type)

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    axs[1].set_xlabel(r"$\pi$", fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(r"$Variable$", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _deg_v_rank(ax, g, net_type)

    # plt.show()
    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"deg_v_rank_{net_type}"),
                bbox_inches='tight')
    

def _deg_v_rank(ax, g, net_type):

    from parse.queries_integration import get_abbrev
    from networks.rank import calc_rank_to_dict
    
    iden = g.name

    ranks_dict_wo_alpha = calc_rank_to_dict(g, alpha=0)

    ranks = []
    ranks_wo_alpha = []
    out_degs = []
    in_degs = []
    deg_subs = []
    deg_divs = []

    rank_column = f"{get_abbrev(iden)}_rank_{net_type}"

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.invert_xaxis()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    for id, data in g.nodes(data=True):

        rank = data[rank_column]
        rank_wo_alpha = ranks_dict_wo_alpha[id]

        out_deg = g.out_degree(id)
        in_deg = g.in_degree(id)

        ranks.append(rank)
        ranks_wo_alpha.append(rank_wo_alpha)

        out_degs.append(out_deg)
        in_degs.append(in_deg)
        deg_subs.append(out_deg - in_deg)

        if in_deg != 0:
            deg_divs.append(10 * out_deg / in_deg)
        else:
            deg_divs.append(0)

    # ax.scatter(ranks, out_degs, label=r'$k_{out}$', alpha=pcfg.alpha)
    # ax.scatter(ranks, in_degs, label=r'$k_{in}$', alpha=pcfg.alpha)
    ax.scatter(ranks, deg_subs, label=r'$k_{out} - k_{in}$', alpha=pcfg.alpha)
    # ax.scatter(ranks, deg_divs, label=r'$10 * k_{out} / k_{in}$', alpha=pcfg.alpha)

    # ax.scatter(ranks_wo_alpha, out_degs, label=r'$k_{out} (\alpha = 0)$', alpha=pcfg.alpha, marker='v')
    # ax.scatter(ranks_wo_alpha, in_degs, label=r'$k_{in} (\alpha = 0)$', alpha=pcfg.alpha, marker='v')
    ax.scatter(ranks_wo_alpha, deg_subs, label=r'$k_{out} - k_{in} (\alpha = 0)$', alpha=pcfg.alpha, marker='v')
    # ax.scatter(ranks_wo_alpha, deg_divs, label=r'$10 * k_{out} / k_{in} (\alpha = 0)$', alpha=pcfg.alpha, marker='v')

    ax.legend()


def outdeg_v_dist():

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    xlabel = 'Distance to Seoul (km)'
    ylabel = 'Out degrees'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        if i == 0:
            handles, labels = _outdeg_v_dist(ax, g, return_legend=True)

        else:
            _outdeg_v_dist(ax, g)

    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               fontsize=pcfg.legend_size, bbox_to_anchor=(0.5, 1.05), frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path("deg_out_v_dist"),
                bbox_inches='tight')


def _outdeg_v_dist(ax, g, return_legend=False):

    iden = g.name

    ranks = []
    distances = []

    ranks_ist = []
    dist_ist = []

    ranks_postech = []
    dist_postech = []

    for id, data in g.nodes(data=True):

        rank = g.out_degree(id)
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

    ax.scatter(distances, ranks, marker='o',
                alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
                edgecolor='black')
    
    ax.scatter(dist_ist, ranks_ist, marker='s',
                alpha=pcfg.alpha, s=150, c='yellow', label='-IST',
                edgecolor='black')
    
    ax.scatter(dist_postech, ranks_postech, marker='v',
                alpha=pcfg.alpha, s=150, c='yellow', label='POSTECH',
                edgecolor='black')

    ax.set_xlim(0, 500)
    ax.set_ylim(0, 200)

    ax.set_xticks(range(0, 501, 100))

    if return_legend:
        return ax.get_legend_handles_labels()
    
    else:
        return None, None


if __name__ == "__main__":

    # lcurve_deg(net_type='global')
    # lcurve_deg(net_type='domestic')

    lcurve_deg_overall()



