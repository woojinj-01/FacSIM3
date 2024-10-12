import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import numpy as np
import scipy.stats
import math
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import statsmodels.api as sm
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

import plot.config as pcfg
import parse.queries_integration as qsi
from networks.construct_net import construct_network
from config.config import identifiers, fig_path
from analysis.doc_type import divide_by_groups
from plot.trained_rank import decades_to_interval, convert_per_to_ranks
from plot.rank_move import _interhistogram
from analysis.geography import area_seoul, area_metro, area_capital, area_others


iden_cs = 'KR_ComputerScience'
iden_bi = 'KR_Biology'
iden_ph = 'KR_Physics'

iden_default = iden_cs

max_wapman_ranks = {"Biology": 201,
                    "ComputerScience": 216,
                    "Physics": 214}

max_domestic_ranks = {"Biology": 86,
                    "ComputerScience": 115,
                    "Physics": 67}

patch_size = pcfg.small_tick_size - 10

doc = 'doc4'


def lcurve_deg_field(iden=iden_default):

    net_g = construct_network(net_type='global').get(iden)
    net_d = construct_network(net_type='domestic').get(iden)

    if any(net is None for net in [net_g, net_d]):
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    xlabel = 'Cumulative proportion of institutions'
    ylabel = 'Cumulative proportion of degrees'

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    _lcurve_deg_field(ax, net_g, net_d)

    handles = [Patch(facecolor=pcfg.color_ut, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.color_kt, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Line2D([0], [0], color='black', linestyle='-', linewidth=4),
               Line2D([0], [0], color='black', linestyle='--', linewidth=4)]

    labels = ["Global", "Domestic", "Out-degrees (faculty production)", "In-degrees (recruitment)"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False)
    
    # plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"./{doc}/lcurve_deg_{iden}"))
    

def _lcurve_deg_field(ax, g, d):

    def draw_lcurve(g_to_draw, c, analyze=False):
    
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

        if analyze:

            print('[Out-degrees]')

            for x, y in zip(xco, yco_out):

                if y >= 0.8:
                    print(f"{prev_x} of institutions produce {prev_y} of total faculty")
                    print(f"{x} of institutions produce {y} of total faculty")

                    grad = (y - prev_y) / (x - prev_x)

                    x_estimate = (0.8 - prev_y) / grad + prev_x

                    print(x_estimate)
                    break

                prev_x = x
                prev_y = y

            print('[In-degrees]')

            for x, y in zip(xco, yco_in):

                if y >= 0.8:
                    print(f"{prev_x} of institutions hire {prev_y} of total faculty")
                    print(f"{x} of institutions hire {y} of total faculty")

                    grad = (y - prev_y) / (x - prev_x)

                    x_estimate = (0.8 - prev_y) / grad + prev_x

                    print(x_estimate)
                    break

                prev_x = x
                prev_y = y

        return gini_in, gini_out

    draw_lcurve(g, pcfg.color_ut)
    draw_lcurve(d, pcfg.color_kt)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='x', which='major', pad=15)

    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    ax.plot([0, 1], [0, 1], c='black', alpha=pcfg.alpha)


def fac_type():

    from analysis.doc_type import calc_doc_type_total

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='x', which='major', labelsize=pcfg.small_tick_size)
    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)

    ax.set_ylabel("Fraction of faculty", fontsize=pcfg.xlabel_size)
    ax.set_ylim(0, 1)

    stats = calc_doc_type_total()

    for iden in identifiers:

        stat = stats[iden]
        field = iden.split('_')[1]

        if field == 'ComputerScience':
            field = 'Computer\nScience'

        kr = stat["KR"] / stat["Total"]
        us = stat["US"] / stat["Total"]
        others = stat["Others"] / stat["Total"]

        print(iden, us)

        ax.bar(field, us, color=pcfg.color_ut, alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(field, others, color=pcfg.color_et, bottom=us, alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(field, kr, color=pcfg.color_kt, bottom=us+others, alpha=pcfg.alpha, edgecolor='black', linewidth=2)

    handles = [Patch(facecolor=pcfg.color_ut, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.color_et, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.color_kt, alpha=pcfg.alpha, edgecolor='black', linewidth=3),]

    labels = ["US-trained faculty", "Others",  "KR-trained faculty"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False)

    # plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"./{doc}/fac_type"))
    plt.clf()


def fac_type_us_kr(iden=iden_default):

    net = construct_network().get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize * 1.3, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.set_yticks(range(1, 11))

    ax.set_xlabel("Fraction of faculty", fontsize=pcfg.xlabel_size)
    ax.set_ylabel("Rank decile", fontsize=pcfg.xlabel_size)

    _fac_type_us_kr(ax, net)

    handles = [Patch(facecolor=pcfg.color_ut, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.color_et, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.color_kt, alpha=pcfg.alpha, edgecolor='black', linewidth=3),]

    labels = ["US-trained faculty", "Others",  "KR-trained faculty"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False)
    
    plt.savefig(fig_path(f"./{doc}/fac_type_grouped_{iden}"))
    

def _fac_type_us_kr(ax, g):

    from analysis.doc_type import _calc_doc_type
    stats = _calc_doc_type(g)

    x_co = [i + 1 for i in range(10)]
    y_co_us = [stat["US"] / stat["Total"] for stat in stats]
    y_co_kr = [stat["KR"] / stat["Total"] for stat in stats]
    y_co_others = [stat["Others"] / stat["Total"] for stat in stats]

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.25))

    # ax.invert_yaxis()

    bars = ax.barh(x_co, y_co_us, color=pcfg.color_ut,
            alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_others, color=pcfg.color_et, left=y_co_us,
            alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_kr, color=pcfg.color_kt,
            left=[us + others for us, others in zip(y_co_us, y_co_others)],
            alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    
    for bar in bars[-3:]:
        width = bar.get_width()
        ax.text(width / 2, bar.get_y() + bar.get_height() / 2 - 0.05,
                f'{width: .2f}', ha='center', va='center')
        

def trained_rank_v_placed_rank(dec="Overall", average=False, normalize=False, iden=iden_default):

    net = construct_network().get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    xlabel = r"$\pi_{placed}$"
    ylabel = r"$\pi_{trained}$" if not average else r"$\pi_{trained, avg}$"

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    _trained_rank_v_placed_rank(ax, net, dec, average, normalize)

    handles = [Patch(facecolor='mediumblue', edgecolor='black', alpha=pcfg.alpha, linewidth=2),
               Patch(facecolor='tomato', edgecolor='black', alpha=pcfg.alpha, linewidth=2)]
    
    labels = ['US-trained faculty', 'KR-trained faculty']

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False, fontsize=pcfg.legend_size)

    # plt.show()
    plt.savefig(fig_path(f"./{doc}/trank_v_prank_{dec}_average({average})_normalize({normalize})_{iden}"))


def _trained_rank_v_placed_rank(ax, g, dec, average, normalize):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    trained_ranks_ut = []
    placed_ranks_ut = []

    trained_ranks_kt = []
    placed_ranks_kt = []

    max_trained_rank_ut = max_wapman_ranks[iden.split('_')[1]]
    max_trained_rank_kt = max_domestic_ranks[iden.split('_')[1]]

    max_placed_rank = max_domestic_ranks[iden.split('_')[1]]

    interval = decades_to_interval(dec)

    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if src_nation not in ['US', 'KR'] or dst_nation != 'KR':
            continue

        phd_granted_year = data['phd_end_year']

        if phd_granted_year is None:
            continue

        if not interval[0] <= phd_granted_year < interval[1]:
            continue
        
        if src_nation == 'US':
            t_rank = g.nodes[src_id][f'{abbrev}_rank_wapman']

        else:
            t_rank = g.nodes[src_id][f'{abbrev}_rank_domestic']

        if t_rank is None:
            continue

        p_rank = g.nodes[dst_id][f'{abbrev}_rank_domestic']

        if p_rank is None:
            continue
        
        if src_nation == 'US':
            trained_ranks_ut.append(t_rank)
            placed_ranks_ut.append(p_rank)
        
        else:
            trained_ranks_kt.append(t_rank)
            placed_ranks_kt.append(p_rank)

    if normalize:
        trained_ranks_ut = list(np.array(trained_ranks_ut) / max_trained_rank_ut)
        placed_ranks_ut = list(np.array(placed_ranks_ut) / max_placed_rank)

        trained_ranks_kt = list(np.array(trained_ranks_kt) / max_trained_rank_kt)
        placed_ranks_kt = list(np.array(placed_ranks_kt) / max_placed_rank)

    if average:

        gathering = {}

        for t, p in zip(trained_ranks_ut, placed_ranks_ut):

            if p not in gathering:
    
                gathering[p] = []
                gathering[p].append(t)

            else:
                gathering[p].append(t)

        gathering_mean = {}

        for p, ts in gathering.items():
            gathering_mean[p] = np.mean(ts)

        trained_ranks_ut = list(gathering_mean.values())
        placed_ranks_ut = list(gathering_mean.keys())

        gathering = {}

        for t, p in zip(trained_ranks_kt, placed_ranks_kt):

            if p not in gathering:
    
                gathering[p] = []
                gathering[p].append(t)

            else:
                gathering[p].append(t)

        gathering_mean = {}

        for p, ts in gathering.items():
            gathering_mean[p] = np.mean(ts)

        trained_ranks_kt = list(gathering_mean.values())
        placed_ranks_kt = list(gathering_mean.keys())

    xlim = max_placed_rank
    ylim = max([max_trained_rank_ut, max_trained_rank_kt])

    if normalize:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xticks(np.arange(0, 1.1, 0.25))
        ax.set_yticks(np.arange(0, 1.1, 0.25))

    else:
        ax.set_xlim(0, xlim + 1)
        ax.set_ylim(0, ylim + 1)

        ax.set_xticks(range(0, xlim + 1, 25))
        ax.set_yticks(range(0, ylim + 1, 25))

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.scatter(placed_ranks_ut, trained_ranks_ut, marker='o',
               alpha=pcfg.alpha, s=150, c=pcfg.color_ut,
               edgecolor='black')
    
    ax.scatter(placed_ranks_kt, trained_ranks_kt, marker='o',
               alpha=pcfg.alpha, s=150, c=pcfg.color_kt,
               edgecolor='black')
    
    if all(data is not None for data in [trained_ranks_ut, placed_ranks_ut, trained_ranks_kt, placed_ranks_kt]):

        placed_ranks = np.array(placed_ranks_ut)
        trained_ranks = np.array(trained_ranks_ut)

        results = scipy.stats.linregress(placed_ranks, trained_ranks)
        slope_ut = results.slope
        intercept_ut = results.intercept
        r_value_ut = results.rvalue
        p_value_ut = results.pvalue
        std_err = results.stderr

        eq_ut = f"y={slope_ut:.2e}x + {intercept_ut:.2e}"

        print(f"=== US-trained ({iden}, {dec}) ===")
        print(f"Slope: {slope_ut}")
        print(f"Intercept: {intercept_ut}")
        print(f"R-squared: {r_value_ut**2}")
        print(f"P-value: {p_value_ut}")
        print(f"Standard error: {std_err}")
        
        x_vals = np.array(ax.get_xlim())
        y_vals = slope_ut * x_vals + intercept_ut
        ax.plot(x_vals, y_vals, color=pcfg.color_ut, linestyle='-', linewidth=2)

        placed_ranks = np.array(placed_ranks_kt)
        trained_ranks = np.array(trained_ranks_kt)

        results = scipy.stats.linregress(placed_ranks, trained_ranks)
        slope_kt = results.slope
        intercept_kt = results.intercept
        r_value_kt = results.rvalue
        p_value_kt = results.pvalue
        std_err = results.stderr

        eq_kt = f"y={slope_kt:.2e}x + {intercept_kt:.2e}"

        print(f"=== KR-trained ({iden}, {dec}) ===")
        print(f"Slope: {slope_kt}")
        print(f"Intercept: {intercept_kt}")
        print(f"R-squared: {r_value_kt**2}")
        print(f"P-value: {p_value_kt}")
        print(f"Standard error: {std_err}")
        
        x_vals = np.array(ax.get_xlim())
        y_vals = slope_kt * x_vals + intercept_kt
        ax.plot(x_vals, y_vals, color=pcfg.color_kt, linestyle='-', linewidth=2)

        label_ut = f'<US-trained faculty>\n{eq_ut}'
        label_kt = f'<KR-trained faculty>\n{eq_kt}'

        label_to_put = '\n'.join([label_ut, label_kt])

        ax.text(0.95, 0.05,
                label_to_put,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.stat_size, bbox=dict(facecolor='white', alpha=0.8))
        

def mobility_kr_us(normalize=True, iden=iden_default):

    assert (isinstance(normalize, bool))

    net = construct_network().get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ylabel = 'Fraction' if normalize else 'Number'

    # axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    # fig.supxlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    _mobility_kr_us_both(ax, net, normalize)

    handles = [Patch(facecolor='mediumblue', edgecolor='black', alpha=pcfg.alpha, linewidth=2),
               Patch(facecolor='tomato', edgecolor='black', alpha=pcfg.alpha, linewidth=2)]
    
    labels = ['Up-hires', 'Down-hires']

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False, fontsize=pcfg.legend_size)

    plt.savefig(fig_path(f"./{doc}/mobility_us_kr_normalize({normalize})_{iden}"))


def _mobility_kr_us_both(ax, g, normalize):

    from analysis.mobility import calc_mobility

    stats = calc_mobility(g)

    print(stats)

    category = ['KR-trained\nKR-placed',
                'US-trained\nKR-placed',
                'KR-trained\nUS-placed']
    
    x = np.arange(len(category))

    ktkp_up = stats['KR2KR'][0] / stats['KR2KR'][3] if normalize else stats['KR2KR'][0]
    ktkp_do = stats['KR2KR'][2] / stats['KR2KR'][3] if normalize else stats['KR2KR'][2]

    utkp_up = stats['US2KR'][0] / stats['US2KR'][3] if normalize else stats['US2KR'][0]
    utkp_do = stats['US2KR'][2] / stats['US2KR'][3] if normalize else stats['US2KR'][2]

    ktup_up = stats['KR2US'][0] / stats['KR2US'][3] if normalize else stats['KR2US'][0]
    ktup_do = stats['KR2US'][2] / stats['KR2US'][3] if normalize else stats['KR2US'][2]

    data_up = [ktkp_up, utkp_up, ktup_up]
    data_do = [ktkp_do, utkp_do, ktup_do]

    for c, data in zip(category, data_up):
        print(c, data)

    ax.tick_params(axis='x', which='major', labelsize=pcfg.small_tick_size)
    ax.tick_params(axis='x', labelrotation=30)

    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)

    bar_width = 0.25

    ax.bar(x - bar_width / 2, data_up, width=bar_width, color='mediumblue',
           alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    ax.bar(x + bar_width / 2, data_do, width=bar_width, color='tomato',
           alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(category)

    if normalize:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.25))
    else:
        ax.set_ylim(0, 800)
        ax.set_yticks(range(0, 801, 200))


def placed_rank_density(range_trained=(0, 20), iden=iden_default):

    net = construct_network().get(iden)

    if net is None:
        return None

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlabel("Rank decile", fontsize=pcfg.xlabel_size)
    ax.set_ylabel("Counts", fontsize=pcfg.ylabel_size)

    modes = ['utkp', 'ktkp']
    styles = ['-', '-']
    colors = ['mediumblue', 'tomato']

    for i in range(len(modes)):

        m = modes[i]
        ls = styles[i]
        c = colors[i]

        _placed_rank_density(ax, net, range_trained, m, c=c, ls=ls)

    # ax2.set_ylabel("Counts", fontsize=pcfg.ylabel_size)

    handles = [Patch(facecolor=pcfg.color_ut, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Line2D([0], [0], color=pcfg.color_ut, linestyle='-', linewidth=4),
               Patch(facecolor=pcfg.color_kt, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Line2D([0], [0], color=pcfg.color_kt, linestyle='-', linewidth=4)]

    labels = ["US-trained faculty (count)", "US-trained faculty (density)",
              "KR-trained faculty (count)", "KR-trained faculty (density)"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)

    plt.savefig(fig_path(f"./{doc}/placed_rank_density_{range_trained[0]}_{range_trained[1]}_{iden}"))
    

def _placed_rank_density(ax, g, range_trained, mode, ls=None, c=None):
    
    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    placed_ranks = []

    match mode:
        case 'utkp':
            max_trained_rank = max_wapman_ranks[iden.split('_')[1]]
            max_placed_rank = max_domestic_ranks[iden.split('_')[1]]

            trank_str = f'{abbrev}_rank_wapman'
            prank_str = f'{abbrev}_rank_domestic'

            right_src_nation = 'US'
            right_dst_nation = 'KR'

        case 'ktup':
            max_trained_rank = max_domestic_ranks[iden.split('_')[1]]
            max_placed_rank = max_wapman_ranks[iden.split('_')[1]]

            trank_str = f'{abbrev}_rank_domestic'
            prank_str = f'{abbrev}_rank_wapman'

            right_src_nation = 'KR'
            right_dst_nation = 'US'

        case 'ktkp':
            max_trained_rank = max_domestic_ranks[iden.split('_')[1]]
            max_placed_rank = max_domestic_ranks[iden.split('_')[1]]

            trank_str = f'{abbrev}_rank_domestic'
            prank_str = f'{abbrev}_rank_domestic'

            right_src_nation = 'KR'
            right_dst_nation = 'KR'

        case _:
            return

    num_bins = 10
    bins = []
    
    ranges = divide_by_groups(max_placed_rank, num_bins)[::-1]

    for i, value in enumerate(ranges):
        
        bins.append(value[0])

        if i == len(ranges) - 1:
            bins.append(value[1])

    lower_b, upper_b = convert_per_to_ranks(max_trained_rank, range_trained)

    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if src_nation != right_src_nation or dst_nation != right_dst_nation:
            continue

        t_rank = g.nodes[src_id][trank_str]

        if t_rank is None:
            continue

        if not lower_b <= t_rank < upper_b:
            continue

        p_rank = g.nodes[dst_id][prank_str]

        if p_rank is None:
            continue

        placed_ranks.append(p_rank)

    data = np.array(placed_ranks)

    x = np.linspace(1, max_placed_rank, 1000)

    if len(data) > 1:
        kde = scipy.stats.gaussian_kde(data, bw_method=0.5)
        y = kde(x)
    else:
        y = [0 for co in x]

    # ax2 = ax.twinx()

    if range_trained == (0, 20):
        ylim = 150
        yticks = range(0, 151, 50)

    elif range_trained == (20, 40):
        ylim = 50
        yticks = range(0, 51, 25)

    elif range_trained == (40, 60):
        ylim = 25
        yticks = range(0, 26, 25)

    else:
        ylim = 10
        yticks = range(0, 11, 5)
        
    ax.set_xlim(1, max_placed_rank)
    ax.set_ylim(0, ylim)
    ax.set_yticks(yticks)
    
    ax.invert_xaxis()

    if c is None:
        c = pcfg.colors[iden][1]

    if ls is None:
        ls = '--'

    counts, bins, _ = ax.hist(data, color=c, alpha=pcfg.alpha,
                            edgecolor='black', linewidth=3, bins=bins)
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    ax.set_xticks(bin_centers)  # Set positions for x-ticks (1 to 10)
    ax.set_xticklabels([str(11 - i) for i in range(1, 11)]) 
    ax.plot(x, np.array(y) * ylim / 0.125, label='Density', color=c, linestyle=ls, linewidth=4)

    # max_density = int(max(y) / 0.05) * 0.05

    # ax.set_ylim(0, 0.125)
    # ax.set_yticks(np.arange(0, 0.126, 0.125))


def hires_stat(net_type='domestic', iden=iden_default, ax_export=None):

    net = construct_network(net_type=net_type, data=True).get(iden)

    if net is None:
        return
    
    from analysis.hires import _calc_hires

    stat = _calc_hires(net, net_type)

    plt.rc('font', **pcfg.fonts)

    if ax_export is None:
        fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    else:
        ax = ax_export

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    do = stat['Down'] / stat['Total']
    se = stat['Self'] / stat['Total']
    up = stat['Up'] / stat['Total']

    ax.set_ylim(0, 1)

    bar_width = 0.25
    x_pos = 0

    x_pos_1 = x_pos
    x_pos_2 = x_pos + bar_width
    x_pos_3 = x_pos + 2 * bar_width

    ax.bar(x_pos_1, 0, width=bar_width)

    ax.bar(x_pos_2, do, color=pcfg.colors_cs[0], width=bar_width * 1.5,
           alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    ax.bar(x_pos_2, se, color=pcfg.colors_cs[2], width=bar_width * 1.5,
           bottom=do, alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    ax.bar(x_pos_2, up, color=pcfg.colors_cs[4], width=bar_width * 1.5,
           bottom=do+se, alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    
    ax.bar(x_pos_3, 0, width=bar_width)

    ax.set_xticks([x_pos_1, x_pos_2, x_pos_3], ['' for _ in range(3)])

    handles = [Patch(facecolor=pcfg.colors[iden][0], alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.colors[iden][2], alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.colors[iden][4], alpha=pcfg.alpha, edgecolor='black', linewidth=3)]

    labels = ['Down hires', 'Self hires', 'Up hires']

    if ax_export is None:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=3, frameon=False)
        plt.savefig(fig_path(f'./{doc}/hires_stat_{net_type}_{iden}'))

        plt.clf()


def hires_z(net_type='domestic', iden=iden_default):

    net = construct_network(net_type=net_type, data=True).get(iden)

    if net is None:
        return
    
    from analysis.hires import _calc_hires_z, _calc_hires

    stat = _calc_hires_z(net, net_type=net_type)
    stat_net = _calc_hires(net, net_type=net_type)

    print(stat)

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_ylabel('Z-scores', fontsize=pcfg.ylabel_size)

    ax.set_ylim(-10, 20)
    ax.set_yticks(range(-10, 21, 5))

    ax.tick_params(axis='x', which='major', labelsize=pcfg.small_tick_size)
    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)

    bar_width = 0.25
    x_pos = 0

    for i in range(2):
    
        x_pos_1 = x_pos + i
        x_pos_2 = x_pos_1 + bar_width / 2
        x_pos_3 = x_pos_1 + 2 * bar_width / 2

        if i == 1:
            ax.set_xticks([x_pos_1, x_pos_2, x_pos_3], ['Down hires', 'Self hires', 'Up hires'])

        do_z = (stat_net['Down'] / stat_net['Total'] - stat['Down'][0]) / stat['Down'][1] if i == 1 else 0
        se_z = (stat_net['Self'] / stat_net['Total'] - stat['Self'][0]) / stat['Self'][1] if i == 1 else 0
        up_z = (stat_net['Up'] / stat_net['Total'] - stat['Up'][0]) / stat['Up'][1] if i == 1 else 0

        ax.bar(x_pos_1, do_z, width=bar_width/2, color=pcfg.colors[iden][0],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_2, se_z, width=bar_width/2, color=pcfg.colors[iden][2],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_3, up_z, width=bar_width/2, color=pcfg.colors[iden][4],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)

    ax.axhline(0, color='black', linewidth=1)
    ax.tick_params(axis='x', labelrotation=45)

    ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper left')

    hires_stat(net_type=net_type, iden=iden, ax_export=ax_inset)

    handles = [Patch(facecolor=pcfg.colors[iden][0], alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.colors[iden][2], alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.colors[iden][4], alpha=pcfg.alpha, edgecolor='black', linewidth=3)]

    labels = ['Down hires', 'Self hires', 'Up hires']

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=3, frameon=False)
    
    plt.savefig(fig_path(f'./{doc}/hires_z_{net_type}_{iden}'))
    plt.clf()


def rank_move_grouped(net_type='domestic', inset=False, iden=iden_default):

    assert (net_type in ['global', 'domestic'])

    net = construct_network(net_type=net_type).get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)

    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    groups = [(1, 5), (3, 5), (5, 5)]
    styles = ['-', ':', '--']

    if inset:
        ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
        ax_inset.set_xlim(-1, 1)

    for group, s in zip(groups, styles):
        _rank_move(net, None, ax, net_type=net_type, group=group, export_style=s)

        if inset:
            _rank_move(net, None, ax_inset, net_type=net_type, group=group, export_style=s, normalize_sep=False, for_inset=True)

    x_label = 'Relative Movement of Faculty\n(Normalized)'
    y_label = 'Density'

    ax.set_ylabel(y_label, fontsize=pcfg.ylabel_size)
    ax.set_xlabel(x_label, fontsize=pcfg.xlabel_size)
    
    handles = [Line2D([0], [0], color=pcfg.colors[iden][1], linestyle=s, linewidth=4, alpha=pcfg.alpha) for s in styles]

    labels = ["Top quintile", "Middle quntile", "Bototm quntile"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=3, frameon=False)

    ax.set_xlim(-1, 1)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))

    plt.savefig(fig_path(f'./{doc}/rank_move_grouped_{net_type}_inset({inset})_{iden}'))
    plt.clf()


def _rank_move(g, d, ax, net_type, group=None, export_style=None, export_color=None, normalize_sep=True, for_inset=False):

    from analysis.rank_move import calc_rank_move

    iden = g.name

    if net_type == 'both':

        moves_g = calc_rank_move(g, net_type='global', group=group)
        moves_d = calc_rank_move(d, group=group)

        moves = [moves_g, moves_d]
        linestyles = ['-', ':']

    else:

        moves_g = calc_rank_move(g, net_type=net_type, group=group)

        moves = [moves_g]
        linestyles = ['-']

    bins = np.linspace(-1, 1, 40)

    x = (bins[:-1] + bins[1:]) / 2

    for move, style in zip(moves, linestyles):

        hist, _ = np.histogram(move, bins=bins)
        normalized_hist = hist / np.sum(hist)

        style_to_put = style if export_style is None else export_style
        color_to_put = pcfg.colors[iden][0] if export_color is None else export_color

        (x_interp, y_interp) = _interhistogram(x, normalized_hist)

        if not normalize_sep:

            ratio = len(move) / g.number_of_edges()

            print(ratio)

            y_interp = [y * ratio for y in y_interp]

        ax.plot(x_interp, y_interp, color=color_to_put, linewidth=5,
                alpha=pcfg.alpha, linestyle=style_to_put)

    if for_inset:
        ax.set_yticks([])

    else:

        ymin = 0    
        ymax = 20

        ax.set_ylim(ymin, ymax)
        ax.set_yticks(range(ymin, ymax + 1, 5))


def radg_by_rank(dist_type='rank', ver=1, add_reg=False, iden=iden_default):

    assert (dist_type in ['geo', 'rank'])
    assert (ver in [1, 2])

    net = construct_network(net_type='domestic').get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    xlabel = 'Rank'

    if ver == 1:
        ylabel = 'RMS displacement (km)' if dist_type == 'geo' else 'RMS rank move'

    else:
        ylabel = 'Radius of gyration (km)' if dist_type == 'geo' else 'Radius of gyration (rank)'

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    _radg_by_rank(ax, net, dist_type, ver, add_reg)

    if add_reg:

        handles = [Line2D([0], [0], color=pcfg.color_ut, linewidth=6, alpha=pcfg.alpha),
                   Line2D([0], [0], color=pcfg.color_kt, linewidth=6, alpha=pcfg.alpha)]
        
        labels = ['1st order fit', '2nd order fit']

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9),
                   ncol=2, frameon=False, fontsize=pcfg.legend_size)

    plt.savefig(fig_path(f"./{doc}/radg_{dist_type}_v_rank_ver{ver}_reg({add_reg})_{iden}"))


def _radg_by_rank(ax, g, dist_type, ver, add_reg):

    from analysis.geography import _calc_radg, _calc_radg_ver2
    from parse.queries_integration import get_abbrev 

    iden = g.name
    abbrev = get_abbrev(iden)

    func_calc_radg = _calc_radg if ver == 1 else _calc_radg_ver2

    radgs = func_calc_radg(g, dist_type=dist_type)

    xco = []
    yco = []
    
    for id, data in g.nodes(data=True):

        if id not in radgs:
            continue

        rank = data[f'{abbrev}_rank_domestic']

        xco.append(rank)
        yco.append(radgs[id])

    ax.scatter(xco, yco, marker='o',
               alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
               edgecolor='black')
    
    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    max_rank = max_domestic_ranks[iden.split('_')[1]]
    ax.set_xlim(0, max_rank)
    ax.set_xticks(range(0, max_rank + 1, 20))

    if add_reg:

        xco = np.array(xco)
        yco = np.array(yco)
        sorted_indices = np.argsort(xco)

        xco = xco[sorted_indices]
        yco = yco[sorted_indices]

        xco_with_const = sm.add_constant(xco)

        # 1st order (linear) regression
        linear_model = sm.OLS(yco, xco_with_const)
        linear_results = linear_model.fit()
        ax.plot(xco, linear_results.predict(xco_with_const),
                color=pcfg.color_ut, linewidth=6, alpha=pcfg.alpha)

        slope_first = linear_results.params[1]
        intercept_first = linear_results.params[0]
        rsquared_first = linear_results.rsquared
        pvalue_first = linear_results.pvalues[1]
        std_first = linear_results.bse[1]

        eq_first = rf"{intercept_first:.2e} + {slope_first:.2e}x"

        print(f"1st Order Fit for {iden}:")
        print(f"Slope: {slope_first}")
        print(f"Intercept: {intercept_first}")
        print(f"R-squared: {rsquared_first}")
        print(f"P-value: {pvalue_first}")
        print(f"Standard error: {std_first}")

        xco_quad = np.column_stack((xco, np.power(xco, 2)))
        xco_quad_with_const = sm.add_constant(xco_quad)
        quadratic_model = sm.OLS(yco, xco_quad_with_const)
        quadratic_results = quadratic_model.fit()
        ax.plot(xco, quadratic_results.predict(xco_quad_with_const),
                color=pcfg.color_kt, linewidth=6, alpha=pcfg.alpha)

        coeff_second = quadratic_results.params
        rsquared_second = quadratic_results.rsquared
        pvalue_second = quadratic_results.pvalues
        std_second = quadratic_results.bse

        eq_second = f"{coeff_second[0]:.2e} + {coeff_second[1]:.2e}x + {coeff_second[2]:.2e}x^2"

        print(f"2nd Order Fit for {iden}:")
        print(f"Coefficients: {coeff_second}")
        print(f"R-squared: {rsquared_second}")
        print(f"P-values: {pvalue_second}")
        print(f"Standard errors: {std_second}")

        label_first = f'<1st order fit>\nEq: {eq_first}\n$R^2$: {rsquared_first:.2e}\nP-value: {pvalue_first:.2e}'
        label_second = f'<2nd order fit>\nEq: {eq_second}\n$R^2$: {rsquared_second:.2e}\nP-values: {pvalue_second[0]:.2e}/{pvalue_second[1]:.2f}/{pvalue_second[2]:.2f}'

        label_to_put = '\n'.join([label_first, label_second])

        ax.text(0.95, 0.95,
                label_to_put,
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.stat_size - 10, bbox=dict(facecolor='white', alpha=0.8))

    if dist_type == 'geo':
        ax.set_ylim(0, 500)
        ax.set_yticks(range(0, 501, 100))

    else:
        pass
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))


def us_trained_rank_v_dist_from_seoul(cleaning='raw', normalize=True, to_exclude=['ist', 'flagship'], iden=iden_default):

    net = construct_network().get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='x', which='major', pad=15)

    xlabel = 'Distance from Seoul (km)'

    if cleaning == 'raw':
        ylabel = r"$\pi_{trained}$"
    elif cleaning == 'average':
        ylabel = r"$\pi_{trained, avg}$"
    elif cleaning == 'median':
        ylabel = r"$\pi_{trained, median}$"

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 2.5, pad=0.1, sharex=ax)

    _us_trained_rank_v_dist_from_seoul(ax, ax_histx, net, cleaning, normalize, to_exclude)
    _us_trained_rank_v_dist_from_seoul(ax, ax_histx, net, cleaning, normalize, to_exclude, fac_type='ktkp')
    
    handles = [Patch(facecolor=pcfg.color_ut, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.color_kt, alpha=pcfg.alpha, edgecolor='black', linewidth=3)]

    labels = ["US-trained faculty", "KR-trained faculty"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False)

    plt.savefig(fig_path(f"./{doc}/us_trained_rank_v_dist_{cleaning}_exclude_{'_'.join(to_exclude)}_{iden}"))
    

def _us_trained_rank_v_dist_from_seoul(ax, ax_histx, g, cleaning, normalize, to_exclude, fac_type='utkp'):

    assert (fac_type in ['utkp', 'ktkp'])

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    trained_ranks = []
    career_years = []

    exclude_ist = True if 'ist' in to_exclude else False
    exclude_flagship = True if 'flagship' in to_exclude else False

    max_trained_rank = max_wapman_ranks[iden.split('_')[1]] if fac_type == 'utkp' else max_domestic_ranks[iden.split('_')[1]]

    t_rank_key = f"{abbrev}_rank_wapman" if fac_type == 'utkp' else f"{abbrev}_rank_domestic"
    target_src_nation = 'US' if fac_type == 'utkp' else 'KR'
    target_dst_nation = 'KR'
    color = pcfg.color_ut if fac_type == 'utkp' else pcfg.color_kt
    
    for src_id, dst_id, data in g.edges(data=True):

        if exclude_ist:
            if dst_id in [2, 12, 15, 48, 229, 282]:
                continue

        if exclude_flagship:
            if dst_id in [4, 6, 9, 10, 11, 54, 74, 97, 128, 250, 355]:
                continue

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if src_nation != target_src_nation or dst_nation != target_dst_nation:
            continue

        placed_dist = g.nodes[dst_id]['distance_to_seoul']

        if placed_dist is None:
            continue

        t_rank = g.nodes[src_id][t_rank_key]

        if t_rank is None:
            continue

        if normalize:
            t_rank = t_rank / max_trained_rank

        trained_ranks.append(t_rank)
        career_years.append(placed_dist)

    if cleaning == 'average' or cleaning == 'median':

        gathering = {}

        for t, y in zip(trained_ranks, career_years):

            if y not in gathering:
    
                gathering[y] = []
                gathering[y].append(t)

            else:
                gathering[y].append(t)

        gathering_cleaned = {}

        if cleaning == 'average':
            for y, ts in gathering.items():
                gathering_cleaned[y] = np.mean(ts)
        else:
            for y, ts in gathering.items():
                gathering_cleaned[y] = np.median(ts)

        trained_ranks = list(gathering_cleaned.values())
        career_years = list(gathering_cleaned.keys())

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='x', labelrotation=45)

    ax.set_xlim(0, 500)

    if normalize:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, max_trained_rank + 1)

    ax.set_xticks(range(0, 501, 100))

    ax.invert_yaxis()

    ax.scatter(career_years, trained_ranks, marker='o',
                alpha=pcfg.alpha, s=150, c=color,
                edgecolor='black')

    sns.kdeplot(x=career_years, ax=ax_histx, fill=True, color=color, alpha=0.2)

    ax_histx.yaxis.tick_right()
    ax_histx.tick_params(axis='x', which='both', bottom=False, top=False, labelsize=0)
    ax_histx.tick_params(axis='y', which='both', left=False, right=False, labelsize=30)
    ax_histx.set_yticks(np.arange(0, 0.016, 0.005))
    ax_histx.set_ylabel('')


def rank_v_dist(iden=iden_default):

    net = construct_network(net_type='domestic').get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ylabel = r"$\pi_{KR}$"
    xlabel = 'Distance from Seoul (km)'

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    _rank_v_dist(ax, net)

    plt.savefig(fig_path(f"./{doc}/rank_v_dist_{iden}"))


def _rank_v_dist(ax, g):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)
    max_rank = max_domestic_ranks[iden.split('_')[1]]

    ist_id = [2, 12, 15, 48, 229, 282]
    areas = [area_seoul, area_capital, area_metro, area_others]

    ranks = [[] for _ in range(5)]
    dists = [[] for _ in range(5)]

    for id, data in g.nodes(data=True):

        name = data['name']

        if name is None:
            return
        
        rank = data[f'{abbrev}_rank_domestic']
        dist = data['distance_to_seoul']
        
        if id in ist_id:
            ranks[4].append(rank)
            dists[4].append(dist)
            continue
        
        region = name.split(',')[1].strip().lower()

        for i, area in enumerate(areas):

            if region in area:
                ranks[i].append(rank)
                dists[i].append(dist)
                break

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlim(0, 500)
    ax.set_ylim(0, max_rank + 1)

    ax.set_xticks(range(0, 501, 100))

    # ax.invert_xaxis()
    ax.invert_yaxis()

    for i in range(5):

        r = ranks[i]
        d = dists[i]

        ax.scatter(d, r, marker='o',
                    alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
                    edgecolor='black')


def rank_v_dist_hmap(iden=iden_default):

    net = construct_network(net_type='domestic').get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ylabel = r"$\pi_{KR}$"
    xlabel = 'Distance from Seoul (km)'

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    _rank_v_dist_hmap(ax, net)

    plt.savefig(fig_path(f"./{doc}/rank_v_dist_hmap_{iden}"))


def _rank_v_dist_hmap(ax, g):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)
    max_rank = max_domestic_ranks[iden.split('_')[1]]

    rgb_cs = (0.9568627450980393, 0.41568627450980394, 0.3058823529411765)

    ranks = []
    dists = []

    for id, data in g.nodes(data=True):
        name = data['name']
        if name is None:
            continue
        
        rank = data[f'{abbrev}_rank_domestic']
        dist = data['distance_to_seoul']

        if rank is not None and dist is not None:
        
            ranks.append(rank)
            dists.append(dist)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    # Create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(dists, ranks, bins=[10, 10], range=[[0, 500], [0, max_rank + 1]])

    # Plot the 2D histogram using imshow

    colors = [(1, 1, 1), rgb_cs]  # Red gradient
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap=custom_cmap)

    ax.set_xlim(0, 500)
    ax.set_ylim(0, max_rank + 1)

    ax.set_xticks(range(0, 501, 100))

    # ax.invert_xaxis()
    ax.invert_yaxis()

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Institutions', fontsize=pcfg.ylabel_size - 10)
        

def rank_v_region(iden=iden_default):

    net = construct_network(net_type='domestic').get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='x', which='major', labelsize=pcfg.small_tick_size)
    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)

    ylabel = r"$\pi_{KR}$"
    # xlabel = 'Region'

    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    _rank_v_region(ax, net)

    plt.savefig(fig_path(f"./{doc}/rank_v_region_{iden}"))


def _rank_v_region(ax, g):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    ist_id = [2, 12, 15, 48, 229, 282]
    areas = [area_seoul, area_capital, area_metro, area_others]

    ranks = [[] for _ in range(5)]

    for id, data in g.nodes(data=True):

        name = data['name']

        if name is None:
            return
        
        rank = data[f'{abbrev}_rank_domestic']
        
        if id in ist_id:
            ranks[4].append(rank)
            continue
        
        region = name.split(',')[1].strip().lower()

        for i, area in enumerate(areas):

            if region in area:
                ranks[i].append(rank)
                break

    bplot = ax.boxplot(ranks, patch_artist=True, vert=True)

    palette = [pcfg.colors[iden][1] for _ in range(5)]

    for patch, color in zip(bplot['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(pcfg.alpha)
        
    # ax.set_xticks([1, 2, 3, 4, 5], labels)

    ax.invert_yaxis()

    ax.set_xticklabels(['Seoul', 'Capital\narea', 'Metro-\npolitan cities', 'Others', '-IST'])
    # ax.set_yticklabels([])

    ax.tick_params(axis='x', which='major', labelsize=patch_size)
    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)


def self_hires_v_rank_grouped(net_type='domestic', annotate_bs=True, iden=iden_default):
 
    net = construct_network(net_type=net_type).get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlabel("Rank decile", fontsize=pcfg.xlabel_size)
    ax.set_ylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    _self_hires_v_rank_grouped(ax, net, annotate_bs)

    if annotate_bs:

        from matplotlib.lines import Line2D

        handles = [Line2D([0], [0], color='red', marker='s', markersize=13, alpha=pcfg.alpha)]
        labels = ["Fraction of self-hiring own bachelor's graduates"]

        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.9), ncol=1, frameon=False,
                   fontsize=pcfg.legend_size)

    plt.savefig(fig_path(f"./{doc}/self_hires_v_rank_grouped_{net_type}_annotate({annotate_bs})_{iden}"))


def _self_hires_v_rank_grouped(ax, g, annotate_bs):

    iden = g.name
    field = iden.split('_')[1]
    abbrev = qsi.get_abbrev(iden)

    max_rank = max_domestic_ranks[field]
    num_bins = 10

    from analysis.doc_type import divide_by_groups

    ranges = divide_by_groups(max_rank, num_bins)

    fractions = {i: [0, 0, 0, 0] for i in range(len(ranges))}

    for src_id, dst_id, data in g.edges(data=True):

        drank = g.nodes[dst_id][f"{abbrev}_rank_domestic"]

        if drank is None:
            continue

        is_self_hire = 0
        is_bs_self_hire = 0
        is_phd_self_hire = 0

        bs_id = data['bs_inst_id']
        
        if bs_id is not None and bs_id == dst_id:

            is_self_hire = 1
            is_bs_self_hire = 1

        elif src_id == dst_id:
            is_self_hire = 1
            is_phd_self_hire = 1

        for i, value in enumerate(ranges):
            if value[0] <= drank < value[1]:
                fractions[i][0] += is_self_hire
                fractions[i][1] += is_bs_self_hire
                fractions[i][2] += is_phd_self_hire
                fractions[i][3] += 1

    frac_to_put = []
    frac_bs_to_put = []
    frac_phd_to_put = []

    for i, value in sorted(fractions.items()):

        frac = value[0] / value[3]
        frac_bs = value[1] / value[3]
        frac_phd = value[2] / value[3]

        frac_to_put.append(frac)
        frac_bs_to_put.append(frac_bs)
        frac_phd_to_put.append(frac_phd)

        title = f"Decile {i + 1}"

        if i == 0:
            title += " (Lowest prestige)"
        elif i == 9:
            title += " (Highest prestige)"

        print(title)
        print(f"BS: {frac_bs: .3f}\nPhD: {frac_phd: .3f}\nBS+PhD: {frac: .3f}")

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    categories = [f"{i + 1}" for i in range(len(ranges))]

    ax.bar(categories, frac_to_put, alpha=pcfg.alpha,
           color=pcfg.colors[iden][1], edgecolor='black', linewidth=2)
    
    if annotate_bs:
        ax.plot(categories, frac_bs_to_put, c='red', marker='s',
                markersize=13, alpha=pcfg.alpha)


if __name__ == '__main__':

    # lcurve_deg_field()
    # fac_type()
    # fac_type_us_kr()
    # trained_rank_v_placed_rank(average=True, normalize=True)
    mobility_kr_us()
    # placed_rank_density(range_trained=(0, 20))
    # placed_rank_density(range_trained=(80, 100))
    # hires_stat()
    # hires_z()
    # rank_move_grouped(net_type='domestic', inset=True)
    # radg_by_rank(add_reg=True)
    # us_trained_rank_v_dist_from_seoul()
    # rank_v_dist()
    # rank_v_dist_hmap()
    # rank_v_region()
    # self_hires_v_rank_grouped()