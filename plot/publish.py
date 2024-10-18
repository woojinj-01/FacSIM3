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
from plot_deprecated.trained_rank import decades_to_interval, convert_per_to_ranks
from plot_deprecated.rank_move import _interhistogram
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

# doc = 'doc4_wo_alpha'
doc = 'doc4'


def rgba_from_hex(hex_color, opacity=pcfg.alpha):

    opacity = max(0, min(opacity, 1))
    
    alpha_value = int(opacity * 255)

    alpha_hex = f'{alpha_value:02X}'

    return f'{hex_color}{alpha_hex}'


def deprecated_lcurve_deg_field(iden=iden_default):

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

    _deprecated_lcurve_deg_field(ax, net_g, net_d)

    handles = [Patch(facecolor=rgba_from_hex(pcfg.COLOR_GLOBAL_NET), 
                     edgecolor='black', linewidth=3),
               Patch(facecolor=rgba_from_hex(pcfg.COLOR_DOMESTIC_NET), 
                     edgecolor='black', linewidth=3),
               Line2D([0], [0], color='black', linestyle='-', linewidth=4),
               Line2D([0], [0], color='black', linestyle='--', linewidth=4)]

    labels = ["Global", "Domestic", "Out-degrees (production)", "In-degrees (recruitment)"]

    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.8, 0.2), ncol=1, frameon=False)
    
    plt.savefig(fig_path(f"./{doc}/deprecated_lcurve_deg_{iden}"))
    

def _deprecated_lcurve_deg_field(ax, g, d):

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

    draw_lcurve(g, pcfg.COLOR_GLOBAL_NET)
    draw_lcurve(d, pcfg.COLOR_DOMESTIC_NET)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='x', which='major', pad=15)

    ticks = np.arange(0, 1.1, 0.25)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.plot([0, 1], [0, 1], c='black', alpha=pcfg.alpha)


def lcurve_deg_field(iden=iden_default, draw='all'):

    net_g = construct_network(net_type='global').get(iden)
    net_d = construct_network(net_type='domestic').get(iden)

    if any(net is None for net in [net_g, net_d]):
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize + 2, pcfg.fig_ysize + 2), dpi=pcfg.dpi)

    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    xlabel = 'Cumulative ratio of institutions (%)'
    ylabel = 'Cumulative ratio of degrees (%)'

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.ylabel_size)

    _lcurve_deg_field(ax, net_g, net_d, draw=draw)

    handles = [Patch(facecolor=pcfg.COLOR_GLOBAL_NET,  edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.COLOR_DOMESTIC_NET,  edgecolor='black', linewidth=3),
               Line2D([0], [0], color='black', linestyle='-', linewidth=4),
               Line2D([0], [0], color='black', linestyle='--', linewidth=4)]

    labels = ["Global", "Domestic", "Out-degrees (production)", "In-degrees (recruitment)"]

    if draw == 'g':
        handles.pop(1)
        labels.pop(1)

    elif draw == 'd':
        handles.pop(0)
        labels.pop(0)

    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.8, 0.2), ncol=1, frameon=False)
    
    plt.savefig(fig_path(f"./{doc}/lcurve_deg_{iden}_{draw}"))


def _lcurve_deg_field(ax, g, d, draw='g'): # or draw = 'd' , 'all' 

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

        ax.tick_params(axis='both', which='major', labelsize=13)

        ax.plot(xco, yco_out, c=c, linewidth=6)
        ax.plot(xco, yco_in,  c=c, linewidth=6, linestyle='--')

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

    if draw == 'g': 
        draw_lcurve(g, pcfg.COLOR_GLOBAL_NET)

    elif draw == 'd': 
        draw_lcurve(d, pcfg.COLOR_DOMESTIC_NET)

    elif draw == 'all': 
        draw_lcurve(g, pcfg.COLOR_GLOBAL_NET)
        draw_lcurve(d, pcfg.COLOR_DOMESTIC_NET)

    ax.set_xlim(-0.001, 1.001)
    ax.set_ylim(-0.001, 1.001)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='x', which='major', pad=15)

    ax.set_xticks(np.arange(0, 1.001, 0.2))
    ax.set_yticks(np.arange(0, 1.001, 0.2))
    ax.set_xticklabels(['0', '20', '40', '60', '80', '100'], 
                       fontsize=pcfg.tick_size)
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], 
                       fontsize=pcfg.tick_size)
    
    ax.plot([-0.001, 1.001], [-0.001, 1.001], c='black', alpha=pcfg.alpha)


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

        print(f"{iden}: KR {kr}, US {us}, Others {others}")

        bars = ax.bar(field, us, color=rgba_from_hex(pcfg.COLOR_US_TRAINED), 
                      edgecolor='black', linewidth=2)
        ax.bar(field, others, color=rgba_from_hex(pcfg.COLOR_ELSE_TRAINED), 
               bottom=us, edgecolor='black', linewidth=2)
        ax.bar(field, kr, color=rgba_from_hex(pcfg.COLOR_KR_TRAINED), 
               bottom=us+others, edgecolor='black', linewidth=2)

        bar = bars[0]
        width = bar.get_width()
        height = bar.get_height()

        ax.text(bar.get_x() + width / 2, height / 2,
                f'{height * 100: .1f}%', ha='center', va='center',
                color='#D5EDEC', fontsize=40, weight='bold')

    handles = [Patch(facecolor=pcfg.COLOR_US_TRAINED, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.COLOR_ELSE_TRAINED, alpha=pcfg.alpha, edgecolor='black', linewidth=3),
               Patch(facecolor=pcfg.COLOR_KR_TRAINED, alpha=pcfg.alpha, edgecolor='black', linewidth=3),]

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

    yticks = ax.get_yticks()
    ax.set_yticklabels(yticks[::-1])

    ax.set_xlabel("Fraction of faculty", fontsize=pcfg.xlabel_size)
    ax.set_ylabel("Rank group", fontsize=pcfg.xlabel_size)

    _fac_type_us_kr(ax, net)

    handles = [Patch(facecolor=rgba_from_hex(pcfg.COLOR_US_TRAINED), 
                     edgecolor='black', linewidth=3),
               Patch(facecolor=rgba_from_hex(pcfg.COLOR_ELSE_TRAINED), 
                     edgecolor='black', linewidth=3),
               Patch(facecolor=rgba_from_hex(pcfg.COLOR_KR_TRAINED), 
                     edgecolor='black', linewidth=3),]

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

    bars = ax.barh(x_co, y_co_us, color=rgba_from_hex(pcfg.COLOR_US_TRAINED),
                   edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_others, color=rgba_from_hex(pcfg.COLOR_ELSE_TRAINED), 
            left=y_co_us, edgecolor='black', linewidth=2)
    ax.barh(x_co, y_co_kr, color=rgba_from_hex(pcfg.COLOR_KR_TRAINED),
            left=[us + others for us, others in zip(y_co_us, y_co_others)],
            edgecolor='black', linewidth=2)
    
    for bar in bars[-3:]:
        width = bar.get_width()
        ax.text(width / 2, bar.get_y() + bar.get_height() / 2 - 0.05,
                f'{width * 100: .1f}%', ha='center', va='center',
                color='#D5EDEC', fontsize=35, weight='bold')
        

def trained_rank_v_placed_rank(dec="Overall", average=False, normalize=False, iden=iden_default):

    net = construct_network().get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='x', which='major', pad=25)

    xlabel = r"$\pi_{placed}$"
    ylabel = r"$\pi_{trained}$" if not average else r"$\pi_{trained, avg}$"

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    eq_ut, eq_kt = _trained_rank_v_placed_rank(ax, net, dec, average, normalize)
    
    # handles = [Line2D([0], [0], marker='o', color='black', alpha=pcfg.alpha,
    #                   markerfacecolor=pcfg.COLOR_US_TRAINED, markersize=20, linestyle='None'),
    #            Line2D([0], [0], marker='o', color='black', alpha=pcfg.alpha,
    #                   markerfacecolor=pcfg.COLOR_KR_TRAINED, markersize=20, linestyle='None')]
    
    # labels = ['US-trained faculty', 'KR-trained faculty']

    # fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.7, 0.285),
    #            ncol=1, frameon=False, fontsize=pcfg.legend_size,
    #            handletextpad=0.2, columnspacing=0.3)

    # handles_eq = [Line2D([0], [0], color=pcfg.COLOR_US_TRAINED, alpha=pcfg.alpha, linestyle='-', linewidth=5),
    #               Line2D([0], [0], color=pcfg.COLOR_KR_TRAINED, alpha=pcfg.alpha, linestyle='-', linewidth=5)]
    
    # labels_eq = [f"{eq_ut}", f"{eq_kt}"]

    # fig.legend(handles_eq, labels_eq, loc='lower right', bbox_to_anchor=(0.81, 0.19),
    #            ncol=1, frameon=False, fontsize=pcfg.legend_size,
    #            handletextpad=0.4, columnspacing=0.3)
    
    handles = [Line2D([0], [0], marker=pcfg.MARKER_US_TRAINED, color='black',
                      markerfacecolor=rgba_from_hex(pcfg.COLOR_US_TRAINED),
                      markeredgewidth=2, markeredgecolor='black',
                      markersize=20, linestyle='None'),
               Line2D([0], [0], marker=pcfg.MARKER_KR_TRAINED, color='black',
                      markerfacecolor=rgba_from_hex(pcfg.COLOR_KR_TRAINED), 
                      markeredgewidth=2, markeredgecolor='black',
                      markersize=20, linestyle='None'),
               Line2D([0], [0], color=pcfg.COLOR_US_TRAINED, alpha=pcfg.alpha, 
                      linestyle='-', linewidth=5),
               Line2D([0], [0], color=pcfg.COLOR_KR_TRAINED, alpha=pcfg.alpha, 
                      linestyle='-', linewidth=5)]
    
    labels = ['US-trained faculty', 
              'KR-trained faculty',
              f"{eq_ut}", 
              f"{eq_kt}"]

    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.81, 0.19),
               ncol=1, frameon=False, fontsize=pcfg.legend_size,
               handletextpad=0.5, columnspacing=0.2)

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
        gathering_std = {}

        for p, ts in gathering.items():
            gathering_mean[p] = np.mean(ts)
            gathering_std[p] = np.std(ts)

            print(ts)
            print(gathering_mean[p], gathering_std[p])

        trained_ranks_ut = list(gathering_mean.values())
        placed_ranks_ut = list(gathering_mean.keys())

        trained_ranks_ut_std = list(gathering_std.values())
        placed_ranks_ut_std = list(gathering_std.keys())

        sorted_pairs = sorted(zip(placed_ranks_ut_std, trained_ranks_ut_std))

        placed_ranks_ut_std_sorted, trained_ranks_ut_std_sorted = zip(*sorted_pairs)

        placed_ranks_ut_std_sorted = list(placed_ranks_ut_std_sorted)
        trained_ranks_ut_std_sorted = list(trained_ranks_ut_std_sorted)

        gathering = {}

        for t, p in zip(trained_ranks_kt, placed_ranks_kt):

            if p not in gathering:
    
                gathering[p] = []
                gathering[p].append(t)

            else:
                gathering[p].append(t)

        gathering_mean = {}
        gathering_std = {}

        for p, ts in gathering.items():
            gathering_mean[p] = np.mean(ts)
            gathering_std[p] = np.std(ts)

        trained_ranks_kt = list(gathering_mean.values())
        placed_ranks_kt = list(gathering_mean.keys())

        trained_ranks_kt_std = list(gathering_std.values())
        placed_ranks_kt_std = list(gathering_std.keys())

        sorted_pairs = sorted(zip(placed_ranks_kt_std, trained_ranks_kt_std))

        placed_ranks_kt_std_sorted, trained_ranks_kt_std_sorted = zip(*sorted_pairs)

        placed_ranks_kt_std_sorted = list(placed_ranks_kt_std_sorted)
        trained_ranks_kt_std_sorted = list(trained_ranks_kt_std_sorted)

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
        ax.plot(x_vals, y_vals, color=pcfg.COLOR_US_TRAINED, 
                linestyle='-', linewidth=3)

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
        ax.plot(x_vals, y_vals, color=pcfg.COLOR_KR_TRAINED, 
                linestyle='-', linewidth=3)

        # label_ut = f'<US-trained faculty>\n{eq_ut}'
        # label_kt = f'<KR-trained faculty>\n{eq_kt}'

        # label_to_put = '\n'.join([label_ut, label_kt])

        # ax.text(0.95, 0.05,
        #         label_to_put,
        #         verticalalignment='bottom', horizontalalignment='right',
        #         transform=ax.transAxes,
        #         fontsize=pcfg.stat_size, bbox=dict(facecolor='white', alpha=0.8))

        if average:

            alpha_to_use = pcfg.alpha

            ax.errorbar(placed_ranks_ut, trained_ranks_ut,
                        yerr=trained_ranks_ut_std_sorted,
                        fmt=pcfg.MARKER_US_TRAINED, 
                        markersize=15, markeredgecolor='black', 
                        markeredgewidth=2,
                        elinewidth=3, capsize=5, ecolor='grey',
                        c=rgba_from_hex(pcfg.COLOR_US_TRAINED, alpha_to_use))
    
            ax.errorbar(placed_ranks_kt, trained_ranks_kt,
                        yerr=trained_ranks_kt_std_sorted,
                        fmt=pcfg.MARKER_KR_TRAINED, 
                        markersize=15, markeredgecolor='black',
                        markeredgewidth=2,
                        elinewidth=3, capsize=5, ecolor='grey',
                        c=rgba_from_hex(pcfg.COLOR_KR_TRAINED, alpha_to_use))
            
            return eq_ut, eq_kt
            
    else:
        ax.scatter(placed_ranks_ut, trained_ranks_ut, 
                   marker=pcfg.MARKER_US_TRAINED,
                   alpha=pcfg.alpha, s=150, c=pcfg.COLOR_US_TRAINED,
                   edgecolor='black')
    
        ax.scatter(placed_ranks_kt, trained_ranks_kt, 
                   marker=pcfg.MARKER_KR_TRAINED,
                   alpha=pcfg.alpha, s=150, c=pcfg.COLOR_KR_TRAINED,
                   edgecolor='black')
        
        return '', ''
        

def mobility_kr_us(normalize=True, iden=iden_default):

    assert (isinstance(normalize, bool))

    net = construct_network().get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    fig = plt.figure(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    ylabel = 'Fraction' if normalize else 'Number'

    # axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    # fig.supxlabel(xlabel, fontsize=pcfg.xlabel_size)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    _mobility_kr_us_both(ax, net, normalize)

    handles = [Patch(facecolor=pcfg.COLOR_UP_HIRE, edgecolor='black', alpha=pcfg.alpha, linewidth=2),
               Patch(facecolor=pcfg.COLOR_DO_HIRE, edgecolor='black', alpha=pcfg.alpha, linewidth=2)]
    
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

    ax.tick_params(axis='x', which='major',
                   labelsize=pcfg.small_tick_size, labelrotation=30)

    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)

    bar_width = 0.4

    ax.bar(x - bar_width / 2, data_up, width=bar_width, color=pcfg.COLOR_UP_HIRE,
           alpha=pcfg.alpha, edgecolor='black', linewidth=2)
    ax.bar(x + bar_width / 2, data_do, width=bar_width, color=pcfg.COLOR_DO_HIRE,
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

    ax.set_xlabel("Rank group", fontsize=pcfg.xlabel_size)
    ax.set_ylabel("Counts", fontsize=pcfg.ylabel_size)

    modes = ['utkp', 'ktkp', ]
    styles = ['-', '-']
    colors = [pcfg.COLOR_US_TRAINED, pcfg.COLOR_KR_TRAINED]

    for i in range(len(modes)):

        m = modes[i]
        ls = styles[i]
        c = colors[i]

        _placed_rank_density(ax, net, range_trained, m, c=c, ls=ls)

    # ax2.set_ylabel("Counts", fontsize=pcfg.ylabel_size)

    handles = [Patch(facecolor=rgba_from_hex(pcfg.COLOR_US_TRAINED, opacity=0.5),
                     edgecolor='black', linewidth=3),
               Line2D([0], [0], color=pcfg.COLOR_US_TRAINED, 
                      linestyle='-', linewidth=4),
               Patch(facecolor=rgba_from_hex(pcfg.COLOR_KR_TRAINED, opacity=0.5),
                     edgecolor='black', linewidth=3),
               Line2D([0], [0], color=pcfg.COLOR_KR_TRAINED, 
                      linestyle='-', linewidth=4)]

    labels = ["US-trained faculty (count)", "US-trained faculty (density)",
              "KR-trained faculty (count)", "KR-trained faculty (density)"]

    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)

    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.80, 0.80), ncol=1, frameon=False)

    plt.savefig(fig_path(f"./{doc}/placed_rank_density_{range_trained[0]}_{range_trained[1]}_{iden}"))
    

def _placed_rank_density(ax, g, range_trained, mode, ls=None, c=None):
    
    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    placed_ranks = []

    alpha_to_use = 0.5

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

    counts, bins, _ = ax.hist(data, color=rgba_from_hex(c, alpha_to_use),
                              edgecolor='black', linewidth=3, bins=bins)
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    ax.set_xticks(bin_centers)  # Set positions for x-ticks (1 to 10)
    ax.set_xticklabels([str(i) for i in range(1, 11)]) 
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

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_ylabel('Z-scores', fontsize=pcfg.ylabel_size)

    ax.set_ylim(-10, 20)
    ax.set_yticks(range(-10, 21, 5))

    ax.tick_params(axis='x', which='major', labelsize=pcfg.small_tick_size)
    ax.tick_params(axis='x', which='major', pad=15)

    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='y', which='major', pad=5)

    bar_width = 0.15
    x_pos = 1  # Center position for the bars

    # Adjust positions of the bars
    x_pos_1 = x_pos - 1.5 * bar_width  # Down hires
    x_pos_2 = x_pos               # Self hires
    x_pos_3 = x_pos + 1.5 * bar_width   # Up hires

    # Update the x-ticks to reflect the new positions
    ax.set_xticks([x_pos_1, x_pos_2, x_pos_3])
    ax.set_xticklabels(['Down hires', 'Self hires', 'Up hires'])

    do_z = (stat_net['Down'] / stat_net['Total'] - stat['Down'][0]) / stat['Down'][1]
    se_z = (stat_net['Self'] / stat_net['Total'] - stat['Self'][0]) / stat['Self'][1]
    up_z = (stat_net['Up'] / stat_net['Total'] - stat['Up'][0]) / stat['Up'][1]

    ax.bar(x_pos_1, do_z, width=bar_width, 
           color=rgba_from_hex(pcfg.COLOR_DO_HIRE),
           edgecolor='black', linewidth=3)
    ax.bar(x_pos_2, se_z, width=bar_width, 
           color=rgba_from_hex(pcfg.COLOR_SE_HIRE),
           edgecolor='black', linewidth=3)
    ax.bar(x_pos_3, up_z, width=bar_width, 
           color=rgba_from_hex(pcfg.COLOR_UP_HIRE),
           edgecolor='black', linewidth=3)

    ax.axhline(0, color='black', linewidth=1)
    ax.tick_params(axis='x')

    handles = [Patch(facecolor=pcfg.COLOR_DO_HIRE, alpha=pcfg.alpha,
                     edgecolor='black', linewidth=4),
               Patch(facecolor=pcfg.COLOR_SE_HIRE, alpha=pcfg.alpha,
                     edgecolor='black', linewidth=4),
               Patch(facecolor=pcfg.COLOR_UP_HIRE, alpha=pcfg.alpha,
                     edgecolor='black', linewidth=4)]

    labels = ['Down hires', 'Self hires', 'Up hires']

    # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.81, 0.83),
    #            ncol=1, frameon=False, handlelength=0.9, handleheight=0.9)
    
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

    ax.tick_params(axis='x', which='major', pad=15)

    groups = [(1, 5), (3, 5), (5, 5)]
    styles = ['-', ':', '--']
    colors = [pcfg.COLOR_DOMESTIC_NET_WARMER, 
              pcfg.COLOR_DOMESTIC_NET, 
              pcfg.COLOR_DOMESTIC_NET_COOLER]

    if inset:
        ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
        ax_inset.set_xlim(-1, 1)

    for i in range(3):

        group = groups[i]
        s = styles[i]
        c = colors[i]

        _rank_move(net, None, ax, net_type=net_type, group=group, export_style=s, export_color=c)

        if inset:
            _rank_move(net, None, ax_inset, net_type=net_type, group=group, 
                       export_style=s, normalize_sep=False, for_inset=True,
                       export_color=c)

    x_label = 'Relative Movement of Faculty\n(Normalized)'
    y_label = 'Density'

    ax.set_ylabel(y_label, fontsize=pcfg.ylabel_size)
    ax.set_xlabel(x_label, fontsize=pcfg.xlabel_size)
    
    handles = [Line2D([0], [0], color=c, linestyle=s, linewidth=4, alpha=pcfg.alpha) for s, c in zip(styles, colors)]

    labels = ["Top 20%", "40-60%", "Bottom 20%"]

    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.81, 0.2), ncol=1, frameon=False)

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

    c_default = pcfg.COLOR_GLOBAL_NET if net_type == 'global' else pcfg.COLOR_DOMESTIC_NET

    for move, style in zip(moves, linestyles):

        hist, _ = np.histogram(move, bins=bins)
        normalized_hist = hist / np.sum(hist)

        style_to_put = style if export_style is None else export_style
        color_to_put = c_default if export_color is None else export_color

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

    eq_first = _radg_by_rank(ax, net, dist_type, ver, add_reg)

    if add_reg:

        handles = [Line2D([0], [0], color='black', linewidth=3)]
        
        labels = [f'{eq_first}']

        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.81, 0.81),
                   ncol=1, frameon=False, fontsize=pcfg.legend_size)

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
               s=150, c=rgba_from_hex(pcfg.COLOR_DOMESTIC_NET),
               edgecolor='black', linewidths=2)
    
    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    max_rank = max_domestic_ranks[iden.split('_')[1]]
    ax.set_xlim(0, max_rank)
    ax.set_xticks(range(0, max_rank + 1, 20))

    if dist_type == 'geo':
        ax.set_ylim(0, 500)
        ax.set_yticks(range(0, 501, 100))

    else:
        pass
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))

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
                color='black', linewidth=3)

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

        return eq_first

    return ''


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

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size, labelpad=20)

    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 2.5, pad=0.1, sharex=ax)

    _us_trained_rank_v_dist_from_seoul(ax, ax_histx, net, cleaning, normalize, to_exclude)
    _us_trained_rank_v_dist_from_seoul(ax, ax_histx, net, cleaning, normalize, to_exclude, fac_type='ktkp')
    
    handles = [Line2D([0], [0], marker=pcfg.MARKER_US_TRAINED, color='black',
                      markerfacecolor=rgba_from_hex(pcfg.COLOR_US_TRAINED, 0.5), markersize=20, 
                      markeredgecolor='black', markeredgewidth=2,
                      linestyle='None'),
               Line2D([0], [0], marker=pcfg.MARKER_KR_TRAINED, color='black',
                      markerfacecolor=rgba_from_hex(pcfg.COLOR_KR_TRAINED, 0.5), markersize=20, 
                      markeredgecolor='black', markeredgewidth=2,
                      linestyle='None')]

    labels = ["US-trained faculty", "KR-trained faculty"]

    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.80, 0.80),
               ncol=1, frameon=False, handlelength=0.9, handleheight=0.9)

    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.48, 0.87), ncol=2, frameon=False)

    plt.savefig(fig_path(f"./{doc}/us_trained_rank_v_dist_{cleaning}_exclude_{'_'.join(to_exclude)}_{iden}"))
    

def _us_trained_rank_v_dist_from_seoul(ax, ax_histx, g, cleaning, normalize, to_exclude, fac_type='utkp'):

    assert (fac_type in ['utkp', 'ktkp'])

    mtype = pcfg.MARKER_US_TRAINED if fac_type == 'utkp' else pcfg.MARKER_KR_TRAINED

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    trained_ranks = []
    career_years = []

    alpha_to_use = 0.5

    exclude_ist = True if 'ist' in to_exclude else False
    exclude_flagship = True if 'flagship' in to_exclude else False

    max_trained_rank = max_wapman_ranks[iden.split('_')[1]] if fac_type == 'utkp' else max_domestic_ranks[iden.split('_')[1]]

    t_rank_key = f"{abbrev}_rank_wapman" if fac_type == 'utkp' else f"{abbrev}_rank_domestic"
    target_src_nation = 'US' if fac_type == 'utkp' else 'KR'
    target_dst_nation = 'KR'
    color = pcfg.COLOR_US_TRAINED if fac_type == 'utkp' else pcfg.COLOR_KR_TRAINED
    
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

    ax.scatter(career_years, trained_ranks, marker=mtype, 
               s=150, c=rgba_from_hex(color, alpha_to_use), 
               edgecolor='black', linewidths=2)

    sns.kdeplot(x=career_years, ax=ax_histx, fill=True, color=color, alpha=0.5, linewidth=3)

    ax_histx.yaxis.tick_right()
    ax_histx.tick_params(axis='x', which='both', 
                         bottom=False, top=False, labelsize=0)
    ax_histx.tick_params(axis='y', which='both', 
                         left=False, right=False, labelsize=30)
    # ax_histx.set_yticks(np.arange(0, 0.016, 0.005))
    ax_histx.set_yticks([])
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


def rank_v_dist_hmap_1(iden=iden_default):

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

    _rank_v_dist_hmap_1(ax, net)

    plt.savefig(fig_path(f"./{doc}/rank_v_dist_hmap_{iden}"))


def _rank_v_dist_hmap_1(ax, g):

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


def rank_v_dist_hmap(iden=iden_default):

    net = construct_network(net_type='domestic').get(iden)

    if net is None:
        return

    plt.rc('font', **pcfg.fonts)
    
    fig, ax = plt.subplots(figsize=(pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    ax_position = [0.2, 0.2, 0.6, 0.6]  # [left, bottom, width, height]
    ax.set_position(ax_position)

    cbar_ax_position = [0.85, 0.2, 0.03, 0.6]
    cbar_ax = fig.add_axes(cbar_ax_position)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ylabel = r"$\pi_{KR}$"
    xlabel = 'Distance from Seoul (km)'

    ax.set_xlabel(xlabel, fontsize=pcfg.xlabel_size, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    _rank_v_dist_hmap(ax, net, cbar_ax)

    plt.savefig(fig_path(f"./{doc}/rank_v_dist_hmap_{iden}"))


def _rank_v_dist_hmap(ax, g, cbar_ax):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)
    max_rank = max_domestic_ranks[iden.split('_')[1]]

    rgb_to_use = (0.7568627451, 0.2509803922, 0.2392156863)

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

    heatmap, xedges, yedges = np.histogram2d(dists, ranks, bins=[10, 10], range=[[0, 500], [0, max_rank + 1]])

    colors = [(1, 1, 1), rgb_to_use]  # Red gradient
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    pcm = ax.pcolormesh(xedges, yedges, heatmap.T, cmap=custom_cmap, shading='auto')

    ax.set_xlim(0, 500)
    ax.set_ylim(0, max_rank + 1)

    ax.set_xticks(range(0, 501, 100))

    ax.invert_yaxis()

    cbar = plt.colorbar(pcm, cax=cbar_ax)

    cbar.ax.tick_params(labelsize=pcfg.small_tick_size)  # Set tick label size
    cbar.set_label('Number of Institutions', fontsize=pcfg.ylabel_size - 5, 
                   labelpad=20)


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

    ax.set_xlabel("Rank group", fontsize=pcfg.xlabel_size)
    ax.set_ylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size, labelpad=20)

    _self_hires_v_rank_grouped(ax, net, annotate_bs)

    if annotate_bs:

        from matplotlib.lines import Line2D

        legend_size = 30

        handles = [Line2D([0], [0], color='black', marker='s', linewidth=3,
                          markersize=15, markeredgewidth=4, markerfacecolor='#D3D3D3',
                          markeredgecolor='black')]
        labels = ["Fraction of self-hiring\nown bachelor's graduates"]

        fig.legend(handles, labels, loc='upper right',
                   bbox_to_anchor=(0.8, 0.8), ncol=1, frameon=False,
                   fontsize=legend_size)

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

        print(f"[{title}] BS: {frac_bs: .3f} PhD: {frac_phd: .3f} BS+PhD: {frac: .3f}")

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    categories = [f"{i}" for i in range(len(ranges), 0, -1)]

    ax.bar(categories, frac_to_put, alpha=pcfg.alpha,
           color=pcfg.COLOR_DOMESTIC_NET, edgecolor='black', linewidth=3)
    
    if annotate_bs:
        ax.plot(categories, frac_bs_to_put, c='black', linewidth=3,
                marker='s', markersize=15, markeredgewidth=4,
                markerfacecolor='#D3D3D3', markeredgecolor='black')


if __name__ == '__main__':

    lcurve_deg_field(draw='g')
    lcurve_deg_field(draw='d')
    # fac_type()
    # fac_type_us_kr()
    # trained_rank_v_placed_rank(average=True, normalize=True)
    # mobility_kr_us()
    # placed_rank_density(range_trained=(0, 20))
    # placed_rank_density(range_trained=(80, 100))
    # hires_z()
    # rank_move_grouped(net_type='domestic', inset=True)
    # radg_by_rank(add_reg=True)
    # us_trained_rank_v_dist_from_seoul()
    # rank_v_dist()
    # rank_v_dist_hmap()
    # rank_v_region()
    # self_hires_v_rank_grouped()