import matplotlib.pyplot as plt
import sqlite3 as sql
import scipy.stats
import googlemaps
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from networks.construct_net import construct_network
from analysis.doc_type import divide_by_groups
from config.config import net_db_path, fig_path, identifiers
import parse.queries_integration as qsi
import plot.config as pcfg


# TODO: trained v. placed curve for [US->KR (cross-hires) < KR->KR (auto-hires) < KR->US (cross-hires)] 

max_wapman_ranks = {"Biology": 201,
                    "ComputerScience": 216,
                    "Physics": 214}

max_domestic_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}


def convert_per_to_ranks(max_rank, percent_tuple):

    lower_bound = int(max_rank * percent_tuple[0] / 100)
    upper_bound = int(max_rank * percent_tuple[1] / 100)
        
    if lower_bound == 0:
        lower_bound = 1
    else:
        lower_bound += 1

    return lower_bound, upper_bound


def decades_to_interval(decade):

    match decade:
        case "Overall":
            return (0, 10000)
        case "70s":
            return (1970, 1980)
        case "80s":
            return (1980, 1990)
        case "90s":
            return (1990, 2000)
        case "00s":
            return (2000, 2010)
        case "10s":
            return (2010, 2020)
        case "20s":
            return (2020, 2030)
        case _:
            return None
        

def _classify_career_year(year):

    if year == 0 or year is None:
        return None

    match int(year / 10):
        case 197:
            return "70s"
        case 198:
            return "80s"
        case 199:
            return "90s"
        case 200:
            return "00s"
        case 201:
            return "10s"
        case 202:
            return "20s"


def trained_rank_v_placed_rank(dec="Overall", average=False, normalize=False):

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = r"$\pi_{placed}$"
    ylabel = r"$\pi_{trained}$" if not average else r"$\pi_{trained, avg}$"

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _trained_rank_v_placed_rank(ax, g, dec, average, normalize)

    # plt.show()
    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"trained_rank_v_placed_rank_{dec}_average({average})_normalize({normalize})"),
                bbox_inches='tight')


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

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

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

        eq_ut = f"{slope_ut:.2e}x + {intercept_ut:.2e}"

        print(f"=== {iden}, {dec} ===")
        print(f"Slope: {slope_ut}")
        print(f"Intercept: {intercept}")
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

        eq_kt = f"{slope_kt:.2e}x + {intercept_kt:.2e}"

        print(f"=== {iden}, {dec} ===")
        print(f"Slope: {slope_kt}")
        print(f"Intercept: {intercept}")
        print(f"R-squared: {r_value_kt**2}")
        print(f"P-value: {p_value_kt}")
        print(f"Standard error: {std_err}")
        
        x_vals = np.array(ax.get_xlim())
        y_vals = slope_kt * x_vals + intercept_kt
        ax.plot(x_vals, y_vals, color=pcfg.color_kt, linestyle='-', linewidth=2)

        label_ut = f'<US-trained>\n{eq_ut}'
        label_kt = f'<KR-trained>\n{eq_kt}'

        label_to_put = '\n'.join([label_ut, label_kt])

        ax.text(0.95, 0.05,
                label_to_put,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.small_tick_size, bbox=dict(facecolor='white', alpha=0.8))
        

def us_trained_rank_v_kr_rank(dec="Overall", average=False, normalize=False):

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = r"$\pi_{placed, KR}$"
    ylabel = r"$\pi_{trained, US}$" if not average else r"$\pi_{trained, US, avg}$"

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _us_trained_rank_v_kr_rank(ax, g, dec, average, normalize)

    # plt.show()
    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"us_trained_rank_v_kr_placed_rank_{dec}_average({average})_normalize({normalize})"),
                bbox_inches='tight')


def _us_trained_rank_v_kr_rank(ax, g, dec, average, normalize):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    trained_ranks = []
    placed_ranks = []

    max_trained_rank = max_wapman_ranks[iden.split('_')[1]]
    max_placed_rank = max_domestic_ranks[iden.split('_')[1]]

    interval = decades_to_interval(dec)

    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if src_nation != 'US' or dst_nation != 'KR':
            continue

        phd_granted_year = data['phd_end_year']

        if phd_granted_year is None:
            continue

        if not interval[0] <= phd_granted_year < interval[1]:
            continue

        t_rank = g.nodes[src_id][f'{abbrev}_rank_wapman']

        if t_rank is None:
            continue

        p_rank = g.nodes[dst_id][f'{abbrev}_rank_domestic']

        if p_rank is None:
            continue

        trained_ranks.append(t_rank)
        placed_ranks.append(p_rank)

    if normalize:
        trained_ranks = list(np.array(trained_ranks) / max_trained_rank)
        placed_ranks = list(np.array(placed_ranks) / max_placed_rank)

    if average:

        gathering = {}

        for t, p in zip(trained_ranks, placed_ranks):

            if p not in gathering:
    
                gathering[p] = []
                gathering[p].append(t)

            else:
                gathering[p].append(t)

        gathering_mean = {}

        for p, ts in gathering.items():
            gathering_mean[p] = np.mean(ts)

        trained_ranks = list(gathering_mean.values())
        placed_ranks = list(gathering_mean.keys())

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    xlim = max_placed_rank
    ylim = max_trained_rank

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

    ax.scatter(placed_ranks, trained_ranks, marker='o',
               alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
               edgecolor='black')

    if placed_ranks and trained_ranks:
        placed_ranks = np.array(placed_ranks)
        trained_ranks = np.array(trained_ranks)

        results = scipy.stats.linregress(placed_ranks, trained_ranks)
        slope = results.slope
        intercept = results.intercept
        r_value = results.rvalue
        p_value = results.pvalue
        std_err = results.stderr

        print(f"=== {iden}, {dec} ===")
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"R-squared: {r_value**2}")
        print(f"P-value: {p_value}")
        print(f"Standard error: {std_err}")
        
        x_vals = np.array(ax.get_xlim())
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, color=pcfg.colors[iden][2], linestyle='-', linewidth=2)

        ax.text(0.95, 0.05,
                f'Slope: {slope:.2f}\n$R^2$: {r_value**2:.2f}\nP-value: {p_value:.2e}',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.small_tick_size, bbox=dict(facecolor='white', alpha=0.8))


def us_trained_rank_v_kr_rank_by_year():

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    axs[1].set_xlabel(r"$\pi_{placed, KR}$", fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(r"$\pi_{trained, US}$", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _us_trained_rank_v_kr_rank_by_year(ax, g)

    # plt.show()
    plt.tight_layout(pad=1)
    plt.savefig(fig_path("us_trained_rank_v_kr_placed_rank_by_year"),
                bbox_inches='tight')
    

def _us_trained_rank_v_kr_rank_by_year(ax, g):

    iden = g.name

    conn = sql.connect(net_db_path(iden))
    cursor = conn.cursor()

    cursor.execute(qsi.GET_US2KR_EDGES)
    edges = cursor.fetchall()

    decades = ["70s", "80s", "90s", "00s", "10s", "20s"]

    trained_ranks_dict = {dec: [] for dec in decades}
    placed_ranks_dict = {dec: [] for dec in decades}

    cursor.execute(qsi.GET_MAX_DOMESTICRANK)
    max_trained_rank = max_wapman_ranks[iden.split('_')[1]]
    max_placed_rank = cursor.fetchone()[0]
    
    for edge in edges:

        src_id = edge[3]
        dst_id = edge[4]

        phd_granted_year = edge[7]
        decade = _classify_career_year(phd_granted_year)

        if decade is None:
            continue 

        cursor.execute(qsi.GET_WAPMANRANK_BY_NODEID, (src_id,))

        t_rank = cursor.fetchone()[0]

        if t_rank is None:
            continue

        cursor.execute(qsi.GET_DOMESTICRANK_BY_NODEID, (dst_id,))

        p_rank = cursor.fetchone()[0]

        trained_ranks_dict[decade].append(t_rank)
        placed_ranks_dict[decade].append(p_rank)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlim(0, max_placed_rank + 1)
    ax.set_ylim(0, max_trained_rank + 1)

    ax.set_xticks(range(0, max_placed_rank + 1, 25))
    ax.set_yticks(range(0, max_trained_rank + 1, 25))

    ax.invert_xaxis()
    ax.invert_yaxis()

    markers = ['o', 's', 'v']

    print(iden)

    for i, dec in enumerate(decades[1:4]):

        if len(placed_ranks_dict[dec]) == 0:
            # plt.plot([], [], label=dec, marker='s')
            continue

        pairs = zip(placed_ranks_dict[dec], trained_ranks_dict[dec])
        sorted_pairs = sorted(pairs)

        placed_sorted, trained_sorted = zip(*sorted_pairs)

        color = pcfg.colors[iden][2 * i]

        ax.scatter(placed_sorted, trained_sorted, label=dec, marker=markers[i],
                   alpha=0.5, c=color, s=150)
        
        print(dec, scipy.stats.spearmanr(placed_sorted, trained_sorted))

    legend = ax.legend(fontsize=pcfg.legend_size, edgecolor='black',
                       framealpha=0.8, fancybox=False, loc='lower right')
    # legend.get_frame().set_linewidth(1)
    legend.get_frame().set_linestyle('-')  # Solid line

    conn.commit()
    cursor.close()
    conn.close()


def us_trained_rank_v_year(cleaning='raw'):

    assert (cleaning in ['raw', 'average', 'median'])

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = 'Ph.D. graduation year'

    if cleaning == 'raw':
        ylabel = r"$\pi_{trained, US}$"
    elif cleaning == 'average':
        ylabel = r"$\pi_{trained, US, avg}$"
    elif cleaning == 'median':
        ylabel = r"$\pi_{trained, US, med}$"

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _us_trained_rank_v_year(ax, g, cleaning)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"us_trained_rank_v_year_{cleaning}"),
                bbox_inches='tight')


def _us_trained_rank_v_year(ax, g, cleaning):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    trained_ranks = []
    career_years = []

    max_trained_rank = max_wapman_ranks[iden.split('_')[1]]
    
    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if src_nation != 'US' or dst_nation != 'KR':
            continue

        phd_year = data['phd_end_year']

        if phd_year is None:
            continue

        t_rank = g.nodes[src_id][f"{abbrev}_rank_wapman"]

        if t_rank is None:
            continue

        trained_ranks.append(t_rank)
        career_years.append(phd_year)

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

    min_year_tick = 1980
    max_year_tick = 2030

    ax.set_xlim(min_year_tick, max_year_tick)
    ax.set_ylim(0, max_trained_rank + 1)

    ax.set_xticks(range(min_year_tick, max_year_tick + 1, 10))
    ax.set_yticks(range(0, max_trained_rank + 1, 25))

    ax.invert_yaxis()

    ax.scatter(career_years, trained_ranks, marker='o',
                alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
                edgecolor='black')


def us_trained_rank_v_year_by_group(average=False):

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = r"$Year_{Ph.D.}$"
    ylabel = r"$\pi_{trained, US}$" if not average else r"$\pi_{trained, US, avg}$"

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _us_trained_rank_v_year_by_group(ax, g, average)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"us_trained_rank_v_year_by_group_average({average})"),
                bbox_inches='tight')


def _us_trained_rank_v_year_by_group(ax, g, average):

    from collections import defaultdict

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    conn = sql.connect(net_db_path(iden))
    cursor = conn.cursor()

    trained_ranks = []
    career_years = []

    max_trained_rank = max_wapman_ranks[iden.split('_')[1]]
    
    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if src_nation != 'US' or dst_nation != 'KR':
            continue

        phd_year = data['phd_end_year']

        if phd_year is None:
            continue

        t_rank = g.nodes[src_id][f"{abbrev}_rank_wapman"]

        if t_rank is None:
            continue

        trained_ranks.append(t_rank)
        career_years.append(phd_year)

    if average:

        interval_gathering = defaultdict(list)

        for t, y in zip(trained_ranks, career_years):
            interval = (y // 5) * 5  # Get the 5-year interval start year
            interval_gathering[interval].append(t)

        interval_mean = {}
        interval_min = {}
        interval_max = {}

        for interval, ts in interval_gathering.items():
            interval_mean[interval] = np.mean(ts)
            interval_min[interval] = np.min(ts)
            interval_max[interval] = np.max(ts)

        trained_ranks = list(interval_mean.values())
        career_years = list(interval_mean.keys())
        min_ranks = list(interval_min.values())
        max_ranks = list(interval_max.values())

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='x', labelrotation=45)

    min_year_tick = 1980
    max_year_tick = 2030

    ax.set_xlim(min_year_tick, max_year_tick)
    ax.set_ylim(0, max_trained_rank + 1)

    ax.set_xticks(range(min_year_tick, max_year_tick + 1, 10))
    ax.set_yticks(range(0, max_trained_rank + 1, 25))

    ax.invert_yaxis()

    if average:
        ax.errorbar(career_years, trained_ranks, yerr=[np.subtract(trained_ranks, min_ranks), np.subtract(max_ranks, trained_ranks)], fmt='o',
                    alpha=pcfg.alpha, color=pcfg.colors[iden][1], ecolor='black', capsize=5)
    else:
        ax.scatter(career_years, trained_ranks, marker='o',
                   alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
                   edgecolor='black')

    conn.commit()
    cursor.close()
    conn.close()


def us_trained_rank_v_year_box(to=(0, 100)):

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    axs[1].set_xlabel('Ph.D. graduation decade', fontsize=pcfg.xlabel_size)
    # fig.supxlabel('Ph.D. graduation decade', fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(r"$\pi_{trained, US}$", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _us_trained_rank_v_year_box(ax, g)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path("us_trained_rank_v_year_box"),
                bbox_inches='tight')


def _us_trained_rank_v_year_box(ax, g):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    trained_ranks = [[], [], [], [], []]

    max_trained_rank = max_wapman_ranks[iden.split('_')[1]]
    
    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if src_nation != 'US' or dst_nation != 'KR':
            continue

        phd_year = data['phd_end_year']

        if phd_year is None:
            continue
        
        if 1980 <= phd_year < 1990:
            index = 0
        elif 1990 <= phd_year < 2000:
            index = 1
        elif 2000 <= phd_year < 2010:
            index = 2
        elif 2010 <= phd_year < 2020:
            index = 3
        elif 2020 <= phd_year < 2030:
            index = 4   
        else:
            continue

        t_rank = g.nodes[src_id][f"{abbrev}_rank_wapman"]

        if t_rank is None:
            continue

        trained_ranks[index].append(t_rank)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='x')

    # ax.set_xlim(min_year_tick, max_year_tick)
    ax.set_ylim(0, max_trained_rank + 1)

    # ax.set_xticks(range(min_year_tick, max_year_tick + 1, 10))
    ax.set_yticks(range(0, max_trained_rank + 1, 25))
    
    ax.invert_yaxis()
    
    labels = ['80s', '90s', '00s', '10s', '20s']
    box = ax.boxplot(trained_ranks, vert=True, patch_artist=True, widths=0.35, labels=labels)

    color = pcfg.colors[iden][1]

    for patch in box['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(pcfg.alpha)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)


def us_trained_rank_v_dist_from_seoul(cleaning='raw', to_exclude=[]):

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = 'Distance to Seoul (km)'

    if cleaning == 'raw':
        ylabel = r"$\pi_{trained, US}$"
    elif cleaning == 'average':
        ylabel = r"$\pi_{trained, US, avg}$"
    elif cleaning == 'median':
        ylabel = r"$\pi_{trained, US, med}$"

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _us_trained_rank_v_dist_from_seoul(ax, g, cleaning, to_exclude)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"us_trained_rank_v_dist_{cleaning}_exclude_{'_'.join(to_exclude)}"),
                bbox_inches='tight')
    

def _us_trained_rank_v_dist_from_seoul(ax, g, cleaning, to_exclude):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    trained_ranks = []
    career_years = []

    exclude_ist = True if 'ist' in to_exclude else False
    exclude_flagship = True if 'flagship' in to_exclude else False

    max_trained_rank = max_wapman_ranks[iden.split('_')[1]]
    
    for src_id, dst_id, data in g.edges(data=True):

        if exclude_ist:
            if dst_id in [2, 12, 15, 48, 229, 282]:
                continue

        if exclude_flagship:
            if dst_id in [4, 6, 9, 10, 11, 54, 74, 97, 128, 250, 355]:
                continue

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if src_nation != 'US' or dst_nation != 'KR':
            continue

        placed_dist = g.nodes[dst_id]['distance_to_seoul']

        if placed_dist is None:
            continue

        t_rank = g.nodes[src_id][f"{abbrev}_rank_wapman"]

        if t_rank is None:
            continue

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
    ax.set_ylim(0, max_trained_rank + 1)

    ax.set_xticks(range(0, 501, 100))

    ax.invert_yaxis()

    ax.scatter(career_years, trained_ranks, marker='o',
                alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
                edgecolor='black')
    
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)

    sns.kdeplot(x=career_years, ax=ax_histx, fill=True, color=pcfg.colors[iden][1], alpha=0.2)

    ax_histx.tick_params(axis='x', which='both', bottom=False, top=False, labelsize=0)
    ax_histx.tick_params(axis='y', which='both', left=False, right=False, labelsize=0)
    ax_histx.set_yticks([])
    ax_histx.set_ylabel('')


def placed_rank_density(range_trained=(0, 20), mode='utkp'):

    assert (mode in ['utkp', 'ktup', 'ktkp', 'kp', 'total'])

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    axs[1].set_xlabel("Rank decile", fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel("Density", fontsize=pcfg.ylabel_size)

    axs2 = []

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        if mode == 'total':

            modes = ['utkp', 'ktup', 'ktkp']
            styles = ['--', '-.', '-']
            colors = ['mediumblue', 'tomato', 'mediumseagreen']

            for i in range(len(modes)):

                m = modes[i]
                ls = styles[i]
                c = colors[i]

                ax2 =_placed_rank_density(ax, g, range_trained, m, c=c, ls=ls)

        elif mode == 'kp':

            modes = ['utkp', 'ktkp']
            styles = ['--', '-']
            colors = ['mediumblue', 'tomato']

            for i in range(len(modes)):

                m = modes[i]
                ls = styles[i]
                c = colors[i]

                ax2 =_placed_rank_density(ax, g, range_trained, m, c=c, ls=ls)

        else:
            ax2 =_placed_rank_density(ax, g, range_trained, mode)

        axs2.append(ax2)

    axs2[2].set_ylabel("Counts", fontsize=pcfg.ylabel_size)

    axs2[0].get_shared_y_axes().join(axs2[0], axs2[1], axs2[2])

    # axs2[0].sharey(axs2[1])
    # axs2[1].sharey(axs2[2])

    # axs2[1].set_yticks(axs2[0].get_yticks())
    # axs2[2].set_yticks(axs2[0].get_yticks())

    axs2[0].set_yticklabels([])
    axs2[1].set_yticklabels([])

    axs2[0].tick_params(axis='y', which='both', length=0)
    axs2[1].tick_params(axis='y', which='both', length=0)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"{mode}_placed_rank_density_{range_trained[0]}_{range_trained[1]}"),
                bbox_inches='tight')
    

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

    ax2 = ax.twinx()

    ax.set_xlim(1, max_placed_rank)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax2.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    
    ax.invert_xaxis()

    if c is None:
        c = pcfg.colors[iden][1]

    if ls is None:
        ls = '--'

    counts, bins, _ = ax2.hist(data, color=c, alpha=pcfg.alpha,
                            edgecolor='black', linewidth=3, bins=bins)
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    ax.set_xticks(bin_centers)  # Set positions for x-ticks (1 to 10)
    ax.set_xticklabels([str(11 - i) for i in range(1, 11)]) 
    ax.plot(x, y, label='Density', color=c, linestyle=ls, linewidth=4)

    # max_density = int(max(y) / 0.05) * 0.05

    ax.set_ylim(0, 0.125)
    ax.set_yticks(np.arange(0, 0.126, 0.125))

    if range_trained == (0, 20):
        ax2.set_ylim(0, 150)
        ax2.set_yticks(range(0, 151, 50))

    elif range_trained == (20, 40):
        ax2.set_ylim(0, 50)
        ax2.set_yticks(range(0, 51, 25))

    elif range_trained == (40, 60):
        ax2.set_ylim(0, 25)
        ax2.set_yticks(range(0, 26, 25))

    else:
        ax2.set_ylim(0, 10)
        ax2.set_yticks(range(0, 11, 5))

        # ax2.set_ylim(0, 50)
        # ax2.set_yticks(range(0, 51, 25))

    return ax2


if __name__ == "__main__":

    # us_trained_rank_v_kr_rank(normalize=True, average=True)
    # us_trained_rank_v_kr_rank(normalize=True, dec='70s')
    # us_trained_rank_v_kr_rank(normalize=True, dec='80s', average=True)
    # us_trained_rank_v_kr_rank(normalize=True, dec='90s', average=True)
    # us_trained_rank_v_kr_rank(normalize=True, dec='00s', average=True)
    # us_trained_rank_v_kr_rank(normalize=True, dec='10s', average=True)
    # us_trained_rank_v_kr_rank(normalize=True, dec='20s')

    # us_trained_rank_v_dist_from_seoul(cleaning='raw', exclude_ist=True)
    # us_trained_rank_v_dist_from_seoul(cleaning='raw', exclude_ist=False)

    # us_trained_rank_v_kr_rank()
    # us_trained_rank_v_kr_rank(dec='70s')
    # us_trained_rank_v_kr_rank(dec='80s')
    # us_trained_rank_v_kr_rank(dec='90s')
    # us_trained_rank_v_kr_rank(dec='00s')
    # us_trained_rank_v_kr_rank(dec='10s')
    # us_trained_rank_v_kr_rank(dec='20s')

    # us_trained_rank_v_dist_from_seoul()
    # us_trained_rank_v_dist_from_seoul(cleaning='average')

    # us_trained_rank_v_year_box()
    
    # us_trained_rank_v_year(cleaning='raw')
    # us_trained_rank_v_year(cleaning='average')
    # us_trained_rank_v_year(cleaning='median')

    # for mode in ['kp']:

    #     placed_rank_density(range_trained=(0, 20), mode=mode)
    #     placed_rank_density(range_trained=(20, 40), mode=mode)
    #     placed_rank_density(range_trained=(40, 60), mode=mode)
    #     placed_rank_density(range_trained=(60, 80), mode=mode)
    #     placed_rank_density(range_trained=(80, 100), mode=mode)

    # us_trained_rank_v_dist_from_seoul()
    # us_trained_rank_v_dist_from_seoul(to_exclude=['ist'])
    # us_trained_rank_v_dist_from_seoul(to_exclude=['ist', 'flagship'])

    for dec in ['Overall', '80s', '90s', '00s', '10s']:

        trained_rank_v_placed_rank(average=False, dec=dec)
        trained_rank_v_placed_rank(average=True, dec=dec)

        trained_rank_v_placed_rank(average=False, normalize=True, dec=dec)
        trained_rank_v_placed_rank(average=True, normalize=True, dec=dec)

    # placed_rank_density(range_trained=(0, 20), mode='total')
    # placed_rank_density(range_trained=(20, 40), mode=mode)
    # placed_rank_density(range_trained=(40, 60), mode=mode)
    # placed_rank_density(range_trained=(60, 80), mode=mode)
    # placed_rank_density(range_trained=(80, 100), mode=mode)

