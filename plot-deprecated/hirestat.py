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


def self_hires_v_rank():
 
    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    axs[1].set_xlabel(r"$\pi_{placed, KR}$", fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(r"$Fraction_{self-hire}$", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _self_hires_v_rank(ax, g)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"self_hires_v_rank"), bbox_inches='tight')

    
def _self_hires_v_rank(ax, g):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    ranks = []
    ratio = []

    stats = {}

    fractions = {}

    for src_id, dst_id, data in g.edges(data=True):

        if dst_id not in stats:
            stats[dst_id] = [0, 0]

        if src_id == dst_id:
            stats[dst_id][0] += 1
        else:

            bs_id = data['bs_inst_id']
            
            if bs_id is not None:
                if bs_id == dst_id:
                    stats[dst_id][0] += 1

        stats[dst_id][1] += 1

    for key, value in stats.items():

        if value[1] == 0:
            continue
        else:
            to_put = value[0] / value[1]

        drank = g.nodes[key][f"{abbrev}_rank_domestic"]

        if drank is not None:
            fractions[drank] = to_put

    sorted_fracs = dict(sorted(fractions.items()))

    # Step 2: Extract the keys and values into separate lists
    ranks = list(sorted_fracs.keys())
    ratio = list(sorted_fracs.values())

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.scatter(ranks, ratio, marker='o',
               alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
               edgecolor='black')
    
    ax.set_ylim(0, 1)


def self_hires_v_rank_mean(include_zero=False):
 
    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    # axs[1].set_xlabel("Rank group", fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    fig.supxlabel("Rank group", fontsize=pcfg.xlabel_size)
    # fig.supylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _self_hires_v_rank_mean(ax, g, include_zero)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"self_hires_v_rank_mean_{include_zero}"), bbox_inches='tight')


def _self_hires_v_rank_mean(ax, g, include_zero):

    iden = g.name
    field = iden.split('_')[1]
    abbrev = qsi.get_abbrev(iden)

    max_rank = max_domestic_ranks[field]
    num_bins = 10

    from analysis.doc_type import divide_by_groups

    ranges = divide_by_groups(max_rank, num_bins)

    stats = {}

    fractions = {i: [] for i in range(len(ranges))}

    for src_id, dst_id, data in g.edges(data=True):

        if dst_id not in stats:
            stats[dst_id] = [0, 0]

        if src_id == dst_id:
            stats[dst_id][0] += 1
        else:

            bs_id = data['bs_inst_id']
            
            if bs_id is not None:
                if bs_id == dst_id:
                    stats[dst_id][0] += 1

        stats[dst_id][1] += 1

    for key, value in stats.items():

        if value[1] == 0:
            continue
        else:
            to_put = value[0] / value[1]

        if include_zero is False and to_put == 0:
            continue

        drank = g.nodes[key][f"{abbrev}_rank_domestic"]

        if drank is None:
            continue

        for i, value in enumerate(ranges):
            if value[0] <= drank < value[1]:
                fractions[i].append(to_put)

    means = []
    stds = []

    for i, value in sorted(fractions.items()):

        mean = np.mean(value)
        std = np.std(value)

        means.append(mean)
        stds.append(std)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    categories = [f"{i + 1}" for i in range(len(ranges))]

    ax.bar(categories, means, yerr=stds, alpha=pcfg.alpha,
           color=pcfg.colors[iden][1], capsize=10,
           edgecolor='black')
    

def self_hires_v_rank_grouped(net_type='global', annotate_bs=False):
 
    nets = construct_network(net_type=net_type)

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    # axs[1].set_xlabel("Rank group", fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    fig.supxlabel("Rank decile", fontsize=pcfg.xlabel_size)
    # fig.supylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _self_hires_v_rank_grouped(ax, g, annotate_bs)

    if annotate_bs:

        from matplotlib.lines import Line2D

        handles = [Line2D([0], [0], color='red', marker='s', markersize=13, alpha=pcfg.alpha)]
        labels = ["Fraction of self-hiring own bachelor's graduates"]

        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False,
                   fontsize=pcfg.legend_size)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"self_hires_v_rank_grouped_{net_type}_annotate({annotate_bs})"), bbox_inches='tight')


def _self_hires_v_rank_grouped(ax, g, annotate_bs):

    iden = g.name
    field = iden.split('_')[1]
    abbrev = qsi.get_abbrev(iden)

    max_rank = max_domestic_ranks[field]
    num_bins = 10

    from analysis.doc_type import divide_by_groups

    ranges = divide_by_groups(max_rank, num_bins)

    fractions = {i: [0, 0, 0] for i in range(len(ranges))}

    for src_id, dst_id, data in g.edges(data=True):

        drank = g.nodes[dst_id][f"{abbrev}_rank_domestic"]

        if drank is None:
            continue

        is_self_hire = 0
        is_bs_self_hire = 0

        bs_id = data['bs_inst_id']
        
        if bs_id is not None and bs_id == dst_id:

            is_self_hire = 1
            is_bs_self_hire = 1

        elif src_id == dst_id:
            is_self_hire = 1

        for i, value in enumerate(ranges):
            if value[0] <= drank < value[1]:
                fractions[i][0] += is_self_hire
                fractions[i][1] += is_bs_self_hire
                fractions[i][2] += 1

    frac_to_put = []
    frac_bs_to_put = []

    for i, value in sorted(fractions.items()):

        frac = value[0] / value[2]
        frac_bs = value[1] / value[2]

        frac_to_put.append(frac)
        frac_bs_to_put.append(frac_bs)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    categories = [f"{i + 1}" for i in range(len(ranges))]

    ax.bar(categories, frac_to_put, alpha=pcfg.alpha,
           color=pcfg.colors[iden][1], edgecolor='black', linewidth=2)
    
    if annotate_bs:
        ax.plot(categories, frac_bs_to_put, c='red', marker='s',
                markersize=13, alpha=pcfg.alpha)
    

def self_hires_v_rank_box(include_zero=False):
 
    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    # axs[1].set_xlabel("Rank group", fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    fig.supxlabel("Rank decile", fontsize=pcfg.xlabel_size)
    # fig.supylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _self_hires_v_rank_box(ax, g, include_zero)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"self_hires_v_rank_box_{include_zero}"), bbox_inches='tight')


def _self_hires_v_rank_box(ax, g, include_zero):

    iden = g.name
    field = iden.split('_')[1]
    abbrev = qsi.get_abbrev(iden)

    max_rank = max_domestic_ranks[field]
    num_bins = 10

    from analysis.doc_type import divide_by_groups

    ranges = divide_by_groups(max_rank, num_bins)

    stats = {}

    fractions = {i: [] for i in range(len(ranges))}

    for src_id, dst_id, data in g.edges(data=True):

        if dst_id not in stats:
            stats[dst_id] = [0, 0]

        if src_id == dst_id:
            stats[dst_id][0] += 1
        else:

            bs_id = data['bs_inst_id']
            
            if bs_id is not None:
                if bs_id == dst_id:
                    stats[dst_id][0] += 1

        stats[dst_id][1] += 1

    for key, value in stats.items():

        if value[1] == 0:
            continue
        else:
            to_put = value[0] / value[1]

        drank = g.nodes[key][f"{abbrev}_rank_domestic"]

        if drank is None:
            continue

        for i, value in enumerate(ranges):
            if value[0] <= drank < value[1]:
                fractions[i].append(to_put)

    if include_zero is False:
        for key, value in fractions.items():
            dropped = [v for v in value if v != 0]
            fractions[key] = dropped

    data = [fractions[i] for i in range(len(ranges))]

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    # ax.set_ylim(0, 1)
    # ax.set_yticks(np.arange(0, 1.1, 0.25))

    categories = [f"{i + 1}" for i in range(len(ranges))]

    box = ax.boxplot(data, vert=True, patch_artist=True, widths=0.35, labels=categories)

    color = pcfg.colors[iden][1]

    for patch in box['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(pcfg.alpha)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    

def self_hires_v_rank_z(include_zero=True):
 
    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    axs[1].set_xlabel(r"$Group_{\pi_{placed, KR}}$", fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(r"$Fraction_{self-hire}$", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _self_hires_v_rank_z(ax, g, include_zero)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"self_hires_v_rank_z_{include_zero}"), bbox_inches='tight')


def _self_hires_v_rank_z(ax, g, include_zero):

    from networks.randomize import randomize

    iden = g.name
    field = iden.split('_')[1]
    abbrev = qsi.get_abbrev(iden)

    trial = 500

    iden = g.name
    randoms = randomize(g, times=trial)

    max_rank = max_domestic_ranks[field]
    num_bins = 10

    from analysis.doc_type import divide_by_groups

    ranges = divide_by_groups(max_rank, num_bins)

    stats = {}

    fractions = {i: [] for i in range(len(ranges))}

    for src_id, dst_id, data in g.edges(data=True):

        if dst_id not in stats:
            stats[dst_id] = [0, 0]

        if src_id == dst_id:
            stats[dst_id][0] += 1

        stats[dst_id][1] += 1

    for key, value in stats.items():

        if value[1] == 0:
            continue
        else:
            to_put = value[0] / value[1]

        if include_zero is False and to_put == 0:
            continue

        drank = g.nodes[key][f"{abbrev}_rank_domestic"]

        if drank is None:
            continue

        for i, value in enumerate(ranges):
            if value[0] <= drank < value[1]:
                fractions[i].append(to_put)

    means = []

    for i, value in sorted(fractions.items()):

        mean = np.mean(value)
        means.append(mean)

    means_rand_list = []

    for g_rand in randoms:

        stats_rand = {}
        fractions_rand = {i: [] for i in range(len(ranges))}

        for src_id, dst_id, data in g_rand.edges(data=True):

            if dst_id not in stats_rand:
                stats_rand[dst_id] = [0, 0]

            if src_id == dst_id:
                stats_rand[dst_id][0] += 1

            stats_rand[dst_id][1] += 1

        for key, value in stats_rand.items():

            if value[1] == 0:
                continue
            else:
                to_put = value[0] / value[1]

            if include_zero is False and to_put == 0:
                continue

            drank = g.nodes[key][f"{abbrev}_rank_domestic"]

            if drank is None:
                continue

            for i, value in enumerate(ranges):
                if value[0] <= drank < value[1]:
                    fractions_rand[i].append(to_put)

        means_rand = []

        for i, value in sorted(fractions_rand.items()):

            mean = np.mean(value)
            means_rand.append(mean)

        means_rand_list.append(means_rand)

    means_rand_t = [[] for _ in range(10)]

    for means_rand in means_rand_list:

        for i in range(10):
            means_rand_t[i].append(means_rand[i])

    means_rand = [np.mean(means_rand_t[i]) for i in range(10)]
    std_rand = [np.std(means_rand_t[i]) for i in range(10)]

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    categories = [f"{i + 1}" for i in range(len(ranges))]

    z_score = [(means[i] - means_rand[i]) / std_rand[i] if std_rand[i] > 0 else 0 for i in range(10)]

    ax.scatter(categories, means, alpha=pcfg.alpha, s=120, color='red', edgecolor='black')
    ax.bar(categories, means_rand, yerr=std_rand, alpha=pcfg.alpha,
           color=pcfg.colors[iden][1],
           edgecolor='black')
    
    # ax.scatter(categories, z_score)


def hires():

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(2, 2, figsize=(2 * pcfg.fig_xsize, 2 * pcfg.fig_ysize), dpi=pcfg.dpi)

    _hires_distro(axs[0, 0], 'global')
    _hires_z(axs[0, 1], 'global')

    _hires_distro(axs[1, 0], 'domestic')
    _hires_z(axs[1, 1], 'domestic')

    pal_bio = [pcfg.colors_bio[i] for i in [0, 2, 4]]
    pal_cs = [pcfg.colors_cs[i] for i in [0, 2, 4]]
    pal_phy = [pcfg.colors_phy[i] for i in [0, 2, 4]]
        
    handles_bio = [Patch(facecolor=pal_bio[i],
                         alpha=pcfg.alpha,
                         edgecolor='black',
                         linewidth=3) for i in range(len(pal_bio))]
    
    handles_cs = [Patch(facecolor=pal_cs[i],
                        alpha=pcfg.alpha,
                        edgecolor='black',
                        linewidth=3) for i in range(len(pal_cs))]
    
    handles_phy = [Patch(facecolor=pal_phy[i],
                         alpha=pcfg.alpha,
                         edgecolor='black',
                         linewidth=3) for i in range(len(pal_phy))]
    
    labels_root = ["Down-hires", "Self-hires", "Up-hires"]

    labels_bio = [f"Biology ({root})" for root in labels_root]
    labels_cs = [f"Computer Science ({root})" for root in labels_root]
    labels_phy = [f"Physics ({root})" for root in labels_root]

    handles = [h_field[i] for i in range(3) for h_field in [handles_bio, handles_cs, handles_phy]]
    labels = [l_field[i] for i in range(3) for l_field in [labels_bio, labels_cs, labels_phy]]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path('hires'), bbox_inches='tight')
    plt.clf()


def _hires_distro(ax, net_type):

    from analysis.hires import calc_hires

    results = calc_hires(net_type=net_type)

    ax.set_ylabel("Fraction of hires", fontsize=pcfg.ylabel_size)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    ax.tick_params(axis='x', which='major', labelsize=pcfg.small_tick_size)
    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)

    for iden, stat in results.items():
        
        name_to_put = iden.split('_')[1]

        if name_to_put == 'ComputerScience':
            name_to_put = 'Computer\nScience'

        do = stat['Down'] / stat['Total']
        se = stat['Self'] / stat['Total']
        up = stat['Up'] / stat['Total']

        selected = pcfg.colors[iden]
        palette_to_use = [selected[0], selected[2], selected[4]]

        ax.bar(name_to_put, do, color=palette_to_use[0],
               alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(name_to_put, se, color=palette_to_use[1],
               bottom=do, alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(name_to_put, up, color=palette_to_use[2],
               bottom=do+se, alpha=pcfg.alpha, edgecolor='black', linewidth=2)

                
def _hires_z(ax, net_type):

    from analysis.hires import calc_hires_z

    results = calc_hires_z(net_type)

    ax.set_ylabel('Z-scores', fontsize=pcfg.ylabel_size)

    ax.set_ylim(-10, 20)
    ax.set_yticks(range(-10, 21, 5))

    ax.tick_params(axis='x', which='major', labelsize=pcfg.small_tick_size)
    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)

    bar_width = 0.25
    x_pos = 0

    for iden, stat in results.items():

        x_pos_1 = x_pos
        x_pos_2 = x_pos + bar_width
        x_pos_3 = x_pos + 2 * bar_width

        selected = pcfg.colors[iden]
        palette_to_use = [selected[0], selected[2], selected[4]]

        do = stat["Down"]
        se = stat["Self"]
        up = stat["Up"]

        ax.bar(x_pos_1, do, width=bar_width, color=palette_to_use[0],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_2, se, width=bar_width, color=palette_to_use[1],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_3, up, width=bar_width, color=palette_to_use[2],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)

        x_pos += 1

    x_ticks = []

    names_to_put = ["Biology", "Computer\nScience", "Physics"]

    for name in names_to_put:
        x_ticks.append(name)

    ax.set_xticks([x + bar_width for x in range(3)], x_ticks)
    ax.axhline(0, color='black', linewidth=1)


def mobility_v_random(direction='up'):

    assert (direction in ['up', 'down'])

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    if direction == 'up':
        xlabel = 'Fraction of up-hires'
    else:
        xlabel = 'Fraction of down-hires'

    ylabel = 'Density'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _mobility_v_random(ax, g, direction)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"mobility_v_random_{direction}"), bbox_inches='tight')


def _mobility_v_random(ax, g, direction):

    import networkx as nx
    from scipy.stats import gaussian_kde, percentileofscore
    from networks.randomize import randomize
    from networks.rank import calc_rank_to_dict
    
    def calculate_p_value(empirical, distribution):
    
        ecdf = percentileofscore(distribution, empirical, kind='rank') / 100.0
    
        if ecdf > 0.5:
            p_value = 2 * (1 - ecdf)
        else:
            p_value = 2 * ecdf
        return p_value
    
    g_without_shires = nx.MultiDiGraph()
    iden = g.name

    trial = 1000

    for src_id, dst_id, data in g.edges(data=True):

        if src_id != dst_id:
            g_without_shires.add_edge(src_id, dst_id)

    randoms = randomize(g_without_shires, times=trial)
    ranks_dict = calc_rank_to_dict(g_without_shires)

    stat = 0
    total = 0

    for src_id, dst_id, data in g_without_shires.edges(data=True):

        src_rank = ranks_dict[src_id]
        dst_rank = ranks_dict[dst_id]

        if direction == 'down' and src_rank < dst_rank:
            stat += 1
        elif direction == 'up' and src_rank > dst_rank:
            stat += 1

        total += 1
    
    stat_g = stat / total
    stat_g_rand_list = []

    for g_rand in randoms:

        ranks_dict = calc_rank_to_dict(g_rand)

        stat = 0
        total = 0

        for src_id, dst_id, data in g_rand.edges(data=True):

            src_rank = ranks_dict[src_id]
            dst_rank = ranks_dict[dst_id]

            if direction == 'down' and src_rank < dst_rank:
                stat += 1
            elif direction == 'up' and src_rank > dst_rank:
                stat += 1
            
            total += 1

        ratio = stat / total

        stat_g_rand_list.append(ratio)

    mean_rand = np.mean(stat_g_rand_list)
    std_rand = np.std(stat_g_rand_list)

    data = np.array(stat_g_rand_list)

    kde = gaussian_kde(data, bw_method=0.5)
    x = np.linspace(0, 1, 1000)
    y = kde(x)

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)
    
    ax.plot(x, y, label='Density', color='black', linestyle='-', linewidth=2)
    ax.fill_between(x, y, alpha=pcfg.alpha-0.1, color=pcfg.colors[iden][1])

    ax.axvline(x=stat_g, color='black', linestyle='--', linewidth=2)

    pvalue = calculate_p_value(stat_g, stat_g_rand_list)
    print(pvalue)

    if direction == 'down':
        text_xpos = 0.445
    else:
        text_xpos = 0.95

    ax.text(text_xpos, 0.05,
                f'Z-score: {(stat_g - mean_rand) / std_rand: .3f}\nP-value: {pvalue: .3f}',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.small_tick_size, bbox=dict(facecolor='white', alpha=0.8))
    
    if direction == 'down':
        lower_bound_x = 0.8
        upper_bound_x = 1

    else:
        lower_bound_x = 0
        upper_bound_x = 0.2

    ax.set_xlim(lower_bound_x, upper_bound_x)
    ax.set_ylim(0, 80)

    ax.set_yticks(range(0, 81, 20))


def mobility_kr_us(direction='up', normalize=False):

    assert (direction in ['up', 'down', 'both'])
    assert (isinstance(normalize, bool))

    nets = construct_network()

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = 'Trained-placed relationship'

    unit = 'Fraction' if normalize else 'Number'

    if direction in ['up', 'down']:
        ylabel = f'{unit} of {direction}-hires'
    else:
        ylabel = f'{unit} of hires'

    # axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    # fig.supxlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _mobility_kr_us(ax, g, direction, normalize)

    handles = [Patch(facecolor='mediumblue', edgecolor='black', alpha=pcfg.alpha, linewidth=2),
               Patch(facecolor='tomato', edgecolor='black', alpha=pcfg.alpha, linewidth=2)]
    
    labels = ['Up-hires', 'Down-hires']

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False, fontsize=pcfg.legend_size)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"mobility_{direction}_us_kr_normalize({normalize})"),
                bbox_inches='tight')


def _mobility_kr_us(ax, g, direction, normalize):

    if direction == 'both':
        return _mobility_kr_us_both(ax, g, normalize)

    from analysis.mobility import calc_mobility

    iden = g.name

    stats = calc_mobility(g)

    print(stats)

    category = ['KR-trained\nKR-placed',
                'US-trained\nKR-placed',
                'KR-trained\nUS-placed']
    
    match direction:

        case 'up':
            index = 0
        case 'self':
            index = 1
        case 'down':
            index = 2
        case _:
            index = None

    ktkp = stats['KR2KR'][index]
    utkp = stats['US2KR'][index]
    ktup = stats['KR2US'][index]

    if normalize:
        ktkp = ktkp / stats['KR2KR'][3]
        utkp = utkp / stats['US2KR'][3]
        ktup = ktup / stats['KR2US'][3]

    data = [ktkp, utkp, ktup]

    ax.tick_params(axis='x', which='major', labelsize=pcfg.small_tick_size)
    ax.tick_params(axis='x', labelrotation=30)

    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)

    bar_width = 0.25

    ax.bar(category, data, width=bar_width, color=pcfg.colors[iden][1],
           alpha=pcfg.alpha, edgecolor='black', linewidth=2)

    if normalize:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.25))
    else:
        ax.set_ylim(0, 800)
        ax.set_yticks(range(0, 801, 200))


def _mobility_kr_us_both(ax, g, normalize):

    from analysis.mobility import calc_mobility

    iden = g.name

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

    print(x, category)

    if normalize:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.25))
    else:
        ax.set_ylim(0, 800)
        ax.set_yticks(range(0, 801, 200))


def deprecated_mobility_kr_us(normalize=False):
    
    from analysis.mobility import calc_mobility

    plt.rc('font', **pcfg.fonts)
    fig, ax = plt.subplots(1, 1, figsize=(2 * pcfg.fig_xsize, 1.5 * pcfg.fig_ysize), dpi=pcfg.dpi)

    nets = construct_network()
    results = {}

    for iden, g in nets.items():
        results[iden] = calc_mobility(g)

    ax.set_ylabel('Fraction of hires', fontsize=pcfg.ylabel_size)

    ax.tick_params(axis='x', which='major', labelsize=pcfg.tick_size)
    ax.tick_params(axis='y', which='major', labelsize=pcfg.tick_size)

    if normalize:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.25))
    else:
        ax.set_ylim(0, 800)
        ax.set_yticks(range(0, 801, 200))

    bar_width = 0.25
    x_pos = 0

    for iden, stat in results.items():

        x_pos_1 = x_pos
        x_pos_2 = x_pos + bar_width
        x_pos_3 = x_pos + 2 * bar_width

        selected = pcfg.colors[iden]
        palette_to_use = [selected[0], selected[2], selected[4]]

        kr_kr = stat["KR2KR"]
        us_kr = stat["US2KR"]
        kr_us = stat["KR2US"]

        if normalize:

            kr_kr = np.array(kr_kr) / kr_kr[3]
            us_kr = np.array(us_kr) / us_kr[3]
            kr_us = np.array(kr_us) / kr_us[3]

        ax.bar(x_pos_1, kr_kr[2], width=bar_width, color=palette_to_use[0],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_1, kr_kr[1], bottom=kr_kr[2], width=bar_width, color=palette_to_use[1],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_1, kr_kr[0], bottom=kr_kr[2]+kr_kr[1], width=bar_width, color=palette_to_use[2],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        
        ax.bar(x_pos_2, us_kr[2], width=bar_width, color=palette_to_use[0],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_2, us_kr[1], bottom=us_kr[2], width=bar_width, color=palette_to_use[1],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_2, us_kr[0], bottom=us_kr[2]+us_kr[1], width=bar_width, color=palette_to_use[2],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        
        ax.bar(x_pos_3, kr_us[2], width=bar_width, color=palette_to_use[0],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_3, kr_us[1], bottom=kr_us[2], width=bar_width, color=palette_to_use[1],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)
        ax.bar(x_pos_3, kr_us[0], bottom=kr_us[2]+kr_us[1], width=bar_width, color=palette_to_use[2],
                alpha=pcfg.alpha, edgecolor='black', linewidth=2)

        x_pos += 1

    x_ticks = []

    names_to_put = ["Biology", "Computer\nScience", "Physics"]

    for name in names_to_put:
        x_ticks.append(name)

    ax.set_xticks([x + bar_width for x in range(3)], x_ticks)
    ax.axhline(0, color='black', linewidth=1)# New sample data

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"mobility_kr_us_norm({normalize})"), bbox_inches='tight')


if __name__ == "__main__":

    # self_hires_v_rank()
    # # self_hires_v_rank_mean()
    # self_hires_v_rank_box(include_zero=True)
    # self_hires_v_rank_box(include_zero=False)
    # self_hires_v_rank_mean(include_zero=False)

    # # self_hires_v_rank_z(include_zero=True)
    mobility_v_random(direction='up')
    mobility_v_random(direction='down')
    # self_hires_v_rank_grouped()
    # self_hires_v_rank_grouped(net_type='domestic', annotate_bs=True)

    # mobility_kr_us(direction='both', normalize=True)
    # mobility_kr_us(direction='down', normalize=True)