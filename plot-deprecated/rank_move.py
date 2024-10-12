import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import numpy as np
import math
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import plot.config as pcfg
from networks.construct_net import construct_network
from config.config import identifiers, fig_path

highlighter = '#F7E94A'


def _interhistogram(x, y):

    f = interp1d(x, y, kind='quadratic', fill_value='extrapolate', bounds_error=False)

    x_co = np.arange(-1, 1.01, 0.01)
    y_co = f(x_co)
    y_co = np.maximum(y_co, 0)

    bin_width = np.diff(x_co)[0]
    area_under_curve = np.sum(y_co * bin_width)
    y_co_normalized = y_co / area_under_curve

    return x_co, y_co_normalized


def rank_move(net_type='both'):

    assert (net_type in ['both', 'global', 'domestic'])

    nets_g = construct_network(net_type='global')
    nets_d = construct_network(net_type='domestsic')

    plt.rc('font', **pcfg.fonts)

    fig, axs = plt.subplots(1, 3, sharey='all', figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets_g[iden] 
        d = nets_d[iden] 

        _rank_move(g, d, ax, net_type)

    x_label = 'Relative Movement of Faculty (Normalized)'
    y_label = 'Density'

    axs[0].set_ylabel(y_label, fontsize=pcfg.ylabel_size)
    axs[1].set_xlabel(x_label, fontsize=pcfg.xlabel_size)

    if net_type == 'both':

        handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=5),
                Line2D([0], [0], color='black', linestyle=':', linewidth=5),
                Patch(facecolor=pcfg.colors_bio[0], label="Biology",
                        alpha=pcfg.alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=pcfg.colors_cs[0], label="Computer Science",
                        alpha=pcfg.alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=pcfg.colors_phy[0], label="Physics",
                        alpha=pcfg.alpha, edgecolor='black', linewidth=3)]

        labels = ["Global", "Domestic", "Biology", "Computer Science", "Physics"]

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=5, frameon=False)

    else:

        handles = [Patch(facecolor=pcfg.colors_bio[0], label="Biology",
                        alpha=pcfg.alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=pcfg.colors_cs[0], label="Computer Science",
                        alpha=pcfg.alpha, edgecolor='black', linewidth=3),
                Patch(facecolor=pcfg.colors_phy[0], label="Physics",
                        alpha=pcfg.alpha, edgecolor='black', linewidth=3)]

        labels = ["Biology", "Computer Science", "Physics"]

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)

    for axi in axs.flatten():

        axi.set_xlim(-1, 1)
        axi.set_xticks(np.arange(-1, 1.1, 0.5))
        axi.tick_params(axis='both', which='both', labelsize=pcfg.tick_size)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f'rank_move_{net_type}'), bbox_inches='tight')
    plt.clf()


def _rank_move(g, d, ax, net_type, group=None, export_style=None, normalize_sep=True, for_inset=False):

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

        (x_interp, y_interp) = _interhistogram(x, normalized_hist)

        if not normalize_sep:

            ratio = len(move) / g.number_of_edges()

            print(ratio)

            y_interp = [y * ratio for y in y_interp]

        ax.plot(x_interp, y_interp, color=pcfg.colors[iden][0], linewidth=5,
                alpha=pcfg.alpha, linestyle=style_to_put)

    if for_inset:
        ax.set_yticks([])

    else:

        ymin = 0    
        ymax = 20

        ax.set_ylim(ymin, ymax)
        ax.set_yticks(range(ymin, ymax + 1, 5))


def rank_move_grouped(net_type='domestic', inset=False):

    assert (net_type in ['global', 'domestic'])

    nets = construct_network(net_type=net_type)

    plt.rc('font', **pcfg.fonts)

    fig, axs = plt.subplots(1, 3, sharey='all', figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    groups = [(1, 5), (3, 5), (5, 5)]
    styles = ['-', ':', '--']

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        if inset:
            ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
            ax_inset.set_xlim(-1, 1)

        for group, style in zip(groups, styles):
            _rank_move(g, None, ax, net_type=net_type, group=group, export_style=style)

            if inset:
                _rank_move(g, None, ax_inset, net_type=net_type, group=group, export_style=style, normalize_sep=False, for_inset=True)

    x_label = 'Relative Movement of Faculty (Normalized)'
    y_label = 'Density'

    axs[0].set_ylabel(y_label, fontsize=pcfg.ylabel_size)
    axs[1].set_xlabel(x_label, fontsize=pcfg.xlabel_size)
    
    handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=4),
               Line2D([0], [0], color='black', linestyle=':', linewidth=4),
               Line2D([0], [0], color='black', linestyle='--', linewidth=4)]

    labels = ["Fifth quintile", "Third quntile", "First quntile"]

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)

    for axi in axs.flatten():

        axi.set_xlim(-1, 1)
        axi.set_xticks(np.arange(-1, 1.1, 0.5))
        axi.tick_params(axis='both', which='both', labelsize=pcfg.tick_size)

    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f'rank_move_grouped_{net_type}_inset({inset})'), bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    rank_move_grouped(net_type='domestic', inset=False)
    rank_move_grouped(net_type='domestic', inset=True)
