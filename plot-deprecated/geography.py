import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import scipy.stats
from matplotlib.lines import Line2D
import statsmodels.api as sm

from networks.construct_net import construct_network
from config.config import identifiers, fig_path
import parse.queries_integration as qsi
import plot.config as pcfg
from plot.trained_rank import decades_to_interval

data_format = {
        'latitude': [],
        'longitude': [],
        'size': []
    }

max_wapman_ranks = {"Biology": 201,
                    "ComputerScience": 216,
                    "Physics": 214}

max_domestic_ranks = {"Biology": 78,
                    "ComputerScience": 105,
                    "Physics": 65}


def get_skorea_map(data):

    gdf = gpd.read_file('./data/else/skorea-provinces-geo.json')

    df = pd.DataFrame(data)

    # Create a GeoDataFrame for the bubble plot data
    gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    return gdf, gdf_points


def lcurve_inst_region():

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = 'Fraction of regional classification'
    ylabel = 'Fraction of institutions'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _lcurve_inst_region(ax, g)
    
    plt.tight_layout(pad=1)
    plt.savefig(fig_path("lcurve_inst_region"), bbox_inches='tight')
    

def _lcurve_inst_region(ax, g):

    from analysis.geography import _calc_inst_by_region

    iden = g.name
    
    data = _calc_inst_by_region(g)

    data = list(data.values())

    data.sort(reverse=True)

    xco = []
    yco = []

    x = 0
    y = 0

    xco.append(x)
    yco.append(y)

    for d in data:

        x += 1
        y += d

        xco.append(x)
        yco.append(y)

    xco = np.array(xco) / max(xco)
    yco = np.array(yco) / max(yco)

    area = np.trapz(yco, xco)
    gini = (area - 0.5) / 0.5

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    ax.plot(xco, yco, alpha=pcfg.alpha, c=pcfg.colors[iden][1], linewidth=4, marker='s', markersize=10)
    ax.plot([0, 1], [0, 1], c='black', alpha=pcfg.alpha)

    ax.text(0.95, 0.05,
                f'Gini coefficient: {gini:.3f}',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.small_tick_size, bbox=dict(facecolor='white', alpha=0.8))


def lcurve_deg_region(direction='out'):

    assert (direction in ['in', 'out'])

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = 'Fraction of regional classification'
    ylabel = f'Fraction of {direction}-degrees'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _lcurve_deg_region(ax, g, direction)
    
    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"lcurve_deg_{direction}_region"), bbox_inches='tight')
    

def _lcurve_deg_region(ax, g, direction):

    from analysis.geography import _calc_deg_by_region

    iden = g.name
    
    data = _calc_deg_by_region(g, direction)

    data = list(data.values())

    data.sort(reverse=True)

    xco = []
    yco = []

    x = 0
    y = 0

    xco.append(x)
    yco.append(y)

    for d in data:

        x += 1
        y += d

        xco.append(x)
        yco.append(y)

    xco = np.array(xco) / max(xco)
    yco = np.array(yco) / max(yco)

    area = np.trapz(yco, xco)
    gini = (area - 0.5) / 0.5

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.set_yticks(np.arange(0, 1.1, 0.25))

    ax.plot(xco, yco, alpha=pcfg.alpha, c=pcfg.colors[iden][1], linewidth=4, marker='s', markersize=10)
    ax.plot([0, 1], [0, 1], c='black', alpha=pcfg.alpha)

    ax.text(0.95, 0.05,
                f'Gini coefficient: {gini:.3f}',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.small_tick_size, bbox=dict(facecolor='white', alpha=0.8))
    

def trained_dist_v_placed_dist(dec="Overall", average=False, normalize=False):

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi)

    xlabel = "Distance from placed institution to Seoul"
    ylabel = "Distance from trained institution to Seoul"

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _trained_dist_v_placed_dist(ax, g, dec, average, normalize)

    # plt.show()
    plt.tight_layout(pad=1)
    plt.savefig(fig_path(f"trained_dist_v_placed_dist_{dec}_average({average})_normalize({normalize})"),
                bbox_inches='tight')


def _trained_dist_v_placed_dist(ax, g, dec, average, normalize):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    trained_dists = []
    placed_dists = []

    max_trained_dist = 500
    max_placed_dist = 500

    interval = decades_to_interval(dec)

    for src_id, dst_id, data in g.edges(data=True):

        src_nation = g.nodes[src_id]['nation']
        dst_nation = g.nodes[dst_id]['nation']

        if src_nation != 'KR' or dst_nation != 'KR':
            continue

        phd_granted_year = data['phd_end_year']

        if phd_granted_year is None:
            continue

        if not interval[0] <= phd_granted_year < interval[1]:
            continue
        
        t_dist = g.nodes[src_id]['distance_to_seoul']
        p_dist = g.nodes[dst_id]['distance_to_seoul']

        if t_dist is None or p_dist is None:
            continue

        trained_dists.append(t_dist)
        placed_dists.append(p_dist)

    if normalize:
        trained_dists = list(np.array(trained_dists) / max_trained_dist)
        placed_dists = list(np.array(placed_dists) / max_placed_dist)

    if average:

        gathering = {}

        for t, p in zip(trained_dists, placed_dists):

            if p not in gathering:
    
                gathering[p] = []
                gathering[p].append(t)

            else:
                gathering[p].append(t)

        gathering_mean = {}

        for p, ts in gathering.items():
            gathering_mean[p] = np.mean(ts)

        trained_dists = list(gathering_mean.values())
        placed_dists = list(gathering_mean.keys())

    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    xlim = max_placed_dist
    ylim = max_trained_dist

    if normalize:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xticks(np.arange(0, 1.1, 0.25))
        ax.set_yticks(np.arange(0, 1.1, 0.25))

    else:
        ax.set_xlim(0, xlim + 1)
        ax.set_ylim(0, ylim + 1)

        ax.set_xticks(range(0, xlim + 1, 100))
        ax.set_yticks(range(0, ylim + 1, 100))

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.scatter(placed_dists, trained_dists, marker='o',
               alpha=pcfg.alpha, s=150, c=pcfg.color_kt,
               edgecolor='black')
    
    if all(data is not None for data in [trained_dists, placed_dists]):

        placed_ranks = np.array(placed_dists)
        trained_ranks = np.array(trained_dists)

        results = scipy.stats.linregress(placed_ranks, trained_ranks)
        slope_kt = results.slope
        intercept = results.intercept
        r_value_kt = results.rvalue
        p_value_kt = results.pvalue

        # print(f"=== {iden}, {dec} ===")
        # print(f"Slope: {slope_kt}")
        # print(f"Intercept: {intercept}")
        # print(f"R-squared: {r_value_kt**2}")
        # print(f"P-value: {p_value_kt}")
        # print(f"Standard error: {std_err}")
        
        x_vals = np.array(ax.get_xlim())
        y_vals = slope_kt * x_vals + intercept
        ax.plot(x_vals, y_vals, color=pcfg.color_kt, linestyle='-', linewidth=2)

        label_to_put = f'Slope: {slope_kt:.2f}\n$R^2$: {r_value_kt**2:.2f}\nP-value: {p_value_kt:.2e}'

        ax.text(0.95, 0.05,
                label_to_put,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=pcfg.small_tick_size, bbox=dict(facecolor='white', alpha=0.8))
        

def rank_by_geo():

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    # axs[1].set_xlabel("Rank group", fontsize=pcfg.xlabel_size)
    # axs[0].set_ylabel("Longitude", fontsize=pcfg.xlabel_size)

    # fig.supxlabel("Latitude", fontsize=pcfg.xlabel_size)
    # fig.supylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        if i == 2:
            cmap, norm = _rank_by_geo(ax, g, return_cm=True)
        else:
            _rank_by_geo(ax, g)

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # Only needed for older matplotlib versions
    # cbar = fig.colorbar(sm, ax=axs, orientation='horizontal', pad=0.4, anchor=(1))
    # cbar.set_label('Color Value')

    plt.tight_layout(pad=1)

    plt.savefig(fig_path("rank_v_geography"), bbox_inches='tight')


def _rank_by_geo(ax, g, return_cm=False):

    iden = g.name
    abbrev = qsi.get_abbrev(iden)

    min_rank = 1
    max_rank = max_domestic_ranks[iden.split('_')[1]]

    scale_1 = 1
    scale_2 = 30

    geodata = {
        'latitude': [],
        'longitude': [],
        'size': [],
        'degree': [],
        'color': []
    }
    
    for id, data in g.nodes(data=True):

        lat = data['latitude']
        lng = data['longitude']

        rank = data[f'{abbrev}_rank_domestic']

        geodata['latitude'].append(lat)
        geodata['longitude'].append(lng)
        geodata['size'].append(rank / max_rank * scale_1)
        geodata['degree'].append((g.out_degree(id) + 1) * scale_2)
        geodata['color'].append(None)

    gdf, gdf_points = get_skorea_map(geodata)

    norm = plt.Normalize(0, scale_1)

    cmap = plt.cm.inferno_r

    gdf_points['color'] = gdf_points['size'].apply(lambda x: cmap(norm(x)))

    gdf.plot(ax=ax, color='gainsboro', edgecolor='black', alpha=pcfg.alpha)

    gdf_points.plot(ax=ax, markersize=gdf_points['degree'], 
                    color=gdf_points['color'], alpha=pcfg.alpha,
                    edgecolor='black', linewidth=1)
    
    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xticks([])
    ax.set_yticks([])

    if return_cm:
        return cmap, norm



def outdeg_by_geo():

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    # axs[1].set_xlabel("Rank group", fontsize=pcfg.xlabel_size)
    # axs[0].set_ylabel("Longitude", fontsize=pcfg.xlabel_size)

    # fig.supxlabel("Latitude", fontsize=pcfg.xlabel_size)
    # fig.supylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _outdeg_by_geo(ax, g)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"deg_out_v_geography"), bbox_inches='tight')


def _outdeg_by_geo(ax, g):

    iden = g.name

    geodata = deepcopy(data_format)
    
    for id, data in g.nodes(data=True):

        lat = data['latitude']
        lng = data['longitude']

        deg = g.out_degree(id)

        geodata['latitude'].append(lat)
        geodata['longitude'].append(lng)
        geodata['size'].append(deg * 25)

    gdf, gdf_points = get_skorea_map(geodata)

    gdf.plot(ax=ax, color='gainsboro', edgecolor='black', alpha=pcfg.alpha)

    gdf_points.plot(ax=ax, markersize=gdf_points['size'],
                    color=pcfg.colors[iden][2], alpha=pcfg.alpha,
                    edgecolor='black', linewidth=2)
    
    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xticks([])
    ax.set_yticks([])


def radg_by_dist(dist_type='geo', ver=2, add_reg=False):

    assert (dist_type in ['geo', 'rank'])
    assert (ver in [1, 2])

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    xlabel = 'Distance to Seoul (km)'

    if ver == 1:
        ylabel = 'RMS displacement (km)' if dist_type == 'geo' else 'RMS rank move'

    else:
        ylabel = 'Radius of gyration (km)' if dist_type == 'geo' else 'Radius of gyration (rank)'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    # fig.supxlabel("Latitude", fontsize=pcfg.xlabel_size)
    # fig.supylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _radg_by_dist(ax, g, dist_type, ver, add_reg)

    if add_reg:

        handles = [Line2D([0], [0], color=pcfg.color_ut, linewidth=6, alpha=pcfg.alpha),
                   Line2D([0], [0], color=pcfg.color_kt, linewidth=6, alpha=pcfg.alpha)]
        
        labels = ['1st order fit', '2nd order fit']

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
                   ncol=2, frameon=False, fontsize=pcfg.legend_size)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"radg_{dist_type}_v_dist_ver{ver}_reg({add_reg})"), bbox_inches='tight')


def _radg_by_dist(ax, g, dist_type, ver, add_reg):

    from analysis.geography import _calc_radg, _calc_radg_ver2

    iden = g.name

    func_calc_radg = _calc_radg if ver == 1 else _calc_radg_ver2

    radgs = func_calc_radg(g, dist_type=dist_type)

    xco = []
    yco = []
    
    for id, data in g.nodes(data=True):

        if id not in radgs:
            continue

        dist = data['distance_to_seoul']

        xco.append(dist)
        yco.append(radgs[id])

    ax.scatter(xco, yco, marker='o',
               alpha=pcfg.alpha, s=150, c=pcfg.colors[iden][1],
               edgecolor='black')
    
    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xlim(0, 500)
    ax.set_xticks(range(0, 501, 100))

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
                fontsize=pcfg.small_tick_size - 10, bbox=dict(facecolor='white', alpha=0.8))

    if dist_type == 'geo':
        ax.set_ylim(0, 500)
        ax.set_yticks(range(0, 501, 100))

    else:
        ax.set_ylim(0, 40)
        ax.set_yticks(range(0, 41, 10))


def radg_by_rank(dist_type='geo', ver=2, add_reg=False):

    assert (dist_type in ['geo', 'rank'])
    assert (ver in [1, 2])

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    xlabel = 'Rank'

    if ver == 1:
        ylabel = 'RMS displacement (km)' if dist_type == 'geo' else 'RMS rank move'

    else:
        ylabel = 'Radius of gyration (km)' if dist_type == 'geo' else 'Radius of gyration (rank)'

    axs[1].set_xlabel(xlabel, fontsize=pcfg.xlabel_size)
    axs[0].set_ylabel(ylabel, fontsize=pcfg.xlabel_size)

    # fig.supxlabel("Latitude", fontsize=pcfg.xlabel_size)
    # fig.supylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _radg_by_rank(ax, g, dist_type, ver, add_reg)

    if add_reg:

        handles = [Line2D([0], [0], color=pcfg.color_ut, linewidth=6, alpha=pcfg.alpha),
                   Line2D([0], [0], color=pcfg.color_kt, linewidth=6, alpha=pcfg.alpha)]
        
        labels = ['1st order fit', '2nd order fit']

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
                   ncol=2, frameon=False, fontsize=pcfg.legend_size)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"radg_{dist_type}_v_rank_ver{ver}_reg({add_reg})"), bbox_inches='tight')


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
                fontsize=pcfg.small_tick_size - 10, bbox=dict(facecolor='white', alpha=0.8))

    if dist_type == 'geo':
        ax.set_ylim(0, 500)
        ax.set_yticks(range(0, 501, 100))

    else:
        pass
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))


def radg_by_geo(dist_type='geo'):

    assert (dist_type in ['geo', 'rank'])

    nets = construct_network(net_type='domestic')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    # axs[1].set_xlabel("Rank group", fontsize=pcfg.xlabel_size)
    # axs[0].set_ylabel("Longitude", fontsize=pcfg.xlabel_size)

    # fig.supxlabel("Latitude", fontsize=pcfg.xlabel_size)
    # fig.supylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _radg_by_geo(ax, g, dist_type)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"radg_{dist_type}_v_geography"), bbox_inches='tight')


def _radg_by_geo(ax, g, dist_type):

    from analysis.geography import _calc_radg_ver2

    scale = 10

    iden = g.name

    geodata = deepcopy(data_format)

    radgs = _calc_radg_ver2(g, dist_type=dist_type)

    print(radgs)
    
    for id, data in g.nodes(data=True):

        if id not in radgs:
            continue

        lat = data['latitude']
        lng = data['longitude']

        size = radgs[id]

        geodata['latitude'].append(lat)
        geodata['longitude'].append(lng)
        geodata['size'].append(size * scale)

    gdf, gdf_points = get_skorea_map(geodata)

    gdf.plot(ax=ax, color='gainsboro', edgecolor='black', alpha=pcfg.alpha)

    gdf_points.plot(ax=ax, markersize=gdf_points['size'],
                    color=pcfg.colors[iden][2], alpha=pcfg.alpha,
                    edgecolor='black', linewidth=2)
    
    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xticks([])
    ax.set_yticks([])


def doctype_by_geo():

    nets = construct_network(net_type='global')

    plt.rc('font', **pcfg.fonts)
    fig, axs = plt.subplots(1, 3, figsize=(3 * pcfg.fig_xsize, pcfg.fig_ysize), dpi=pcfg.dpi, sharey=True)

    # axs[1].set_xlabel("Distance to Seoul (km)", fontsize=pcfg.xlabel_size)
    # axs[0].set_ylabel("Radius of ", fontsize=pcfg.xlabel_size)

    # # fig.supxlabel("Latitude", fontsize=pcfg.xlabel_size)
    # fig.supylabel("Fraction of self-hires", fontsize=pcfg.xlabel_size)

    for i, iden in enumerate(identifiers):

        ax = axs[i]
        g = nets[iden] 

        _doctype_by_geo(ax, g)

    plt.tight_layout(pad=1)

    plt.savefig(fig_path(f"doc_type_v_geography"), bbox_inches='tight')


def _doctype_by_geo(ax, g):

    from analysis.doc_type import _calc_doc_type_per_inst

    scale = 1000

    iden = g.name

    geodata = deepcopy(data_format)

    stats = _calc_doc_type_per_inst(g)

    stats_frac = {}

    for id, stat in stats.items():

        if stat['Total'] == 0:
            continue

        stats_frac[id] = stat['US'] / stat['Total']
    
    for id, data in g.nodes(data=True):

        if id not in stats_frac:
            continue

        lat = data['latitude']
        lng = data['longitude']

        size = stats_frac[id]

        geodata['latitude'].append(lat)
        geodata['longitude'].append(lng)
        geodata['size'].append(size * scale)

    gdf, gdf_points = get_skorea_map(geodata)

    gdf.plot(ax=ax, color='gainsboro', edgecolor='black', alpha=pcfg.alpha)

    gdf_points.plot(ax=ax, markersize=gdf_points['size'],
                    color=pcfg.colors[iden][2], alpha=pcfg.alpha,
                    edgecolor='black', linewidth=2)
    
    ax.tick_params(axis='both', which='major', labelsize=pcfg.tick_size)

    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":

    # lcurve_deg_region(direction='out')

    radg_by_rank(dist_type='rank', ver=1, add_reg=True)
    radg_by_rank(dist_type='geo', ver=1, add_reg=True)

    radg_by_dist(dist_type='rank', ver=1, add_reg=True)
    radg_by_dist(dist_type='geo', ver=1, add_reg=True)

    # trained_dist_v_placed_dist(average=False, normalize=False)
    # trained_dist_v_placed_dist(average=False, normalize=True)

    # trained_dist_v_placed_dist(average=True, normalize=False)
    # trained_dist_v_placed_dist(average=True, normalize=True)