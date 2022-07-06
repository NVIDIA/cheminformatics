import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import seaborn as sns
from .data import PHYSCHEM_UNIT_RENAMER, MEGAMOLBART_SAMPLE_RADIUS, load_aggregated_metric_results, make_aggregated_embedding_df
from .utils import _label_bars, ACCEPTANCE_CRITERIA

__ALL__ = ['create_timeseries_aggregated_plots']


def get_constant_inferrers(dat, group_col):
    mask = dat['inferrer'].str.contains('MegaMolBART').pipe(np.invert)
    constant_dat = dat[mask]
    idx = constant_dat.groupby(['inferrer', group_col])['timestamp'].idxmax()
    constant_dat = constant_dat.loc[idx]
    return constant_dat


def get_timeseries_inferrers(dat):
    mask = dat['inferrer'].str.contains('MegaMolBART')
    ts_dat = dat[mask]
    model_size_mask = ts_dat.apply(lambda x: MEGAMOLBART_SAMPLE_RADIUS[x['model_size']] == x['radius'], axis=1)
    ts_dat = ts_dat.loc[model_size_mask]
    return ts_dat


def make_timeseries_sampling_plots(metric_df, axlist):
    """Make aggregate plots for validity, uniqueness, novelty --
       will be bar chart for single date or timeseries for multiple"""

    # Select data
    generative_mask = metric_df['name'].isin(['validity', 'unique', 'novelty'])
    generative_df = metric_df[generative_mask].dropna(axis=1, how='all')

    # Set sort order by using a categorical
    cat = pd.CategoricalDtype(['validity', 'novelty', 'unique'], ordered=True)
    generative_df['name'] = generative_df['name'].astype(cat)

    grouper = generative_df.groupby('name')
    timestamp_lim = (metric_df['timestamp'].min() - 1, metric_df['timestamp'].max() + 1)

    for ax, (metric, dat) in zip(axlist, grouper):
        ax2 = ax.twiny()

        if not isinstance(dat, pd.DataFrame):
            dat = dat.to_frame()

        is_first_col = True if ax is axlist[0] else False

        if metric in ACCEPTANCE_CRITERIA:
            ax.axhline(y=ACCEPTANCE_CRITERIA[metric], xmin=0, xmax=1, color='red', lw=1.0, zorder=-1)

        # Bar plot data
        bar_dat = get_constant_inferrers(dat, group_col='radius')
        bar_dat = bar_dat.pivot(columns='radius', values='value', index='inferrer')
        bar_dat.plot(kind='bar', ax=ax, rot=0, legend=is_first_col)

        # Line plot data
        ts_dat = get_timeseries_inferrers(dat)
        ts_dat = ts_dat.pivot(columns='inferrer', values='value', index='timestamp')
        ts_dat.plot(kind='line', marker='o', ax=ax2, legend=is_first_col)

        ax.set_ylim(0, 1.1)
        # ax.set_xlim(*timestamp_lim) # TODO FIX ME
        ax.set(title=f'Sampling: \n{metric.title()}', xlabel='Model (Timeseries Benchmark)', ylabel='Ratio')


def make_timeseries_nearest_neighbor_plot(embedding_df, axlist):
    """Aggregate plot for nearest neighbor correlation ---
       bar chart for single time point, time series for multiple"""
    ax = axlist[0]
    ax2 = ax.twiny()

    dat = embedding_df[embedding_df.name == 'nearest neighbor correlation']
    dat = dat[['timestamp', 'inferrer', 'top_k', 'value']].drop_duplicates()
    dat['top_k'] = dat['top_k'].astype(int)
    timestamp_lim = (dat['timestamp'].min() - 1, dat['timestamp'].max() + 1)
    
    # Bar plot data
    bar_dat = get_constant_inferrers(dat, group_col='top_k')
    bar_dat = bar_dat.pivot(index='inferrer', columns='top_k', values='value')
    bar_dat.plot(kind='bar', ax=ax, rot=0, legend=True)

    # Line plot data
    ts_dat = dat[dat['inferrer'].str.contains('MegaMolBART')]
    ts_dat = ts_dat.pivot(columns=['inferrer', 'top_k'], values='value', index='timestamp')
    ts_dat.plot(kind='line', marker='o', ax=ax2, legend=False)

    # ax.set_ylim(*ylim)
    # ax.set_xlim(*timestamp_lim)
    ax.set(title='Nearest Neighbor Metric', ylabel="Speaman's Rho", xlabel='Benchmark Date (Development Models)')


def make_timeseries_physchem_plots(embedding_df, output_dir, max_plot_ratio=10):
    """Plots of phychem property results"""

    dat = embedding_df[embedding_df.name == 'physchem']
    dat['property'] = dat['property'].map(lambda x: PHYSCHEM_UNIT_RENAMER[x])
    dat = dat[['timestamp', 'inferrer', 'property', 'model', 'value']]
    timestamp_lim = (dat['timestamp'].min() - 1, dat['timestamp'].max() + 1)

    grouper = dat.groupby('inferrer')
    n_models = len(grouper)
    fig, axes = plt.subplots(ncols=2, nrows=n_models, figsize=(16, 4*n_models))
    axes = axes[np.newaxis, :] if axes.ndim == 1 else axes

    for row, (inferrer, dat_) in enumerate(grouper):
        # Latest values plot
        ax = axes[row, 0]
        last_timestep = dat_.sort_values('timestamp').groupby(['inferrer', 'property', 'model']).last().reset_index()
        last_timestep = last_timestep.pivot(columns=['model'], values='value', index='property')
        _ = last_timestep.plot(kind='bar', width=0.8, legend=False, ax=ax, rot=0)
        ax.axhline(1.0, 0, 1, color='red', lw=1.0, zorder=-1)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize='small')
        ax.set_ylim(0, max_plot_ratio)
        _label_bars(ax, int(0.9 * max_plot_ratio))
        ax.set_title('Physchem Property Prediction (Most Recent Benchmark)') if ax.get_subplotspec().is_first_row() else ax.set_title('')
        ax.set_ylabel(f'{inferrer}\nMSE Ratio') if ax.get_subplotspec().is_first_col() else ax.set_ylabel('')
        ax.set_xlabel('Property') if ax.get_subplotspec().is_last_row() else ax.set_xlabel('')

        if row == 0:
            handles, labels = ax.get_legend_handles_labels()

        # Timeseries plot
        ax = axes[row, 1]
        timeseries_data = dat_.pivot_table(columns=['model'], values='value', index='timestamp', aggfunc='mean')
        _ = timeseries_data.plot(kind='line', marker='o', legend=False, ax=ax, rot=0)
        ax.axhline(1.0, 0, 1, color='red', lw=1.0, zorder=-1)
        ax.set_xlim(*timestamp_lim)
        ax.set_ylim(0, max_plot_ratio)
        ax.set_title('Physchem Property Prediction (Mean of All Properties as Timeseries)') if ax.get_subplotspec().is_first_row() else ax.set_title('')
        ax.set_ylabel(f'Average MSE Ratio (All Properties)')
        ax.set_xlabel('Timestamp') if ax.get_subplotspec().is_last_row() else ax.set_xlabel('')

    fig = plt.gcf()
    fig.legend(handles=handles, labels=labels, loc=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Physchem_Aggregated_Benchmark.png'), dpi=300)


def make_timeseries_bioactivity_plots(embedding_df, output_dir, max_plot_ratio=6):
    sns.set_palette('dark')
    
    dat = embedding_df[embedding_df.name == 'bioactivity']
    dat = dat[['timestamp', 'inferrer', 'gene', 'model', 'value']]
    dat.sort_values('gene', inplace=True)

    gene_list = sorted(dat['gene'].unique())
    n_genes = len(gene_list)
    gene_labels = pd.CategoricalDtype(gene_list, ordered=True)
    dat['gene'] = dat['gene'].astype(gene_labels)

    timestamp_lim = (dat['timestamp'].min() - 1, dat['timestamp'].max() + 1)

    grouper = dat.groupby('inferrer')
    n_models = len(grouper)
    fig = plt.figure(figsize=(16, 4 * n_models))
    plot_dims = (n_models, 4)

    for row, (inferrer, dat) in enumerate(grouper):
        # Line plot of last timestep
        if row == 0:
            ax = plt.subplot2grid(plot_dims, (row, 0), colspan=3)
            ax0 = ax
        else:
            ax = plt.subplot2grid(plot_dims, (row, 0), colspan=3, sharex=ax0)

        last_timestep = dat.sort_values('timestamp').groupby(['inferrer', 'gene', 'model']).last().reset_index()
        last_timestep = last_timestep.pivot(columns=['model'], values='value', index='gene')
        _ = last_timestep.plot(kind='line', marker='o', legend=False, ax=ax, rot=70)
        ax.axhline(1.0, 0, 1, color='red', lw=1.0, zorder=-1)
        ax.set(title='Bioactivity Prediction\n(Most Recent Benchmark)', xlabel='Gene', ylabel=f'{inferrer}\nMSE Ratio')
        ax.set_xlim(-0.1, n_genes + 0.1)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xticks(range(0, n_genes))
        ax.set_xticklabels(gene_list, fontsize='8', rotation=70)
        ax.set_ylim(0, max_plot_ratio)

        # Timeseries plot
        ax = plt.subplot2grid(plot_dims, (row, 3))

        timeseries = dat.pivot_table(columns=['model'], values='value', index='timestamp', aggfunc='mean')
        legend = True if row == 0 else False
        _ = timeseries.plot(kind='line', marker='o', legend=legend, ax=ax, rot=0)
        ax.axhline(1.0, 0, 1, color='red', lw=1.0, zorder=-1)
        ax.set_xlim(*timestamp_lim)
        ax.set_ylim(0, max_plot_ratio)
        ax.set(title='Bioactivity Timeseries\n(Mean over all Genes)', xlabel='Timestamp', ylabel=f'{inferrer}\nMSE Average Ratio')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Bioactivity_Aggregated_Benchmark.png'), dpi=300)


def create_timeseries_aggregated_plots(output_dir, plot_dir):
    """Create all aggregated plots for sampling and embedding metrics"""
    metric_df = load_aggregated_metric_results(output_dir)
    embedding_df = make_aggregated_embedding_df(metric_df)

    sns.set_palette('dark')
    pal = sns.color_palette()
    sns.set_palette([pal[0]] + pal[2:])
    sns.set_style('whitegrid', {'axes.edgecolor': 'black', 'axes.linewidth': 1.5})

    ncols, nrows = 3, 2
    fig = plt.figure(figsize=(ncols*4, nrows*4))
    axlist0 = [plt.subplot2grid((nrows, ncols), (0, x)) for x in range(0, 3)]
    axlist1 = [plt.subplot2grid((nrows, ncols), (1, 0))]
    axlist2 = [plt.subplot2grid((nrows, ncols), (1, 1))]
    axlist3 = [plt.subplot2grid((nrows, ncols), (1, 2))]

    make_timeseries_sampling_plots(metric_df, axlist0)
    make_timeseries_nearest_neighbor_plot(embedding_df, axlist1)
    # make_timeseries_physchem_plots(embedding_df, axlist2)
    # make_timeseries_bioactivity_plots(embedding_df, axlist3)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'Timeseries_Benchmark_Metrics.png'), dpi=300)
    return