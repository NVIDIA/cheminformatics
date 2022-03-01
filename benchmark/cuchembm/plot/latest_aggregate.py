import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import seaborn as sns
from .data import PHYSCHEM_UNIT_RENAMER, load_aggregated_metric_results, make_aggregated_embedding_df
from .utils import _label_bars, ACCEPTANCE_CRITERIA

__ALL__ = ['create_latest_aggregated_plots']


def make_latest_sampling_plots(metric_df, axlist):
    """Make aggregate plots for validity, uniqueness, novelty --
       will be bar chart for single date or timeseries for multiple"""

    # Select data
    generative_mask = metric_df['name'].isin(['validity', 'unique', 'novelty'])
    generative_df = metric_df[generative_mask].dropna(axis=1, how='all')

    # Set sort order by using a categorical
    cat = pd.CategoricalDtype(['validity', 'novelty', 'unique'], ordered=True)
    generative_df['name'] = generative_df['name'].astype(cat)

    grouper = generative_df.groupby('name')
    for ax, (metric, dat) in zip(axlist, grouper):
        if not isinstance(dat, pd.DataFrame):
            dat = dat.to_frame()

        is_first_col = True if ax is axlist[0] else False

        # Bar plot of most recent benchmark for all models
        if metric in ACCEPTANCE_CRITERIA:
            ax.axhline(y=ACCEPTANCE_CRITERIA[metric], xmin=0, xmax=1, color='red', lw=1.0, zorder=-1)

        idx = dat.groupby(['inferrer', 'radius'])['timestamp'].idxmax()
        bar_dat = dat.loc[idx]
        (bar_dat.pivot(columns='radius', 
                             values='value', 
                             index='inferrer')
            .plot(kind='bar', ax=ax, rot=0, legend=is_first_col))

        if is_first_col:
            ax.legend().set_title('Radius')

        _label_bars(ax)
        ax.set_ylim(0, 1.1)
        ax.set(title=f'Sampling: \n{metric.title()}', xlabel='Model (Latest Benchmark)', ylabel='Ratio')


def make_latest_nearest_neighbor_plot(embedding_df, axlist):
    """Aggregate plot for nearest neighbor correlation ---
       bar chart for single time point, time series for multiple"""
    ax = axlist[0]

    dat = embedding_df[embedding_df.name == 'nearest neighbor correlation']
    dat = dat[['timestamp', 'inferrer', 'top_k', 'value']].drop_duplicates()
    dat['top_k'] = dat['top_k'].astype(int)
    
    # Bar plot of most recent benchmark for all models
    idx = dat.groupby(['inferrer', 'top_k'])['timestamp'].idxmax()
    bar_dat = dat.loc[idx].pivot(index='inferrer', columns='top_k', values='value')
    bar_dat.plot(kind='bar', ax=ax, rot=0).legend(loc=3)
    _label_bars(ax)
    ylim = ax.get_ylim()
    if ylim[0] < 0:
        ax.axhline(0, 0, 1, color='black', lw=0.5)
    ax.legend().set_zorder(-1)
    ax.legend().set_title('Top K')
    ax.set(title='Embedding: \nNearest Neighbor', ylabel="Speaman's Rho", xlabel='Model (Latest Benchmark)')


def make_latest_physchem_plots(embedding_df, axlist, max_plot_ratio=10):
    """Plots of physchem property results"""

    dat = embedding_df[embedding_df.name == 'physchem']
    dat['property'] = dat['property'].map(lambda x: PHYSCHEM_UNIT_RENAMER[x])
    dat = dat[['timestamp', 'inferrer', 'property', 'model', 'value']]

    grouper = dat.groupby('property')
    for ax, (property, dat_) in zip(axlist, grouper):
        # Latest values plot
        last_timestep = dat_.sort_values('timestamp').groupby(['inferrer', 'model']).last().reset_index()
        last_timestep = last_timestep.pivot(columns=['model'], values='value', index='inferrer')
        is_first_col = True if ax is axlist[0] else False
        _ = last_timestep.plot(kind='bar', width=0.8, legend=is_first_col, ax=ax, rot=0)

        ax.axhline(1.0, 0, 1, color='red', lw=1.0, zorder=-1)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize='small')

        # ymax = min(ax.get_ylim()[0], max_plot_ratio)
        # ax.set_ylim(0, ymax)
        # _label_bars(ax, int(0.9 * ymax))
        ax.set_ylim(0, max_plot_ratio)
        _label_bars(ax, int(0.9 * max_plot_ratio))

        ax.set(title=f'Embedding, Physchem: \n{property}', ylabel='MSE Ratio', xlabel='Model (Latest Benchmark)')
        if is_first_col:
            ax.legend().set_title('Model')


def make_latest_bioactivity_plots(embedding_df, axlist, max_plot_ratio=10):
    """Plots of bioactivity results"""

    dat = embedding_df[embedding_df.name == 'bioactivity']
    dat = dat[['timestamp', 'inferrer', 'gene', 'model', 'value']]
    dat.sort_values('gene', inplace=True)

    gene_list = sorted(dat['gene'].unique())
    n_genes = len(gene_list)
    gene_labels = pd.CategoricalDtype(gene_list, ordered=True)
    dat['gene'] = dat['gene'].astype(gene_labels)

    grouper = dat.groupby('inferrer')
    for ax, (inferrer, dat) in zip(axlist, grouper):
        is_first_row = True if ax is axlist[0] else False

        last_timestep = dat.sort_values('timestamp').groupby(['inferrer', 'gene', 'model']).last().reset_index()
        last_timestep = last_timestep.pivot(columns=['model'], values='value', index='gene')
        _ = last_timestep.plot(kind='line', marker='o', legend=is_first_row, ax=ax, rot=70)

        ax.axhline(1.0, 0, 1, color='red', lw=1.0, zorder=-1)
        if is_first_row:
            ax.set(title='Embedding, Bioactivity')
        ax.set(xlabel='Gene', ylabel=f'{inferrer}\n MSE Ratio')
        ax.set_xlim(-0.1, n_genes + 0.1)
        ax.set_xticks(range(0, n_genes))
        ax.set_xticklabels(gene_list, fontsize='8', rotation=70)

        # ymax = min(ax.get_ylim()[0], max_plot_ratio)
        # ax.set_ylim(0, ymax)
        ax.set_ylim(0, max_plot_ratio)

        if is_first_row:
            ax.legend().set_title('Model')


def create_latest_aggregated_plots(output_dir, remove_elastic_net=True):
    """Create all aggregated plots for sampling and embedding metrics"""

    metric_df = load_aggregated_metric_results(output_dir)
    embedding_df = make_aggregated_embedding_df(metric_df, models=['linear_regression', 'support_vector_machine', 'random_forest'])

    sns.set_palette('dark')
    pal = sns.color_palette()
    sns.set_palette([pal[0]] + pal[2:])
    sns.set_style('whitegrid', {'axes.edgecolor': 'black', 'axes.linewidth': 1.5})

    ncols, nrows = 7, 3
    fig = plt.figure(figsize=(ncols*4, nrows*4))
    axlist0 = [plt.subplot2grid((nrows, ncols), (0, x)) for x in range(0, 3)]
    axlist1 = [plt.subplot2grid((nrows, ncols), (0, 3))]
    axlist2 = [plt.subplot2grid((nrows, ncols), (0, x)) for x in range(4, 7)]
    axlist3 = [plt.subplot2grid((nrows, ncols), (y, 0), colspan=ncols) for y in range(1, 3)]

    make_latest_sampling_plots(metric_df, axlist0)
    make_latest_nearest_neighbor_plot(embedding_df, axlist1)
    make_latest_physchem_plots(embedding_df, axlist2)
    make_latest_bioactivity_plots(embedding_df, axlist3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Latest_Benchmark_Metrics.png'), dpi=300)
    return