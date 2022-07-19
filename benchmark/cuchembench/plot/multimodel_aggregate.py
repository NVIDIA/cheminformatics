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


def prune_megamolbart_radii(dat):
    mask = dat['inferrer'].str.contains('MegaMolBART')
    dat = dat[mask]

    filter_func = lambda x: (x['radius'] >= MEGAMOLBART_SAMPLE_RADIUS[x['model_size']][0]) & (x['radius'] <= MEGAMOLBART_SAMPLE_RADIUS[x['model_size']][1])
    model_size_mask = dat.apply(filter_func, axis=1)
    dat = dat.loc[model_size_mask]
    return dat


def prune_models(dat):
    mask = dat['inferrer'].str.contains('MegaMolBART').pipe(np.invert)
    other_dat = dat[mask]

    megamolbart_dat = prune_megamolbart_radii(dat)
    return pd.concat([megamolbart_dat, other_dat], axis=0).reset_index(drop=True)


def make_multimodel_sampling_plots(metric_df, output_dir, plot_type='model', prune_radii=True, model_sort_order=None):
    """Make aggregate plots for validity, uniqueness, novelty --
       will be bar chart for single date or timeseries for multiple"""
    
    ncols, nrows = 3, 1
    fig, axlist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*8, nrows*6))
    axlist = axlist.flatten()
    
    generative_mask = metric_df['name'].isin(['validity', 'unique', 'novelty'])
    generative_df = metric_df[generative_mask].dropna(axis=1, how='all')

    # Set sort orders by using a categorical
    cat = pd.CategoricalDtype(['validity', 'novelty', 'unique'], ordered=True)
    generative_df['name'] = generative_df['name'].astype(cat)
    
    if model_sort_order:
        model_cat = pd.CategoricalDtype(model_sort_order, ordered=True)
        generative_df['inferrer'] = generative_df['inferrer'].astype(model_cat)

    grouper = generative_df.groupby('name')
    timestamp_lim = (metric_df['timestamp'].min(), metric_df['timestamp'].max())

    for ax, (metric, dat) in zip(axlist, grouper):
        if not isinstance(dat, pd.DataFrame):
            dat = dat.to_frame()

        is_first_col = True if ax is axlist[0] else False

        if metric in ACCEPTANCE_CRITERIA:
            ax.axhline(y=ACCEPTANCE_CRITERIA[metric], xmin=0, xmax=1, color='red', lw=1.0, zorder=-1)

        if plot_type == 'model':
            # Barplot of each model
            #if prune_radii:
            #    dat = prune_models(dat)
            is_first_col = False

            if dat.groupby(['inferrer']).size().max() == 1:
                dat = dat[['inferrer', 'value']].set_index('inferrer').sort_index()
            else:
                dat = dat.pivot(index='inferrer', columns='radius', values='value')

            dat.plot(kind='bar', ax=ax, legend=is_first_col, width=0.8, rot=10)
            _label_bars(ax)
            xlabel = 'Model'
            
        elif plot_type == 'timeseries':
            # Timeseries -- line plot data
            dat = prune_megamolbart_radii(dat)
            
            dat = dat.pivot(index='timestamp', columns='inferrer', values='value')
            dat.plot(kind='line', marker='o', ax=ax, legend=is_first_col)
            # ax.set_xlim(*timestamp_lim) # TODO FIX ME
            xlabel = 'Date (Timeseries)'

        ax.set_ylim(0, 1.1)
        ax.set(title=f'Sampling: \n{metric.title()}', xlabel=xlabel, ylabel='Ratio')
        #plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'Sampling_Aggregated_Benchmark.png'), dpi=300)

        return


def make_multimodel_nearest_neighbor_plot(embedding_df, output_dir, plot_type='model', model_sort_order=None):
    """Aggregate plot for nearest neighbor correlation ---
       bar chart for single time point, time series for multiple"""
    # TODO add option to select top-k
    
    ncols, nrows = 1, 1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*6, nrows*6))
    ax.axhline(0.0, 0, 1, color='black', lw=1.0, zorder=-1)

    dat = embedding_df[embedding_df.name == 'nearest neighbor correlation']
    dat = dat[['timestamp', 'inferrer', 'top_k', 'value']].drop_duplicates()
    dat['top_k'] = dat['top_k'].astype(int)
    
    if model_sort_order:
        model_cat = pd.CategoricalDtype(model_sort_order, ordered=True)
        dat['inferrer'] = dat['inferrer'].astype(model_cat)
        dat.sort_values('inferrer', inplace=True)
        
    timestamp_lim = (dat['timestamp'].min() - 1, dat['timestamp'].max() + 1)

    if plot_type == 'model':
        # Barplot of each model
        if dat.groupby(['inferrer']).size().max() == 1:
            dat = dat[['inferrer', 'value']].set_index('inferrer').sort_index()
        else:
            dat = dat.pivot(index='inferrer', columns='top_k', values='value')
            
        dat.plot(kind='bar', ax=ax, legend=True, width=0.8, rot=0)
        _label_bars(ax)
        xlabel = 'Model'
        
    elif plot_type == 'timeseries':
        # Timeseries -- line plot data
        dat = dat[dat['inferrer'].str.contains('MegaMolBART')]
        dat = dat.pivot(columns=['inferrer', 'top_k'], values='value', index='timestamp')
        dat.plot(kind='line', marker='o', ax=ax, legend=True)
        # ax.set_xlim(*timestamp_lim)
        xlabel = 'Date (Timeseries)'

    # ax.set_ylim(*ylim)
    ax.set(title='Nearest Neighbor Metric', ylabel="Speaman's Rho", xlabel=xlabel)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Nearest_Neighbor_Aggregated_Benchmark.png'), dpi=300)


def make_multimodel_physchem_plots(embedding_df, output_dir, plot_type='model', filter_genes=False, model_sort_order=None, max_plot_ratio=10):
    # TODO fix missing RF from first model, timeseries is averaging the properties
    """Plots of physchem property results"""

    dat = embedding_df[embedding_df.name == 'physchem']
    assert dat.groupby(['inferrer', 'property', 'model']).size().max() == 1, AssertionError('Duplicate physchem benchmark entries present.')
    
    dat['property'] = dat['property'].map(lambda x: PHYSCHEM_UNIT_RENAMER.get(x, x))
    dat = dat[['timestamp', 'inferrer', 'property', 'model', 'value']]
    
    if model_sort_order:
        model_cat = pd.CategoricalDtype(model_sort_order, ordered=True)
        dat['inferrer'] = dat['inferrer'].astype(model_cat)
        
    timestamp_lim = (dat['timestamp'].min() - 1, dat['timestamp'].max() + 1)

    grouper = dat.groupby('property')
    n_properties = len(grouper)
    n_rows = int((n_properties / 2) + 0.5)
    fig, axes = plt.subplots(ncols=2, nrows=n_rows, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for ax in axes[n_properties:]:
        ax.set_axis_off()

    for idx, (property_, dat_) in enumerate(grouper):
        ax = axes[idx]
        is_first_row = ax.get_subplotspec().is_first_row()
        
        if plot_type == 'model':
            # Model values plot
            dat_ = dat_.pivot(columns=['model'], values='value', index='inferrer')
            _ = dat_.plot(kind='bar', width=0.8, legend=False, ax=ax, rot=0)

            ax.set_xticklabels(ax.get_xticklabels(), fontsize='small')            
            _label_bars(ax, 0.9 * max_plot_ratio)
            
            title = 'Physchem Property Prediction (Model Benchmarks)'
            ylabel = f'{property_}\nMSE Ratio'
            xlabel = 'Model'

        elif plot_type == 'timeseries':
            # Timeseries -- line plot data
            dat_ = dat_.pivot_table(columns=['model'], values='value', index='timestamp', aggfunc='mean')
            _ = dat_.plot(kind='line', marker='o', legend=False, ax=ax, rot=0)
            
            ax.set_xlim(*timestamp_lim)
            title = 'Physchem Property Prediction (Mean of All Properties as Timeseries)'
            ylabel = f'Average MSE Ratio (All Properties)'
            xlabel = 'Timestamp'
            
        ax.axhline(1.0, 0, 1, color='red', lw=1.0, zorder=-1)
        ax.set_ylim(0, max_plot_ratio)
        ax.set(xlabel=xlabel, ylabel=ylabel)
            
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()

    fig = plt.gcf()
    fig.legend(handles=handles, labels=labels, loc=1)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Physchem_Aggregated_Benchmark.png'), dpi=300)


def make_multimodel_bioactivity_plots(embedding_df, output_dir, filter_genes=False, plot_groupby='inferrer', plot_type='model', model_sort_order=None, max_plot_ratio=2):
    """Plots of bioactivity property results"""
    # FIX MODEL SORT ORDER
    dat = embedding_df[embedding_df.name == 'bioactivity']
    #assert dat.groupby(['inferrer', 'model', 'gene']).size().max() == 1, AssertionError('Duplicate bioactivity benchmark entries present.')
    
    dat = dat[['timestamp', 'inferrer', 'gene', 'model', 'value']]
    if model_sort_order:
        model_cat = pd.CategoricalDtype(model_sort_order, ordered=True)
        dat['inferrer'] = dat['inferrer'].astype(model_cat)

    if filter_genes:
        gene_counts = dat[['inferrer', 'gene', 'model']].drop_duplicates().groupby(['gene','inferrer']).size().reset_index()
        gene_counts = gene_counts.pivot_table(columns='inferrer', index='gene', values=0, fill_value=0).min(axis=1)
        gene_list = gene_counts[gene_counts > 0].sort_index()
        gene_list = gene_list.index.to_list()
    else:
        gene_list = sorted(dat['gene'].unique())
        
    n_genes = len(gene_list)
    dat = dat[dat.gene.isin(gene_list)].sort_values('gene')
    gene_labels = pd.CategoricalDtype(gene_list, ordered=True)
    dat['gene'] = dat['gene'].astype(gene_labels)

    timestamp_lim = (dat['timestamp'].min() - 1, dat['timestamp'].max() + 1)

    grouper = dat.groupby(plot_groupby)
    n_plots = len(grouper)
    n_cols = 4
    n_rows = int((n_plots / n_cols) + 0.5)
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()

    for idx, (plot_grouper_value, dat) in enumerate(grouper):
        ax = axes[idx]
        ax.axhline(1.0, 0, 1, color='red', lw=1.0, zorder=-1)

        if plot_type == 'model':
            dat = dat.sort_values('timestamp').groupby(['inferrer', 'gene', 'model']).last().reset_index()
            
            if plot_groupby == 'inferrer':
                dat = dat.pivot_table(columns=['model'], values='value', index='gene', aggfunc='mean')
                #dat = dat.pivot_table(columns=['gene'], values='value', index='model', aggfunc='mean')
                _ = dat.plot(kind='line', marker='o', legend=False, ax=ax, rot=70)
                
                ax.set_xlim(-0.1, n_genes + 0.1)
                ax.xaxis.set_tick_params(labelbottom=True)
                ax.set_xticks(range(0, n_genes))
                ax.set_xticklabels(gene_list, fontsize='7', rotation=70)
                xlabel = 'Gene'
                
            elif plot_groupby == 'gene':
                dat = dat.pivot_table(columns=['model'], values='value', index='inferrer', aggfunc='mean')
                _ = dat.plot(kind='bar', width=0.8, legend=False, ax=ax, rot=0)
            
                ax.set_xticklabels(ax.get_xticklabels(), fontsize='8', rotation=30)
                _label_bars(ax, 0.9 * max_plot_ratio)
                xlabel = 'Model'
            
            title = 'Bioactivity Prediction\n'

        elif plot_type == 'timeseries':
            # Timeseries plot
            dat = dat.pivot_table(columns=['model'], values='value', index='timestamp', aggfunc='mean')
            _ = dat.plot(kind='line', marker='o', legend=False, ax=ax, rot=0)
            
            # ax.set_xlim(*timestamp_lim) # TODO fix this bug
            title = 'Bioactivity Timeseries\n(Mean over all Genes)'
            xlabel='Timestamp', 

        ax.set_ylim(0, max_plot_ratio)
        ax.set(xlabel=xlabel, ylabel=f'{plot_grouper_value}\nMSE Ratio')
        
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles=handles, labels=labels, loc=1)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Bioactivity_Aggregated_Benchmark.png'), dpi=300)


def create_multimodel_aggregated_plots(metric_paths, metric_labels, plot_dir):
    """Create all aggregated plots for sampling and embedding metrics"""
    metric_df = load_aggregated_metric_results(metric_paths=metric_paths, metric_labels=metric_labels, homogenize_timestamp=True)
    embedding_df = make_aggregated_embedding_df(metric_df)

    sns.set_palette('dark')
    pal = sns.color_palette()
    sns.set_palette([pal[0]] + pal[2:])
    sns.set_style('whitegrid', {'axes.edgecolor': 'black', 'axes.linewidth': 1.5})
    
    model_sort_order = ['CDDD', 'MegaMolBART-v0.1', 'MegaMolBART-NeMo1.2', 'MegaMolBART-NeMo1.10'] # TODO add to yaml config? some other way to control sort order
    plot_type = 'model' # TODO add to yaml plot config

    simple_metric_df = metric_df.query("(inferrer == 'MegaMolBART-v0.1' and radius == 0.1) or (inferrer == 'MegaMolBART-NeMo1.2' and radius == 0.75) or (inferrer == 'MegaMolBART-NeMo1.10' and radius == 0.5)")
    make_multimodel_sampling_plots(simple_metric_df, output_dir=plot_dir, plot_type=plot_type, prune_radii=False, model_sort_order=model_sort_order)
    make_multimodel_nearest_neighbor_plot(embedding_df, plot_dir, plot_type=plot_type, model_sort_order=model_sort_order)

    make_multimodel_physchem_plots(embedding_df, plot_dir, plot_type=plot_type, model_sort_order=model_sort_order, max_plot_ratio=6)
    
    plot_groupby = 'inferrer' # inferrer or gene
    filter_genes = True
    make_multimodel_bioactivity_plots(embedding_df, plot_dir, filter_genes=filter_genes, plot_groupby=plot_groupby, model_sort_order=False, plot_type=plot_type, max_plot_ratio=1.5)

    return