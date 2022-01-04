import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from .data import load_aggregated_metric_results, make_aggregated_embedding_df

__ALL__ = ['create_aggregated_plots']

def make_sampling_plots(metric_df, output_dir):
    """Make aggregate plots for validity, uniqueness, novelty --
       will be bar chart for single date or timeseries for multiple"""

    # Select data
    generative_mask = metric_df['name'].isin(['validity', 'unique', 'novelty'])
    generative_df = metric_df[generative_mask].dropna(axis=1, how='all')

    # Set sort order by using a categorical
    cat = pd.CategoricalDtype(['validity', 'novelty', 'unique'], ordered=True)
    generative_df['name'] = generative_df['name'].astype(cat)

    grouper = generative_df.groupby('name')
    n_plots = len(grouper)
    fig, axes = plt.subplots(ncols=n_plots, figsize=(n_plots*4, 4))
    axes = axes.flatten()

    for (metric, dat), ax in zip(grouper, axes):
        if not isinstance(dat, pd.DataFrame):
            dat = dat.to_frame()
            
        n_timestamps = dat['timestamp'].nunique()
        show_legend = True if metric == 'validity' else False

        if n_timestamps > 1: 
            # timeseries plot if multiple dates are present
            (dat.pivot_table(columns=['inferrer', 'radius'], 
                                 values='value', 
                                 index='timestamp', 
                                 aggfunc='mean')
                .plot(kind='line', marker='o', ax=ax, legend=show_legend))
            date_form = mdates.DateFormatter("%Y/%m/%d")
            ax.xaxis.set_major_formatter(date_form)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.set_xlabel('Date')
        else:
            # bar plot if single date is present
            (dat.pivot_table(columns='radius', 
                                 values='value', 
                                 index='inferrer', 
                                 aggfunc='mean')
                .plot(kind='bar', ax=ax, rot=0, legend=show_legend))
            ax.set_xlabel('Model')

        ax.set(title=metric.title(), ylabel='Percentage')
        for p in ax.patches:
            value = p.get_height()
            ax.annotate("{:.2f}".format(value), (p.get_x() * 1.005, value * 1.005))
            
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'Sampling_Metrics_Aggregated_Benchmark.png'), dpi=300)


def make_nearest_neighbor_plot(embedding_df, output_dir):
    """Aggregate plot for nearest neighbor correlation ---
       bar chart for single time point, time series for multiple"""

    dat = embedding_df[embedding_df.name == 'nearest neighbor correlation']
    d = dat[['timestamp', 'inferrer', 'top_k', 'value']].drop_duplicates()
    n_timestamps = dat['timestamp'].nunique()

    if n_timestamps > 1:
        # timeseries
        ax = (d.pivot(columns=['inferrer', 'top_k'], 
                      values='value', 
                      index='timestamp')
                .plot(kind='line', marker='o'))
        ax.set(title='Nearest Neighbor Metric', ylabel="Speaman's Rho", xlabel='Date')
    else:
        # barplot of single timepoint
        ax =  (d.pivot(index='inferrer', columns='top_k', values='value').plot(kind='bar'))
        ax.set(title='Nearest Neighbor Metric', ylabel="Speaman's Rho", xlabel='Groups')

    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(output_dir, 'Nearest_Neighbor_Aggregated_Benchmark.png'), dpi=300)


def make_physchem_plots(embedding_df, output_dir):
    """Plots of phychem property results"""
    # TODO convert xaxis label from units to property
    dat = embedding_df[embedding_df.name == 'physchem']
    d = dat[['timestamp', 'inferrer', 'property', 'model', 'value']].drop_duplicates()

    grouper = d.groupby('inferrer')
    n_models = len(grouper)
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 4*n_models))

    for row, (inferrer,dat) in enumerate(grouper):
        # Timeseries plot
        ax = axes[row, 0]
        _ = dat.pivot_table(columns=['model'], values='value', index='timestamp', aggfunc='mean').plot(kind='line', marker='o', legend=False, ax=ax, rot=0)
        
        ax.set_title('Physchem Property Prediction (Mean of All Properties as Timeseries)') if ax.is_first_row() else ax.set_title('')
        ax.set_ylabel(f'{inferrer}\nMSE Ratio') if ax.is_first_col() else ax.set_ylabel('')
        ax.set_xlabel('Timestamp') if ax.is_last_row() else ax.set_xlabel('')
        
        # Latest values plot
        ax = axes[row, 1]
        last_timestep = dat.sort_values('timestamp').groupby(['inferrer', 'property', 'model']).last().reset_index()
        _ = last_timestep.pivot(columns=['model'], values='value', index='property').plot(kind='bar', width=0.8, legend=False, ax=ax, rot=0)
        ax.set_ylim(0,50)
        
        ax.set_title('Physchem Property Prediction (Most Recent Benchmark)') if ax.is_first_row() else ax.set_title('')
        ax.set_ylabel(f'{inferrer}\nMSE Ratio') if ax.is_first_col() else ax.set_ylabel('')
        ax.set_xlabel('Property') if ax.is_last_row() else ax.set_xlabel('')
        
    fig = plt.gcf()
    fig.legend(loc=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Physchem_Aggregated_Benchmark.png'), dpi=300)


def make_bioactivity_plots(embedding_df, output_dir):
    dat = embedding_df[embedding_df.name == 'bioactivity']
    d = dat[['timestamp', 'inferrer', 'gene', 'model', 'value']].drop_duplicates()

    grouper = d.groupby('inferrer')
    n_models = len(grouper)
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 4*n_models))
    # labels = sorted(d['gene'].unique())

    for row, (inferrer,dat) in enumerate(grouper):
        # Timeseries plot
        ax = axes[row, 0]
        _ = dat.pivot_table(columns=['model'], values='value', index='timestamp', aggfunc='mean').plot(kind='line', marker='o', legend=False, ax=ax, rot=0)
        ax.set_title('Bioactivity Metrics Timeseries (Mean over all Genes)') if ax.is_first_row() else ax.set_title('')
        ax.set_ylabel(f'{inferrer}\nMSE Ratio') if ax.is_first_col() else ax.set_ylabel('')
        ax.set_xlabel('Timestamp') if ax.is_last_row() else ax.set_xlabel('')
        
        ax = axes[row, 1]
        last_timestep = dat.sort_values('timestamp').groupby(['inferrer', 'gene', 'model']).last().reset_index()
        _ = last_timestep.pivot(columns=['model'], values='value', index='gene').plot(kind='bar', width=0.8, legend=True, ax=ax, rot=70)
        # ax.set_xticklabels(labels) # TODO figure out how to align genes if they're different 
        
        ax.set_title('Bioactivity Metrics (Latest)') if ax.is_first_row() else ax.set_title('')
        ax.set_ylabel(f'{inferrer}\nMSE Ratio') if ax.is_first_col() else ax.set_ylabel('')
        ax.set_xlabel('Gene') if ax.is_last_row() else ax.set_xlabel('')
    
    fig = plt.gcf()
    fig.legend(loc=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Bioactivity_Aggregated_Benchmark.png'), dpi=300)


def create_aggregated_plots(output_dir):
    """Create all aggregated plots for sampling and embedding metrics"""
    metric_df = load_aggregated_metric_results(output_dir)
    make_sampling_plots(metric_df, output_dir)

    embedding_df = make_aggregated_embedding_df(metric_df)
    make_nearest_neighbor_plot(embedding_df, output_dir)
    make_physchem_plots(embedding_df, output_dir)
    make_bioactivity_plots(embedding_df, output_dir)
    return