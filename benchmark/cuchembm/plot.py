import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os
import sys
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates


def load_metric_results(output_dir):
    """Load metric results from CSV files"""
    custom_date_parser = lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S")
    file_list = glob.glob(os.path.join(output_dir, '*.csv'))
    metric_df = list()

    for file in file_list:
        df = pd.read_csv(file, parse_dates=['timestamp'], date_parser=custom_date_parser)
        metric_df.append(df)

    metric_df = pd.concat(metric_df, axis=0).reset_index(drop=True)
    metric_df['name'] = metric_df['name'].str.replace('modelability-', '')
    return metric_df


def make_sampling_plots(metric_df, output_dir):
    """Using an input dataframe of all metrics, create sampling plots"""
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

        if n_timestamps > 1: 
            # timeseries plot if multiple dates are present
            (dat.pivot_table(columns=['inferrer', 'radius'], 
                                 values='value', 
                                 index='timestamp', 
                                 aggfunc='mean')
                .plot(kind='line', marker='o', ax=ax))
            date_form = mdates.DateFormatter("%Y/%m/%d")
            ax.xaxis.set_major_formatter(date_form)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        else:
            # bar plot if single date is present
            (dat.pivot_table(columns='radius', 
                                 values='value', 
                                 index='inferrer', 
                                 aggfunc='mean')
                .plot(kind='bar', ax=ax))

        ax.set(title=metric.title(), ylabel='Percentage', xlabel='Date')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'Sampling_Metrics_Benchmark.png'), dpi=300)


def make_embedding_df(metric_df):
    """Select embedding metrics from metric dataframe"""
    embedding_mask = metric_df['name'].isin(['validity', 'unique', 'novelty']).pipe(np.invert)
    embedding_df = metric_df[embedding_mask]

    cat = pd.CategoricalDtype(['nearest neighbor correlation', 'physchem', 'bioactivity'], ordered=True)
    embedding_df['name'] = embedding_df['name'].astype(cat)
    cat = pd.CategoricalDtype(['linear_regression', 'elastic_net', 'support_vector_machine', 'random_forest'], ordered=True)
    embedding_df['model'] = embedding_df['model'].astype(cat)
    return embedding_df


def make_nearest_neighbor_plot(embedding_df, output_dir):
    dat = embedding_df[embedding_df.name == 'nearest neighbor correlation']
    d = dat[['timestamp', 'inferrer', 'top_k', 'value']].drop_duplicates()
    n_timestamps = dat['timestamp'].nunique()

    if n_timestamps > 1:
        # timeseries
        ax = (d.pivot(columns=['inferrer', 'top_k'], 
                      values='value', 
                      index='timestamp')
                .plot(kind='line', marker='o'))
        ax.set(title='Nearest Neighbor Metric', ylabel='Percentage', xlabel='Date')
    else:
        # barplot of single timepoint
        ax =  (d.pivot(index='inferrer', columns='top_k', values='value').plot(kind='bar'))
        ax.set(title='Nearest Neighbor Metric', ylabel='Percentage', xlabel='Groups')

    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(output_dir, 'Nearest_Neighbor_Benchmark.png'), dpi=300)


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
    fig.savefig(os.path.join(output_dir, 'Physchem_Benchmark.png'), dpi=300)


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
    fig.savefig(os.path.join(output_dir, 'Bioactivity_Benchmark.png'), dpi=300)

