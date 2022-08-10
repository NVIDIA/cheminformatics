import pandas as pd
from typing import Optional, List, Union
import os
import matplotlib.pyplot as plt
from .utils import set_plotting_style, setup_plot_grid, label_bars

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

__ALL__ = ['make_multimodel_similarity_search_plot', 'make_multimodel_physchem_plots', 'make_multimodel_bioactivity_plots']


def _extract_physchem_name(dataset_names: List[str]):
    """Helper function for extracting physchem dataset names"""
    dataset_converter = {'esol':'ESOL', 
                         'freesolv':'FreeSolv', 
                         'lipophilicity':'Lipophilicity'}
    
    name = None
    for name in dataset_names:
        if '_' in name:
            key = name.split('_')[-1]
            name = dataset_converter.get(key, None)
    return name


def make_multimodel_similarity_search_plot(df: pd.DataFrame, 
                                           save_plots: bool = False, 
                                           reports_dir: Optional[str] = '.'):
    """Make plot for similarity search

    Args:
        df (pd.DataFrame): Dataframe of input CSV data
        save_plots (bool, optional): Save plots to disk. Defaults to False.
        reports_dir (Optional[str], optional): Directory. Defaults to current directory.
    """
    
    # Other config -- TODO consider making configurable
    kpis = ['nearest neighbor correlation', 'nearest_neighbor_correlation']
    kpi_field = 'name'
    comparison_field = 'exp_name'
    value_field = 'value'
    top_k_field = 'top_k'
    
    # Get selected data and set sort order
    exp_sort_order = pd.CategoricalDtype(df[comparison_field].unique(), ordered=True)
    query = f'({kpi_field} in {kpis})'
    df = df.query(query)[[kpi_field, comparison_field, value_field, top_k_field]]
    df.loc[:, comparison_field] = df[comparison_field].astype(exp_sort_order)

    # Setup plots
    set_plotting_style()
    num_plots = 1
    plots_per_row = 1
    fig, axes_list = setup_plot_grid(num_plots, plots_per_row)
    ax = axes_list[0]
    
    contains_duplicates = df.groupby([top_k_field, comparison_field]).size().max() > 1
    if contains_duplicates:
        logging.info(f'Dataframe contains duplicate values. They will be averaged.')

    plot_df = df.pivot_table(columns=[top_k_field], values=value_field, index=comparison_field, aggfunc='mean')
    plot_df.plot(kind='bar', width=0.8, ax=ax).legend(loc=3)
    label_bars(ax)

    #ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='x', rotation=25)
    ax.set_title(f'Similarity Search')
    ax.set_xlabel('')
    ax.set_ylabel(f"Spearman's Rho")

    fig.subplots_adjust(left=0.01,
               bottom=0.01, 
               right=0.99, 
               top=0.90, 
               wspace=0.1, 
               hspace=0.2)
    
    if save_plots:
        plt.tight_layout()
        save_path = os.path.join(reports_dir, 'Similarity_Search_Aggregated_Benchmark.png')
        fig.savefig(save_path, dpi=300)


def make_multimodel_physchem_plots(df: pd.DataFrame, 
                                   save_plots: bool = False, 
                                   max_plot_ratio: float = 10.0,
                                   plots_per_row: int = 2,
                                   reports_dir: Optional[str] = '.'):
    """Make plots for physchem metrics

    Args:
        df (pd.DataFrame): Dataframe of input CSV data
        save_plots (bool, optional): Save plots to disk. Defaults to False.
        max_plot_ratio (float, optional): Max limit for y-axis. Defaults to 10.
        plots_per_row (int, optional): Number of plots per row. Defaults to 2.
        reports_dir (Optional[str], optional): Directory. Defaults to current directory.
    """

    # Other config -- TODO consider making configurable
    kpis = ['physchem', 'physchem_esol', 'physchem_freesolv', 'physchem_lipophilicity']
    models = ['linear_regression', 'support_vector_machine', 'random_forest']
    kpi_field = 'name'
    model_field = 'model'
    comparison_field = 'exp_name'
    value_field = 'value'
    property_field = 'property'
    
    # Get selected data and set sort order
    exp_sort_order = pd.CategoricalDtype(df[comparison_field].unique(), ordered=True)
    query = f'({kpi_field} in {kpis}) & ({model_field} in {models})'
    df = df.query(query)[[kpi_field, comparison_field, value_field, property_field, model_field]]
    df.loc[:, comparison_field] = df[comparison_field].astype(exp_sort_order)
    
    logging.info(f'{df.groupby([comparison_field, model_field])[value_field].mean()}')

    # Setup plots
    set_plotting_style()
    num_plots = len(df[property_field].unique())
    fig, axes_list = setup_plot_grid(num_plots, plots_per_row)
    
    for i, (property_, plot_df) in enumerate(df.groupby(property_field, sort=False)):        
        ax = axes_list[i]
        
        contains_duplicates = plot_df.groupby([model_field, comparison_field]).size().max() > 1
        if contains_duplicates:
            logging.info(f'Dataframe for {property_} contains duplicate values. They will be averaged.')
        
        dataset_name = plot_df['name'].unique()
        dataset_name = _extract_physchem_name(dataset_name)
        
        plot_df = plot_df.pivot_table(columns=[model_field], values=value_field, index=comparison_field, aggfunc='mean')
        legend = True if i == 1 else False
        plot_df.plot(kind='bar', width=0.8, ax=ax, legend=legend)
        
        ax.axhline(1.0, 0, 1, color='red', lw=2.0, zorder=-1)
        
        ymin, ymax = ax.get_ylim()
        ymax = min(ymax, max_plot_ratio) * 1.1
        label_bars(ax, 0.9 * ymax)
        
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis='x', rotation=15)
        
        if dataset_name:
            title_str = f'Physchem: {dataset_name}'
        else:
            units = property_.replace('_', ' ')
            title_str = f'Physchem: {units}'
        
        ax.set_title(title_str)
        ax.set_xlabel('')
        ax.set_ylabel(f'MSE Ratio')
        
        fig.subplots_adjust(left=0.01,
                    bottom=0.01, 
                    right=0.99, 
                    top=0.90, 
                    wspace=0.1, 
                    hspace=0.2)
    if save_plots:
        plt.tight_layout()
        save_path = os.path.join(reports_dir, 'Physchem_Aggregated_Benchmark.png')
        fig.savefig(save_path, dpi=300)


def make_multimodel_bioactivity_plots(df: pd.DataFrame, 
                                       save_plots: bool = False, 
                                       max_plot_ratio: float = 5.0,
                                       plot_type: str = 'line',
                                       plots_per_row: int = 1,
                                       limit_genes: Optional[Union[List[str], int]] = None,
                                       reports_dir: Optional[str] = '.'):
    """Make plots for bioactivity metrics

    Args:
        df (pd.DataFrame): Dataframe of input CSV data
        save_plots (bool, optional): Save plots to disk. Defaults to False.
        max_plot_ratio (float, optional): Max limit for y-axis. Defaults to 5.0.
        plot_type (str): type of plot to make ('line' or 'bar). Defaults to line plot.
        plots_per_row (int, optional): Number of plots per row. Defaults to 2.
        limit_genes (list of strings or int): Limit the genes plotted to a list of specific genes
            or the first N (integer) number of genes in alphabetical order
        reports_dir (Optional[str], optional): Directory. Defaults to current directory.
    """

    # Other config -- TODO consider making configurable
    kpis = ['bioactivity']
    models = ['linear_regression', 'support_vector_machine', 'random_forest']
    kpi_field = 'name'
    model_field = 'model'
    comparison_field = 'exp_name'
    value_field = 'value'
    property_field = 'gene'
    
    # Get selected data and set sort order
    exp_sort_order = pd.CategoricalDtype(df[comparison_field].unique(), ordered=True)
    query = f'({kpi_field} in {kpis}) & ({model_field} in {models})'
    
    gene_list = sorted(df[property_field].dropna().unique())
    if limit_genes:
        limit_index = gene_list.index(limit_genes) if isinstance(limit_genes, str) else limit_genes
        gene_list = gene_list[: limit_index+1]
        logging.info(f'The genes being plotted have been limited to {gene_list}')
        query += f' & ({property_field} in {gene_list})'
    
    df = df.query(query)[[kpi_field, comparison_field, value_field, property_field, model_field]]
    df.loc[:, comparison_field] = df[comparison_field].astype(exp_sort_order)
    n_genes = len(gene_list)
    
    # Setup plots
    set_plotting_style(show_grid=True)
    num_plots = len(models)
    
    fig, axes_list = setup_plot_grid(num_plots, plots_per_row, xscale=14)
    
    for i, (model_, plot_df) in enumerate(df.groupby(model_field, sort=False)):
        ax = axes_list[i]
        
        contains_duplicates = plot_df.groupby([comparison_field, property_field]).size().max() > 1
        if contains_duplicates:
            logging.info(f'Dataframe for {model_} contains duplicate values. They will be averaged.')
        
        logging.info(f'{model_}')
        logging.info(f'{plot_df.groupby([comparison_field])[value_field].mean()}')
        plot_df = plot_df.pivot_table(columns=[comparison_field], values=value_field, index=property_field, aggfunc='mean')
        legend = True if i == 0 else False
        kwargs = {'kind':'line', 'marker':'o'} if plot_type == 'line' else {'kind':'bar', 'width':0.8}
        plot_df.plot(ax=ax, legend=legend, **kwargs)
        
        ax.axhline(1.0, 0, 1, color='red', lw=2.0, zorder=-1)
        
        ymin, ymax = ax.get_ylim()
        ymax = min(ymax, max_plot_ratio) * 1.05
        label_bars(ax, 0.9 * ymax)
        
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(f'MSE Ratio')
        
        ax.set_xlim(-0.5, n_genes - 0.5)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xticks(range(0, n_genes))
        ax.set_xticklabels(gene_list, rotation=45, size='7')
        ax.set_xlabel('')
        
        model_title = model_.replace('_', ' ').title()
        ax.set_title(f'Bioactivity: {model_title}')
        
        fig.subplots_adjust(left=0.01,
                    bottom=0.01, 
                    right=0.99, 
                    top=0.90, 
                    wspace=0.1, 
                    hspace=0.2)
    if save_plots:
        plt.tight_layout()
        save_path = os.path.join(reports_dir, 'Bioactivity_Aggregated_Benchmark.png')
        fig.savefig(save_path, dpi=300)
