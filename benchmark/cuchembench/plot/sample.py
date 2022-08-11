from typing import Optional
import pandas as pd
import os
import matplotlib.pyplot as plt

from .utils import set_plotting_style, setup_plot_grid, label_bars

__ALL__ = ['make_multimodel_sampling_plots']


def make_multimodel_sampling_plots(df: pd.DataFrame, 
                                   acceptance_criteria: Optional[dict] = None,
                                   save_plots: bool = False, 
                                   plots_per_row: int = 2,
                                   reports_dir: Optional[str] = '.'):
    """Make plots for sampling metrics

    Args:
        df (pd.DataFrame): Dataframe of input CSV data
        acceptance_criteria (Optional[dict], optional): dictionary with minimum acceptance criteria for metrics. 
                                Defaults to None.
        save_plots (bool, optional): Save plots to disk. Defaults to False.
        plots_per_row (int, optional): Number of plots per row. Defaults to 2.
        reports_dir (Optional[str], optional): Directory. Defaults to current directory.
    """
    
    # Other config -- TODO consider making configurable
    kpis = ['validity', 'novelty', 'unique']
    kpi_field = 'name'
    comparison_field = 'exp_name'
    value_field = 'value'
    
    # Get selected data and preserve experiment sort order
    exp_sort_order = pd.CategoricalDtype(df[comparison_field].unique(), ordered=True)
    query = f'{kpi_field} in {kpis}'
    df = df.query(query)[[kpi_field, comparison_field, value_field]]
    df.loc[:, comparison_field] = df[comparison_field].astype(exp_sort_order)

    # Setup plots        
    set_plotting_style()
    num_plots = len(kpis)
    fig, axes_list = setup_plot_grid(num_plots, plots_per_row)

    for i, kpi in enumerate(kpis):
        query_str = f'{kpi_field} == "{kpi}"'
        plt_df = df.query(query_str)
        
        plt_df.loc[:, comparison_field] = plt_df[comparison_field].astype(exp_sort_order)
        
        labels = plt_df[comparison_field]
        values = plt_df[value_field]
        
        ax = axes_list[i]
        
        if acceptance_criteria and acceptance_criteria.get(kpi, False):
            ax.axhline(y=acceptance_criteria[kpi], xmin=0, xmax=1, color='red', lw=2.0, zorder=-1)
            
        ax.bar(labels, values)
        ax.tick_params(axis='x', rotation=15)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'Sampling: {kpi.title()}')
        ax.set_ylabel('Value')
        label_bars(ax)
        
    fig.subplots_adjust(left=0.01,
                        bottom=0.01, 
                        right=0.99, 
                        top=0.90, 
                        wspace=0.1, 
                        hspace=0.2)
    plt.tight_layout()
    if save_plots:
        save_path = os.path.join(reports_dir, 'Sampling_Aggregated_Benchmark.png')
        fig.savefig(save_path, dpi=300)
