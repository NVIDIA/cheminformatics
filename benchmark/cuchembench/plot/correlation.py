import os
import math
from typing import Optional, List
from .data import load_physchem_input_data, load_bioactivity_input_data
from .utils import setup_plot_grid
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

__ALL__ = ['']

def make_correlation_plots(df: pd.DataFrame,
                           exp_name: str,
                           max_seq_len: int, 
                           plot_type: str, 
                           reports_dir: Optional[str] = '.', 
                           num_rows_per_page: Optional[int] = 3,
                           property_list: Optional[List[str]] = None):
    """Make correlation plots of predictions for diagnostics

    Args:
        df (pd.DataFrame): Dataframe of input CSV data
        exp_name (str): Experiment name to be selected
        max_seq_len (int): Maximum sequence length for 
        plot_type
        reports_dir (Optional[str], optional): Directory. Defaults to current directory.
        num_rows_per_page
        property_list        
    """

    assert plot_type in ['physchem', 'bioactivity'], AssertionError(f'Error: plot_type must be one of "physchem", "bioactivity", got {plot_type}.')
    
    kpi_field = 'name'
    model_field = 'model'
    comparison_field = 'exp_name'
    models = ['linear_regression', 'support_vector_machine', 'random_forest']
    exp_name_out = exp_name.replace(' ', '_').replace('/', "-")

    if plot_type == 'physchem':
        kpis = ['physchem', 'physchem_esol', 'physchem_freesolv', 'physchem_lipophilicity']
        properties_field = 'property'
        input_data = load_physchem_input_data(max_seq_len=max_seq_len)
        save_path = os.path.join(reports_dir, f'Physchem_Model_Diagnostic_Plots_{exp_name_out}.pdf')
    else:
        kpis = ['bioactivity']
        properties_field = 'gene'
        input_data = load_bioactivity_input_data(max_seq_len=max_seq_len)
        save_path = os.path.join(reports_dir, f'Bioactivity_Model_Diagnostic_Plots_{exp_name_out}.pdf')

    # Get selected data
    query = f'({kpi_field} in {kpis}) & ({comparison_field} == "{exp_name}")'
    if property_list:
        query += f' & ({properties_field} in {property_list})'
    df = df.query(query)
    properties = df[properties_field].unique()

    plots_per_row = len(models)
    num_rows = len(properties)
    num_plots = plots_per_row * num_rows_per_page
    num_pages = int(math.ceil(num_rows / num_rows_per_page))

    with PdfPages(save_path) as pdf:
        for page in range(num_pages):
            fig, ax_list = setup_plot_grid(num_plots=num_plots, plots_per_row=plots_per_row, xscale=4, yscale=4)
            for row, property_ in enumerate(properties):
                for col, model in enumerate(models):
                    xdata = input_data[property_]
                    mask = (df[model_field] == model) & (df[properties_field] == property_)        

                    predictions = df[mask]['predictions'].values[0]
                    fingerprint_pred = predictions['fingerprint_pred']
                    embedding_pred = predictions['embedding_pred']

                    pos = (row * plots_per_row) + col
                    ax = ax_list[pos]
                    ax.plot(xdata, fingerprint_pred, marker='o', ls='', ms=2, color='blue', label='Fingerprint')
                    ax.plot(xdata, embedding_pred, marker='o', ls='', ms=2, color='orange', label='Embedding')

                    if row == 0:
                        ax.set_title(model)
                    if col == 0:
                        ax.set_ylabel(property_)
                        
                    if (row == 0) & (col == 0):
                        ax.legend()

                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    lim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
                    ax.set_xlim(*lim)
                    ax.set_ylim(*lim)

                    # Label with error values
                    fingerprint_error = df[mask]['fingerprint_error'].values[0]
                    embedding_error = df[mask]['embedding_error'].values[0]
                    error_str = [f'fingerprint_error: {fingerprint_error:.3f}',
                                 f'embedding_error: {embedding_error:.3f}']
                    error_str = '\n'.join(error_str)
                    ax.annotate(error_str, xy=(1.0, 0.05), xycoords='axes fraction', fontsize='small', ha='right', va='bottom')

            fig.suptitle(exp_name)
            plt.tight_layout()
            pdf.savefig(fig)
