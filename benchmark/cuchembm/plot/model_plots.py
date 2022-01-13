import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import pandas as pd
from .data import (PHYSCHEM_UNIT_RENAMER,
                   load_aggregated_metric_results, 
                   load_physchem_input_data, 
                   load_bioactivity_input_data, 
                   load_plot_data)

__ALL__ = ['make_model_plots']


def grouper(list_, num_rows):
    list_ = list(list_)
    for i in range(0, len(list_), num_rows):
        yield list_[i: i + num_rows]


def make_model_plots(max_seq_len, plot_type, output_dir, n_plots_page=10):

    assert plot_type in ['physchem', 'bioactivity'], AssertionError(f"Error: plot type must be one of 'physchem' or 'bioactivity'.")

    if plot_type == 'physchem':
        input_data_func = load_physchem_input_data
        pkl_path = os.path.join(output_dir, '**', '*physchem.pkl')
        group_col = 'property'
        index_cols = ['inferrer', 'property', 'model']
        output_path = os.path.join(output_dir, 'Physchem_Single_Property_Plots.pdf')
    elif plot_type == 'bioactivity':
        input_data_func = load_bioactivity_input_data
        pkl_path = os.path.join(output_dir, '**', '*bioactivity.pkl')
        group_col = 'gene'
        index_cols = ['inferrer', 'gene', 'model']
        output_path = os.path.join(output_dir, 'Bioactivity_Single_Gene_Plots.pdf')

    keep_cols = index_cols + ['fingerprint_error', 'embedding_error']

    input_data = input_data_func(max_seq_len=max_seq_len)
    pred_data = load_plot_data(pkl_path=pkl_path, input_data=input_data, group_col=group_col)

    metric_df = load_aggregated_metric_results(output_dir)
    metric_df = metric_df[metric_df['name'] == plot_type].dropna(axis=1, how='all')

    if plot_type == 'physchem':
        pred_data['property'] = pred_data['property'].map(lambda x: PHYSCHEM_UNIT_RENAMER[x])
        metric_df['property'] = metric_df['property'].map(lambda x: PHYSCHEM_UNIT_RENAMER[x])

    pred_data['row'] = pred_data.apply(lambda x: '+'.join([x['property'], x['inferrer']]), axis=1)
    metric_df = metric_df[keep_cols].set_index(index_cols)

    with PdfPages(output_path) as pdf:
        for pred_data_page in grouper(pred_data.groupby(['row']), n_plots_page):
            pred_data_page = pd.concat([x[1] for x in pred_data_page], axis=0)
            
            g = sns.FacetGrid(pred_data_page, 
                            col='model', 
                            row='row', 
                            hue='feature', 
                            margin_titles=False, 
                            sharex=False, 
                            sharey=False)

            g.map_dataframe(sns.scatterplot, x='value', y='prediction', s=4)
            g.set_xlabels('')
            g.set_ylabels('')
            g.set_titles('')
            g.add_legend(handletextpad=0.01, borderpad=0.02, frameon=True)

            legend_data = g._legend_data
            legend_colors = dict([(x, y.get_facecolor()) for x, y in legend_data.items()])

            for (row, model), ax in g.axes_dict.items():
                prop, inferrer = row.split('+')
                error_vals = metric_df.loc[inferrer, prop, model]

                if ax.is_first_row():
                    ax.set_title(model.replace('_', ' ').title())
                if ax.is_last_row():
                    ax.set_xlabel('Property Value')
                if ax.is_first_col():
                    ax.set_ylabel(f'{inferrer}\n{prop}\nPredicted Value')

                error_str = []
                for feature_type in legend_colors:
                    color = legend_colors[feature_type]
                    error = error_vals[f'{feature_type}_error']
                    error_str += [f'{feature_type.title()}: {error:.3f}']

                error_str = '\n'.join(error_str)
                ax.annotate(error_str, xy=(1.0, 0.05), xycoords='axes fraction', fontsize='small', ha='right', va='bottom')

            plt.tight_layout()
            pdf.savefig(g.fig)
    return
