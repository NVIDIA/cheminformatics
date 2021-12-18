import matplotlib.pyplot as plt
import seaborn as sns
import os
from .data import (load_physchem_input_data, 
                   load_bioactivity_input_data, 
                   load_plot_data)

__ALL__ = ['make_model_plots']

def make_model_plots(max_seq_len, plot_type, output_dir):

    assert plot_type in ['physchem', 'bioactivity'], AssertionError(f"Error: plot type must be one of 'physchem' or 'bioactivity'.")

    if plot_type == 'physchem':
        input_data_func = load_physchem_input_data
        pkl_path = os.path.join(output_dir, '*physchem.pkl')
        group_col = 'property'
        output_path = os.path.join(output_dir, 'Physchem_Single_Property_Plots.png')
    elif plot_type == 'bioactivity':
        input_data_func = load_bioactivity_input_data
        pkl_path = os.path.join(output_dir, '*bioactivity.pkl')
        group_col = 'gene'
        output_path = os.path.join(output_dir, 'Bioactivity_Single_Gene_Plots.png')

    input_data = input_data_func(max_seq_len=max_seq_len)
    pred_data = load_plot_data(pred_path=pkl_path, input_data=input_data, group_col=group_col)

    g = sns.FacetGrid(pred_data, 
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

    for key, ax in g.axes_dict.items():
        if ax.is_first_row():
            ax.set_title(key[1].replace('_', ' ').title())
        if ax.is_last_row():
            ax.set_xlabel('Property Value')
        if ax.is_first_col():
            property, model = key[0].split(', ')
            ax.set_ylabel(f'{model}\n{property}\nPredicted Value')

    g.add_legend()
    fig = plt.gcf()
    fig.legend(loc=7)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    return