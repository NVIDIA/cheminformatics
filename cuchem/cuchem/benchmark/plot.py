#!/usr/bin/env python3

# TODO cleanup and / or remove as appropriate

import glob
import os
import sys
import matplotlib.pyplot as plt
import argparse
import numpy as np
import textwrap
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Plot Results')

    parser.add_argument('-i', '--input_dir',
                        dest='input_dir',
                        type=str,
                        default='/workspace/megamolbart/benchmark',
                        help='Path containing CSV files')

    parser.add_argument('-o', '--output_dir',
                        dest='output_dir',
                        type=str,
                        help='Output directory -- defaults to input_dir')

    parser.add_argument('-r', '--radius',
                        dest='radius',
                        type=float,
                        default=0.1,
                        help='Radius to select for appropriate metrics')

    parser.add_argument('-t', '--top_k',
                        dest='top_k',
                        type=int,
                        default=None,
                        help='Top K for Nearest Neighbor -- default is max value')   

    args = parser.parse_args(sys.argv[1:])

    if args.output_dir is None:
        args.output_dir = args.input_dir

    return args


def create_data_sets(input_dir, radius, top_k):
    """Load data files and coalesce into dataset"""

    # Combine files
    data_files = glob.glob(os.path.join(input_dir, '*.csv'))
    assert len(data_files) > 0
    data_list = list()
    for data_file in data_files:
        data = pd.read_csv(data_file)
        data = data.replace('unique', 'uniqueness')
        data_list.append(data)

    data_agg = pd.concat(data_list, axis=0)

    # Clean up data
    top_k = data_agg['top_k'].max() if top_k is None else top_k
    mask = (data_agg['radius'] == radius) | (data_agg['top_k'] == top_k) | data_agg['model'].notnull()
    data_agg = data_agg[mask]

    # Set sort order
    name_category = pd.CategoricalDtype(['validity', 'novelty', 'uniqueness', 
                                            'nearest neighbor correlation', 'modelability'], 
                                        ordered=True)

    model_category = pd.CategoricalDtype(['linear regression', 'elastic net', 'support vector machine', 'random forest'],
                                        ordered=True)
    data_agg['name'] = data_agg['name'].astype(name_category)
    data_agg['model'] = data_agg['model'].astype(model_category)
    data_agg = data_agg.sort_values(['name', 'model'])

    return data_agg


def create_plot(data, radius, iteration, output_dir):
    """Create plot of metrics"""

    def _clean_label(label):
        label = label.get_text().title()
        label = textwrap.wrap(label, width=20)
        label = '\n'.join(label)
        return label

    green = '#86B637'
    blue = '#5473DC'
    fig, axList = plt.subplots(ncols=2)
    fig.set_size_inches(10, 5)

    # Validity, uniqueness, novelty, and nearest neighbor correlation plot
    ax = axList[0]
    mask = data['name'] != 'modelability'
    data.loc[mask, ['name', 'value']].set_index('name').plot(kind='bar', ax=ax, legend=False, color=green, rot=45)
    xlabels = [_clean_label(x) for x in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)
    ax.set(ylabel='Percentage', xlabel='Metric', title=f'Metrics at Radius {radius} with Model at Iteration {iteration}')
    ax.set_ylim(0, 1.0)

    # ML Model Error Ratios
    ax = axList[1]
    data.loc[mask.pipe(np.invert), ['model', 'value']].set_index('model').plot(kind='bar', ax=ax, legend=False, color=green, rot=45)
    ax.set(ylabel='Ratio of Mean Squared Errors\n(Morgan Fingerprint / Embedding)', xlabel='Model', title='Modelability Ratio: Higher --> Better Embeddings')
    xlabels = [_clean_label(x) for x in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metrics.png'), dpi=300, facecolor='white')


if __name__ == '__main__':
    
    args = parse_args()
    data = create_data_sets(args.input_dir, args.radius, args.top_k)

    assert data['iteration'].nunique() == 1
    iteration = data['iteration'].iloc[0]
    create_plot(data, args.radius, iteration, args.output_dir)

