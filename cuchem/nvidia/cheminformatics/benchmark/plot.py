import glob
import pandas as pd
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = '/workspace/megamolbart/benchmark'
RADIUS = 0.1

def create_data_sets(output_dir=OUTPUT_DIR):
    """Load data files and coalesce into dataset"""

    # Combine files
    data_files = glob.glob(os.path.join(output_dir, 'tables', '*.csv'))
    data_list = list()
    for data_file in data_files:
        data = pd.read_csv(data_file)
        metric = data.columns[0]

        data['metric'] = metric # TODO remove this once data files are changed
        data.rename(columns={metric:'value'}, inplace=True)

        data_list.append(data)
    data_agg = pd.concat(data_list, axis=0)
    data_agg['iteration'] = data_agg['iteration'].astype(int)

    # Find max radius 
    # TODO -- remove this / clean  up once radius is fixed
    val_uniq_data = data_agg[data_agg['radius'] == RADIUS]

    # Create individual datasets
    val_dat = val_uniq_data[val_uniq_data['metric'] == 'validity'][['value', 'iteration']].set_index('iteration').sort_index()['value']
    unq_dat = val_uniq_data[val_uniq_data['metric'] == 'unique'][['value', 'iteration']].set_index('iteration').sort_index()['value']
    nearest_neighbor_data = data_agg[data_agg['metric']=='nearest neighbor correlation'].groupby('iteration')['value'].mean().sort_index()
    unq_dat = unq_dat / val_dat # Normalize to number of valid molecules

    val_dat.name = 'Validity'
    unq_dat.name = 'Uniqueness'
    nearest_neighbor_data.name = 'Nearest Neighbor Correlation'

    return val_dat, unq_dat, nearest_neighbor_data


def create_plot(val_dat, unq_dat, nearest_neighbor_data, output_dir=OUTPUT_DIR):
    """Create plot of metrics"""
    green = '#86B637'
    blue = '#5473DC'
    fig, axList = plt.subplots(ncols=2)
    fig.set_size_inches(10, 5)

    # Validity and uniqueness plots
    ax = axList[0]
    val_dat.plot(kind='line', ax=ax, legend=True, color=green, rot=45, label='Validity', marker='o', ms=10)
    unq_dat.plot(kind='line', ax=ax, legend=True, color=blue, rot=45, label='Uniqueness', marker='o', ms=10)
    ax.set(ylabel='Percentage', xlabel='Iteration', title='Validity and Uniqueness')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower left')

    # Nearest neighbor plot
    ax = axList[1]
    nearest_neighbor_data.plot(kind='line', ax=ax, legend=False, color=green, rot=45, marker='o', ms=10)
    ax.set(ylabel='Correlation', xlabel='Iteration', title='Nearest Neighbor Correlation')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plots', 'metrics.png'), dpi=300, facecolor='white')


if __name__ == '__main__':
    val_dat, unq_dat, nearest_neighbor_data = create_data_sets()
    create_plot(val_dat, unq_dat, nearest_neighbor_data)
