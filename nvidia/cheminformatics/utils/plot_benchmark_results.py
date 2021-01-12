#!/opt/conda/envs/rapids/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

# default input and out files
BENCHMARK_FILE = 'benchmark.csv'

# defaults to categorize steps for sorting
STEP_TYPE_DICT = {'dim_reduction': ['pca', 'svd'],
                  'clustering': ['kmeans'],
                  'embedding': ['umap'],
                  'workflow': ['workflow'],
                  'stats': ['total', 'acceleration']}

STEP_TYPE_CAT = pd.CategoricalDtype(
    ['dim_reduction', 'clustering', 'embedding', 'workflow', 'stats'], ordered=True)

NV_PALETTE = ['#86B637', '#8F231C', '#3D8366', '#541E7D', '#1B36B6']


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze and plot benchmark data')
    parser.add_argument('-b', '--benchmark_file',
                        dest='benchmark_file',
                        type=str,
                        default='/workspace/benchmark/benchmark.csv',
                        help='Path to the CSV file containing benchmark results')
    parser.add_argument('-o', '--output_path',
                        dest='output_path',
                        type=str,
                        default='/workspace/benchmark/benchmark.png',
                        help='Output directory for plot')

    args = parser.parse_args(sys.argv)
    return args


def prepare_benchmark_df(benchmark_file, step_type_dict=STEP_TYPE_DICT, step_type_cat=STEP_TYPE_CAT):
    """Read and prepare the benchmark data as a dataframe"""

    # Load and format data
    bench_df = pd.read_csv(benchmark_file, infer_datetime_format=True).rename(
        columns={'time(hh:mm:ss.ms)': 'time'})
    bench_df['date'] = pd.to_datetime(bench_df['date'])
    bench_df['time'] = pd.to_timedelta(
        bench_df['time']).map(lambda x: x.total_seconds())
    bench_df['benchmark_type'] = pd.Categorical(
        bench_df['benchmark_type'].str.upper())

    # Calculate total later by different method
    bench_df = bench_df[bench_df['step'] != 'total']

    # assign step type as a category to control display order
    bench_df['step_type'] = ''
    for key in step_type_dict:
        bench_df.loc[bench_df['step'].str.lower().isin(
            step_type_dict[key]), 'step_type'] = key
    bench_df['step_type'] = bench_df['step_type'].astype(step_type_cat)

    # convert to a pivot table with columns containing consecutive steps
    bench_time_df = (bench_df
                     .drop(['metric_name', 'metric_value'], axis=1)
                     .pivot(index=['benchmark_type', 'n_workers', 'n_molecules'],
                            columns=['step_type', 'step'],
                            values='time'))
    bench_time_df[('stats', 'total')] = bench_time_df.sum(axis=1)

    # Create dataframe to normalize totals to max workers for CPU
    # Requires manipulation of pivot table index formats
    bench_time_df_norm = bench_time_df.copy()
    bench_time_df_norm.columns = pd.MultiIndex.from_tuples(
        bench_time_df.columns)
    bench_time_df_norm.reset_index(inplace=True)

    mask_indexes = bench_time_df_norm.groupby(['benchmark_type', 'n_molecules'])[
        'n_workers'].transform(lambda x: x == x.max())
    norm_df = bench_time_df_norm[mask_indexes].groupby(
        ['benchmark_type', 'n_workers', 'n_molecules']).mean()[('stats', 'total')].dropna()
    cpu_only_mask = norm_df.index.get_level_values(
        level='benchmark_type') == 'CPU'
    norm_df = norm_df[cpu_only_mask].reset_index(
        level=['benchmark_type', 'n_workers'], drop=True)  # Normalize by n_molecules only

    # Do the normalization
    bench_time_df[('stats', 'acceleration')] = bench_time_df[(
        'stats', 'total')].div(norm_df).pow(-1)

    basename = os.path.splitext(benchmark_file)[0]
    bench_time_df.to_excel(basename + '.xlsx')
    # bench_time_df.to_markdown(basename + '.md') # TODO fix markdown export

    return bench_time_df


def prepare_acceleration_stacked_plot(df, output_path, palette=NV_PALETTE):
    """Prepare single plot of acceleration as stacked bars (by molecule) and hardware workers"""

    grouper = df['stats'].groupby(level='n_molecules')
    n_groups = len(grouper)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(6 * n_groups, 6)

    df_plot = df[('stats', 'total')].reset_index(
        level='n_molecules').pivot(columns='n_molecules')
    df_plot.columns = df_plot.columns.get_level_values(level='n_molecules')
    # this may need to be adjusted with many groups
    bar_width = (1.0 / n_groups) * 1.1
    df_plot.plot(kind='bar', ax=ax, color=palette, width=bar_width)

    xlabel = 'V100 (GPUs or CPU cores)'
    bars = [rect for rect in ax.get_children() if isinstance(
        rect, matplotlib.patches.Rectangle)]
    n_rows = len(df_plot)

    for row, (key, dat) in enumerate(df_plot.iterrows()):
        for i, mol in enumerate(df_plot.columns):

            # Assemble index and get data
            index = tuple(list(key) + [mol])
            total = df.loc[index, ('stats', 'total')]
            accel = df.loc[index, ('stats', 'acceleration')]

            # Construct label and update legend title
            label = '{:.0f} s'.format(total)
            if (not np.isnan(accel)) & (index[0] == 'GPU'):
                label += '\n{:.0f}X'.format(accel)
                xlabel = 'V100 (GPUs or CPU cores)\nAcceleration relative to maximum CPU cores'

            # Get bar position and label
            bar = bars[(n_rows * i) + row]
            ypos = bar.get_height()
            xpos = bar.get_x() + (bar.get_width() / 2.0)
            ax.text(xpos, ypos, label, horizontalalignment='center',
                    verticalalignment='bottom')

    xticklabels = ['{} CPU cores'.format(x[1]) if x[0] == 'CPU' else '{} GPU(s)'.format(
        x[1]) for x in df_plot.index.to_list()]
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set(xlabel=xlabel, ylabel='Compute Time (s)\nfor RAPIDS / Sklearn Workflow',
           title='Cheminformatics Visualization Benchmark')

    ax.legend(title='Num Molecules')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    return


if __name__ == '__main__':

    args = parse_args()

    # Read and prepare the dataframe then plot
    bench_df = prepare_benchmark_df(benchmark_file=args.benchmark_file, step_type_dict=STEP_TYPE_DICT, 
                                    step_type_cat=STEP_TYPE_CAT)
    prepare_acceleration_stacked_plot(bench_df, output_path=args.output_path)
