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

# defaults to categorize steps for sorting
STEP_TYPE_DICT = {'dim_reduction': ['pca', 'svd'],
                  'clustering': ['kmeans'],
                  'embedding': ['umap'],
                  'workflow': ['workflow'],
                  'stats': ['total', 'acceleration']}

STEP_TYPE_CAT = pd.CategoricalDtype(
    ['n_molecules', 'benchmark_type', 'n_workers', 'dim_reduction', 'clustering', 'embedding', 'workflow', 'stats'], ordered=True)

NV_PALETTE = ['#8F231C', '#3D8366', '#541E7D', '#1B36B6', '#7B1D56', '#86B637']


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

    args = parser.parse_args(sys.argv[1:])
    return args


def prepare_benchmark_df(benchmark_file, step_type_dict=STEP_TYPE_DICT, step_type_cat=STEP_TYPE_CAT):
    """Read and prepare the benchmark data as a dataframe"""

    # Load and format data
    with open(benchmark_file, 'r') as fh:
        machine_config = pd.Series({'Machine Config': fh.readlines()[0].replace('#', '').strip()})

    bench_df = pd.read_csv(benchmark_file, infer_datetime_format=True, comment='#').rename(
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
                     .pivot(index=['n_molecules', 'benchmark_type', 'n_workers'],
                            columns=['step_type', 'step'],
                            values='time'))
    bench_time_df[('stats', 'total')] = bench_time_df.sum(axis=1)

    # Create dataframe to normalize totals to max workers for CPU
    # Requires manipulation of pivot table index formats
    bench_time_df_norm = bench_time_df.copy()
    bench_time_df_norm.columns = pd.MultiIndex.from_tuples(
        bench_time_df.columns)
    bench_time_df_norm.reset_index(inplace=True)

    mask_indexes = bench_time_df_norm.groupby(['n_molecules', 'benchmark_type'])[
        'n_workers'].transform(lambda x: x == x.max())
    norm_df = bench_time_df_norm[mask_indexes].groupby(
        ['n_molecules', 'benchmark_type', 'n_workers']).mean()[('stats', 'total')].dropna()
    cpu_only_mask = norm_df.index.get_level_values(
        level='benchmark_type') == 'CPU'
    norm_df = norm_df[cpu_only_mask].reset_index(
        level=['n_workers', 'benchmark_type'], drop=True)  # Normalize by n_molecules only

    # Do the normalization
    bench_time_df[('stats', 'acceleration')] = bench_time_df[(
        'stats', 'total')].div(norm_df).pow(-1)

    # Standardize columns for output
    bench_time_df_output = bench_time_df.copy().round(2)
    columns = bench_time_df_output.columns.get_level_values('step').to_list()
    bench_time_df_output.columns = pd.Categorical(columns, categories=['n_molecules', 'benchmark_type', 'n_workers'] + columns, ordered=True)

    basename = os.path.splitext(benchmark_file)[0]
    with pd.ExcelWriter(basename + '.xlsx') as writer:
        bench_time_df_output.to_excel(writer, sheet_name='Benchmark')
        machine_config.to_excel(writer, sheet_name='Machine Config')

    with open(basename + '.md', 'w') as fh:
        filelines = f'# {machine_config.values[0]}\n\n'
        filelines += bench_time_df_output.reset_index().to_markdown(index=False)
        fh.write(filelines)

    return bench_time_df, machine_config


def prepare_acceleration_stacked_plot(df, machine_config, output_path, palette=NV_PALETTE):
    """Prepare single plot of acceleration as stacked bars (by molecule) and hardware workers"""

    grouper = df['stats'].groupby(level='n_molecules')
    n_groups = len(grouper)
    n_rows = min(2, n_groups)
    n_cols = int(n_groups / n_rows + 0.5)

    fig, axList = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.set_size_inches(6 * n_cols, 6 * n_rows)

    if n_groups == 1:
        axList = [axList]
    else:
        axList = axList.flatten()


    df_plot = df[('stats', 'total')].reset_index(
        level='n_molecules').pivot(columns='n_molecules')
    df_plot.columns = df_plot.columns.get_level_values(level='n_molecules')
    df_plot = df_plot.T

    bar_width = 1.0

    for ax, (n_molecules, dat) in zip(axList, df_plot.iterrows()):
        dat.plot(kind='bar', ax=ax, color=palette, width=bar_width)
        
        bars = [rect for rect in ax.get_children() if isinstance(rect, matplotlib.patches.Rectangle)]
        indexes = [tuple([n_molecules] + list(x)) for x in dat.index.to_list()]
        
        # Assemble index and label bars
        for bar, index in zip(bars, indexes):
            total = df.loc[index, ('stats', 'total')]
            accel = df.loc[index, ('stats', 'acceleration')]
            label = '{:.0f} s'.format(total)
            if (not np.isnan(accel)) & (index[1] == 'GPU'):
                label += '\n{:.0f}X'.format(accel)

            ypos = bar.get_height()
            xpos = bar.get_x() + (bar.get_width() / 2.0)
            ax.text(xpos, ypos, label, horizontalalignment='center',
                    verticalalignment='bottom')
            
        xticklabels = [f'{x[1]} CPU cores' if x[0] == 'CPU' else f'{x[1]} GPU(s)' for x in dat.index.to_list()]
        ax.set_xticklabels(xticklabels, rotation=25)
        ax.set(title=f'{n_molecules:,} Molecules', xlabel='')
        if ax.is_first_col():
            ax.set(ylabel='Compute Time (s)\nfor RAPIDS / Sklearn Workflow')

    title = f'Cheminformatics Visualization Benchmark\n{machine_config.values[0]}\n'
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, transparent=False)
    return


if __name__ == '__main__':

    args = parse_args()

    # Read and prepare the dataframe then plot
    bench_df, machine_config = prepare_benchmark_df(benchmark_file=args.benchmark_file, step_type_dict=STEP_TYPE_DICT, 
                                    step_type_cat=STEP_TYPE_CAT)
    prepare_acceleration_stacked_plot(bench_df, machine_config, output_path=args.output_path)
