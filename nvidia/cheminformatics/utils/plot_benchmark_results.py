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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# default input and out files
BENCHMARK_FILE = 'benchmark.csv'
XLSX_FILE = 'benchmark.xlsx'
PLOT_FILE = 'benchmark.png'

# defaults to categorize steps for sorting
STEP_TYPE_DICT = { 'dim_reduction': ['pca', 'svd'],
                   'clustering': ['kmeans'],
                   'embedding': ['umap'],
                   'workflow': ['workflow'],
                   'stats': ['total', 'acceleration'] }

STEP_TYPE_CAT = pd.CategoricalDtype(['dim_reduction', 'clustering', 'embedding', 'workflow', 'stats'], ordered=True)

NV_PALETTE = ['#86B637', '#8F231C', '#3D8366', '#541E7D', '#1B36B6']

def prepare_benchmark_df(benchmark_file, xlsx_file, step_type_dict, step_type_cat):
    """Read and prepare the benchmark data as a dataframe"""

    # load and format data
    bench_df = pd.read_csv(benchmark_file, infer_datetime_format=True).rename(columns={'time(hh:mm:ss.ms)':'time'})
    bench_df['date'] = pd.to_datetime(bench_df['date'])
    bench_df['time'] = pd.to_timedelta(bench_df['time']).map(lambda x: x.total_seconds())
    bench_df['benchmark_type'] = pd.Categorical(bench_df['benchmark_type'].str.upper())
    bench_df = bench_df[bench_df['step'] != 'total'] # calculate total later

    # assign step type
    bench_df['step_type'] = ''
    for key in step_type_dict:
        bench_df.loc[bench_df['step'].str.lower().isin(step_type_dict[key]), 'step_type'] = key
    bench_df['step_type'] = bench_df['step_type'].astype(step_type_cat)

    # convert to a pivot table
    cpu_baseline_index = ('CPU', bench_df.loc[bench_df['benchmark_type']=='CPU', 'n_workers'].max()) # assumes max CPU entry is normalizer
    bench_time_df = bench_df.pivot(index=['benchmark_type', 'n_workers', 'n_molecules'], columns=['step_type', 'step'], values='time')
    bench_time_df[('stats','total')] = bench_time_df.sum(axis=1)

    if np.isnan(cpu_baseline_index[1]):
        bench_time_df[('stats','acceleration')] = np.NaN
    else:
        bench_time_df[('stats','acceleration')] = bench_time_df.loc[cpu_baseline_index, ('stats','total')] / bench_time_df[('stats','total')]

    bench_time_df.to_xlsx(xlsx_file)
    return bench_time_df


def prepare_acceleration_stacked_plot(df, plot_file, palette=NV_PALETTE):
    """Prepare single plot of acceleration as stacked bars (by molecule) and hardware workers"""

    grouper = df['stats'].groupby(level='n_molecules')
    n_groups = len(grouper)
    fig, ax = plt.subplots(nrows=1, ncols=1)


    fig.set_size_inches(6 * n_groups, 6)

    df_plot = df[('stats', 'total')].reset_index(level='n_molecules').pivot(columns='n_molecules')
    df_plot.columns = df_plot.columns.get_level_values(level='n_molecules')
    df_plot.plot(kind='bar', ax=ax, color=palette)

    xlabel = 'V100 (GPUs or CPU cores)'
    bars = [rect for rect in ax.get_children() if isinstance(rect, matplotlib.patches.Rectangle)]
    n_rows = len(df_plot)

    for row, (key, dat) in enumerate(df_plot.iterrows()):
        for i, mol in enumerate(df_plot.columns):
            
            # assemble index and get data
            index = tuple(list(key) + [mol])
            total = df.loc[index, ('stats', 'total')]
            accel = df.loc[index, ('stats', 'acceleration')]
            
            # construct label
            label = '{:.0f} s'.format(total)
            if (not np.isnan(accel)) & (index[0] == 'GPU'):
                label += '\n{:.0f}X'.format(accel)
            
            # get bar position and label
            bar = bars[(n_rows * i) + row]
            ypos = bar.get_height()
            xpos = bar.get_x() + (bar.get_width() / 2.0)
            ax.text(xpos, ypos, label, horizontalalignment='center', verticalalignment='bottom')

    xticklabels = ['{} CPU cores'.format(x[1]) if x[0]=='CPU' else '{} GPU(s)'.format(x[1]) for x in df_plot.index.to_list()]
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set(xlabel=xlabel, 
           ylabel='Compute Time (s)\nfor RAPIDS / Sklearn Workflow',
           title='Cheminformatics Visualization Benchmark')

    plt.tight_layout()
    fig.savefig(plot_file, dpi=300)
    
    return


def prepare_acceleration_single_plot(df, plot_file, palette=NV_PALETTE):
    """Prepare plot of each molecule group's acceleration as a single plot"""
    # TODO add acceleration label or plot acceleration

    NV_GREEN = '#86B637'
    grouper = df.groupby(level='n_molecules')
    n_groups = len(grouper)

    fig, axList = plt.subplots(nrows=1, ncols=n_groups)
    if n_groups > 1:
        axList = axList.flatten()
    else:
        axList = [axList]
    fig.set_size_inches(6*n_groups, 6)

    for ax, (n_molecules, dat) in zip(axList, grouper):
        _ = dat[('stats','total')].plot(kind='bar', color=palette, ax=ax)

        offset = 0.01 * dat[('stats','total')].max()
        xlabel = 'V100 (GPUs or CPU cores)'

        for i, (_, v) in enumerate(dat.iterrows()):
            label = '{:.0f} s'.format(v[('stats','total')])
            if (not np.isnan(v[('stats','acceleration')])) & (v.name[0] == 'GPU'):
                label += '\n{:.0f}X'.format(v[('stats','acceleration')])
                xlabel = 'V100 (GPUs or CPU cores)\nAcceleration relative to maximum CPU cores'
            ax.text(i, v[('stats','total')] + offset, label, horizontalalignment='center', verticalalignment='bottom')

        xticklabels = ['{} CPU cores'.format(x[1]) if x[0]=='CPU' else '{} GPU(s)'.format(x[1]) for x in dat.index.to_list()]
        ax.set_xticklabels(xticklabels, rotation='horizontal')
        ax.set(xlabel=xlabel, 
               ylabel='Compute Time (s)\nfor RAPIDS / Sklearn Workflow',
               title='Cheminformatics Visualization Benchmark\n{} Molecules'.format(n_molecules))
    
    plt.tight_layout()
    fig.savefig(plot_file, dpi=300)
    
    return

if __name__ == '__main__':

    # Read and prepare the dataframe then plot
    bench_df = prepare_benchmark_df(benchmark_file=BENCHMARK_FILE, xlsx_file=XLSX_FILE, step_type_dict=STEP_TYPE_DICT, step_type_cat=STEP_TYPE_CAT)
    prepare_acceleration_stacked_plot(bench_df, plot_file=PLOT_FILE)

    