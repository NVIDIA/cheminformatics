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
import matplotlib.pyplot as plt

# default csv file
BENCHMARK_FILE = 'benchmark.csv'

# defaults to categorize steps for sorting
STEP_TYPE_DICT = { 'dim_reduction': ['pca', 'svd'],
                   'clustering': ['kmeans'],
                   'embedding': ['umap'],
                   'workflow': ['plotting'] }

STEP_TYPE_CAT = pd.CategoricalDtype(['dim_reduction', 'clustering', 'embedding', 'workflow', 'stats'], ordered=True)


def prepare_benchmark_df(benchmark_file, step_type_dict, step_type_cat):
    """Read and prepare the benchmark data"""

    # load and format data
    bench_df = pd.read_csv(benchmark_file, infer_datetime_format=True).rename(columns={'time(hh:mm:ss.ms)':'time'})
    bench_df['date'] = pd.to_datetime(bench_df['date'])
    bench_df['time'] = pd.to_timedelta(bench_df['time']).map(lambda x: x.total_seconds())
    bench_df['benchmark_type'] = pd.Categorical(bench_df['benchmark_type'].str.upper())

    bench_df = bench_df.replace('workflow', 'plotting')
    bench_df = bench_df[bench_df['step'] != 'total'] # calculate total later

    # assign step type
    bench_df['step_type'] = ''
    for key in step_type_dict:
        bench_df.loc[bench_df['step'].str.lower().isin(step_type_dict[key]), 'step_type'] = key
    bench_df['step_type'] = bench_df['step_type'].astype(step_type_cat)

    # convert to a pivot table
    cpu_baseline_index = ('CPU', bench_df.loc[bench_df['benchmark_type']=='CPU', 'n_workers'].max()) # assumes max CPU entry is normalizer
    bench_time_df = bench_df.pivot(index=['benchmark_type', 'n_workers'], columns=['step_type', 'step'], values='time')
    bench_time_df['total'] = bench_time_df.sum(axis=1)
    bench_time_df['acceleration'] = bench_time_df.loc[cpu_baseline_index, ('total', '')] / bench_time_df['total']
    
    return bench_time_df


def prepare_acceleration_plot(df, output_path):
    """Prepare plot of """
    # TODO add acceleration label or plot acceleration

    NV_GREEN = '#86B637'

    fig = plt.figure()
    fig.set_size_inches(6, 6)
    ax = plt.axes()

    _ = df['total'].plot(kind='bar', color=NV_GREEN, ax=ax)

    for i, (_, v) in enumerate(df['total'].iteritems()):
        ax.text(i, v+100, '{:.0f} s'.format(v), horizontalalignment='center', verticalalignment='bottom')

    xticklabels = ['{} CPU cores'.format(x[1]) if x[0]=='CPU' else '{} GPU(s)'.format(x[1]) for x in df.index.to_list()]
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set(xlabel='V100 (GPUs or CPU cores)', 
           ylabel='Compute Time (s)\nfor RAPIDS / Sklearn Workflow', 
           title='Cheminformatics Visualization Benchmark')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    
    return

if __name__ == '__main__':
    # read and prepare the dataframe
    bench_df = prepare_benchmark_df(benchmark_file=BENCHMARK_FILE, step_type_dict=STEP_TYPE_DICT, step_type_cat=STEP_TYPE_CAT)

    try:
        bench_df.to_xlsx()
    except:
        pass

    prepare_acceleration_plot(bench_df, output_path)

    