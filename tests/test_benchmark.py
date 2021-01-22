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

import pytest
import unittest
import tempfile
import os
import shutil
import sys
import shlex
import glob
import pandas as pd

# Define paths for the tests
_this_directory = os.path.dirname(os.path.realpath(__file__))
_parent_directory = os.path.dirname(_this_directory)
_data_dir = os.path.join(_this_directory, 'data')

sys.path.insert(0, _parent_directory) # TODO better way to add this directory to the path
from startdash import Launcher
from nvidia.cheminformatics.utils.plot_benchmark_results import prepare_benchmark_df, prepare_acceleration_stacked_plot

# Output directory
temp_dir = tempfile.mkdtemp()

# Parameter lists
run_benchmark_params = [ ([{'test_type': 'gpu', 'n_workers':  1, 'n_mol': -1}, 
                           {'test_type': 'cpu', 'n_workers': 19, 'n_mol': -1}], _data_dir, temp_dir) ]
load_benchmark_params = [(temp_dir)]


@pytest.mark.parametrize('benchmark_config_list, data_dir, output_dir', run_benchmark_params)
def test_run_benchmark(benchmark_config_list, data_dir, output_dir):

    output_file = os.path.join(output_dir, 'benchmark.csv')
    if os.path.exists(output_file):
        os.remove(output_file)

    for config in benchmark_config_list:
        test_type = config['test_type']
        n_workers = config['n_workers']
        n_mol = config['n_mol']
        
        # Create run command and inject into sys.argv
        command = f'startdash.py analyze -b --cache {data_dir} '
        if test_type == 'cpu':
            command += f'--{test_type} '
        command += f'--n_{test_type} {n_workers} --n_mol {n_mol} --output_dir {output_dir}'

        sys_argv = shlex.split(command)
        with unittest.mock.patch('sys.argv', sys_argv):
            Launcher()

    # Filename is set in workflow -- move to create randomized name
    temp_file = tempfile.NamedTemporaryFile(prefix='benchmark_', suffix='.csv', dir=output_dir, delete=False).name
    shutil.move(output_file, temp_file)
    assert os.path.exists(temp_file)
    
    benchmark_results = pd.read_csv(temp_file, comment='#')
    nrows, ncols = benchmark_results.shape
    assert nrows == len(benchmark_config_list) * 5
    assert ncols == 8


@pytest.mark.parametrize('output_dir', load_benchmark_params)
def test_load_benchmarks(output_dir):

    csv_path = os.path.join(output_dir, 'benchmark_*.csv')
    for benchmark_file in glob.glob(csv_path):
        df, machine_config = prepare_benchmark_df(benchmark_file)
        basename = os.path.splitext(benchmark_file)[0]
        excel_file = basename + '.xlsx'
        assert os.path.exists(excel_file)
        md_file = basename + '.md'
        assert os.path.exists(md_file)
        
        png_file = basename + '.png'
        prepare_acceleration_stacked_plot(df, machine_config, output_path=png_file)
        assert os.path.exists(png_file)


# TODO add test for metrics and values