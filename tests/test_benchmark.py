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
import sys
import shlex
import shutil
import pandas as pd

# Define paths for the tests
_this_directory = os.path.dirname(os.path.realpath(__file__))
_parent_directory = os.path.dirname(_this_directory)
_data_dir = os.path.join(_this_directory, 'data')

sys.path.insert(0, _parent_directory) # TODO better way to add this to the path
from startdash import Launcher
from nvidia.cheminformatics.utils.plot_benchmark_results import prepare_benchmark_df, prepare_acceleration_stacked_plot

# Output directory and parameters
temp_dir = tempfile.mkdtemp()
benchmark_file = os.path.join(temp_dir, 'benchmark.csv')

# Parameter lists
run_benchmark_params = [(_data_dir, 'gpu',  1, -1, benchmark_file),
                        (_data_dir, 'cpu', 19, -1, benchmark_file)]
load_benchmark_params = [(benchmark_file)]

@pytest.mark.parametrize('data_dir, test_type, n_workers, n_mol, benchmark_file', run_benchmark_params)
def test_run_benchmark(data_dir, test_type, n_workers, n_mol, benchmark_file):

    # Create run command and inject into sys.argv
    command = f'startdash.py analyze -b --cache {data_dir} '
    if test_type == 'cpu':
        command += f'--{test_type} '
    command += f'--n_{test_type} {n_workers} --n_mol {n_mol} --output_path {benchmark_file}'

    sys_argv = shlex.split(command)
    with unittest.mock.patch('sys.argv', sys_argv):
        Launcher()

    print(benchmark_file)
    assert os.path.exists(benchmark_file)
    benchmark_results = pd.read_csv(benchmark_file)
    nrows, ncols = benchmark_results.shape
    assert nrows >= 5
    assert ncols == 8

@pytest.mark.parametrize('benchmark_file', load_benchmark_params)
def test_load_benchmark(benchmark_file):
    df = prepare_benchmark_df(benchmark_file)
    print(df.columns)

    basename = os.path.splitext(benchmark_file)[0]
    excel_file = basename + '.xlsx'
    md_file = basename + '.md'
    assert os.path.exists(excel_file)
    #assert os.path.exists(md_file) # TODO FIX ME