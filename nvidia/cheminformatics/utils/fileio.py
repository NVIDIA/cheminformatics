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

BENCHMARK_FILE = './benchmark.csv'


def initialize_logfile(benchmark_file=BENCHMARK_FILE):
    """Initialize benchmark file with header if needed"""

    if not os.path.exists(benchmark_file):
        with open(benchmark_file, 'w') as fh:
            fh.write('date,benchmark_type,step,time(hh:mm:ss.ms),n_molecules,n_workers,metric_name,metric_value\n')


def log_results(date, benchmark_type, step, time, n_molecules, n_workers, metric_name='', metric_value='', benchmark_file=BENCHMARK_FILE):
    """Log benchmark results to a file"""

    out_list = [date, benchmark_type, step, time, n_molecules, n_workers, metric_name, metric_value]
    out_fmt = ','.join(['{}'] * len(out_list)) + '\n'

    with open(benchmark_file, 'a') as fh:
        out_string = out_fmt.format(*out_list)
        fh.write(out_string)
