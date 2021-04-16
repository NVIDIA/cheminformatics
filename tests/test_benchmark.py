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
import tempfile
import os
import shutil
import glob
import logging
from pydoc import locate
import pandas as pd

from tests.utils import _create_context
from nvidia.cheminformatics.utils.logger import initialize_logfile
from nvidia.cheminformatics.utils.plot_benchmark_results \
    import prepare_benchmark_df, prepare_acceleration_stacked_plot


logger = logging.getLogger(__name__)


# Parameter lists
run_benchmark_params = [([{'test_type': 'nvidia.cheminformatics.wf.cluster.gpukmeansumap.GpuKmeansUmap',
                           'use_gpu': True,
                           'n_workers':  1,
                           'n_mol': 5000},
                          {'test_type': 'nvidia.cheminformatics.wf.cluster.cpukmeansumap.CpuKmeansUmap',
                           'use_gpu': False,
                           'n_workers': 10,
                           'n_mol': 5000}])]


@pytest.mark.parametrize('benchmark_config_list', run_benchmark_params)
def test_run_benchmark(benchmark_config_list):

    output_dir = tempfile.tempdir
    output_file = os.path.join(output_dir, 'benchmark.csv')
    initialize_logfile(output_file)

    max_n_mol = 0
    for config in benchmark_config_list:
        test_type = config['test_type']
        use_gpu = config['use_gpu']
        n_workers = config['n_workers']
        n_mol = config['n_mol']
        max_n_mol = max(max_n_mol, n_mol)

        context = _create_context(use_gpu=use_gpu,
                                  n_workers=n_workers,
                                  benchmark_file=output_file)
        context.n_molecule = n_mol
        context.cache_directory = None
        context.is_benchmark = True

        wf_class = locate(test_type)
        workflow = wf_class()

        workflow.cluster()
        workflow.compute_qa_matric()

        context.dask_client.cluster.close()
        context.dask_client.close()
        context.dask_client = None

    # Filename is set in workflow -- move to create randomized name
    temp_file = tempfile.NamedTemporaryFile(prefix='benchmark_',
                                            suffix='.csv',
                                            dir=output_dir,
                                            delete=False).name
    shutil.move(output_file, temp_file)
    assert os.path.exists(temp_file)

    benchmark_results = pd.read_csv(temp_file, comment='#')
    logger.info(benchmark_results)

    nrows, ncols = benchmark_results.shape
    assert ncols == 8
    assert nrows >= len(benchmark_config_list)
    assert benchmark_results['n_molecules'].min() > 0
    assert benchmark_results['n_molecules'].min() < max_n_mol

    df, machine_config = prepare_benchmark_df(temp_file)
    basename = os.path.splitext(temp_file)[0]
    excel_file = basename + '.xlsx'
    assert os.path.exists(excel_file)
    md_file = basename + '.md'
    assert os.path.exists(md_file)

    png_file = basename + '.png'
    prepare_acceleration_stacked_plot(df, machine_config, output_path=png_file)
    assert os.path.exists(png_file)
