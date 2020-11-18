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

import time
import atexit
import logging

import logging
import warnings
from argparse import Action, ArgumentParser

from datetime import datetime

import rmm
import cupy

from dask_cuda import initialize, LocalCUDACluster
from dask.distributed import Client, LocalCluster

from nvidia.cheminformatics.workflow import CpuWorkflow, GpuWorkflow
from nvidia.cheminformatics.chembldata import ChEmblData
from nvidia.cheminformatics.chemvisualize import ChemVisualization

warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('nv_chem_dash')
formatter = logging.Formatter(
        '%(asctime)s %(name)s [%(levelname)s]: %(message)s')

# Positive number for # of molecules to select and negative number for using
# all available molecules
MAX_MOLECULES=200000
BATCH_SIZE=5000

client = None
cluster = None


@atexit.register
def closing():
    if cluster:
        cluster.close()
    if client:
        client.close()


def init_arg_parser():
    """
    Constructs command-line argument parser definition.
    """
    parser = ArgumentParser(
        description='Nvidia Cheminfomatics')

    parser.add_argument('--cpu',
                        dest='cpu',
                        action='store_true',
                        default=False,
                        help='Use CPU')

    parser.add_argument('-b', '--benchmark',
                        dest='benchmark',
                        action='store_true',
                        default=False,
                        help='Execute for benchmark')

    parser.add_argument('-p', '--pca_comps',
                        dest='pca_comps',
                        type=int,
                        default=64,
                        help='Numer of PCA components')

    parser.add_argument('-c', '--clusters',
                        dest='clusters',
                        type=int,
                        default=7,
                        help='Numer of clusters(KMEANS)')

    return parser


if __name__=='__main__':

    arg_parser = init_arg_parser()
    args = arg_parser.parse_args()

    rmm.reinitialize(managed_memory=True)
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    enable_tcp_over_ucx = True
    enable_nvlink = False
    enable_infiniband = False

    logger.info('Starting dash cluster...')
    if not args.cpu:
        initialize.initialize(create_cuda_context=True,
                            enable_tcp_over_ucx=enable_tcp_over_ucx,
                            enable_nvlink=enable_nvlink,
                            enable_infiniband=enable_infiniband)
        cluster = LocalCUDACluster(protocol="ucx",
                                dashboard_address=':9001',
                                # TODO: Find a way to automate visible device
                                CUDA_VISIBLE_DEVICES=[0, 1],
                                enable_tcp_over_ucx=enable_tcp_over_ucx,
                                enable_nvlink=enable_nvlink,
                                enable_infiniband=enable_infiniband)
    else:
        cluster = LocalCluster(dashboard_address=':9001',
                            n_workers=12,
                            threads_per_worker=4)

    client = Client(cluster)

    start_time = datetime.now()
    task_start_time = datetime.now()
    chem_data = ChEmblData()
    mol_df = chem_data.fetch_all_props(num_recs=MAX_MOLECULES,
                                       batch_size=BATCH_SIZE)

    task_start_time = datetime.now()
    if not args.cpu:
        workflow = GpuWorkflow(client,
                               pca_comps=64,
                               n_clusters=7)
    else:
        workflow = CpuWorkflow(client,
                               pca_comps=64,
                               n_clusters=7)

    df_fingerprints = workflow.execute(mol_df)

    print(df_fingerprints.head())
    logger.info("Starting interactive visualization...")
    if args.benchmark:
        df_fingerprints.compute()

        logger.info('Runtime workflow (hh:mm:ss.ms) {}'.format(
            datetime.now() - task_start_time))
        logger.info('Runtime Total (hh:mm:ss.ms) {}'.format(
            datetime.now() - start_time))
    else:
        # start dash
        v = ChemVisualization(
                df_fingerprints.copy(),
                mol_df,
                workflow,
                gpu=not args.cpu)

        logger.info('navigate to https://localhost:5000')
        v.start('0.0.0.0')