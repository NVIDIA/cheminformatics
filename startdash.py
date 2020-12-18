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

from nvidia.cheminformatics.utils.fileio import initialize_logfile
import os
import sys
import atexit
import logging

import logging
import warnings
import argparse

from datetime import datetime
from dask_cuda.local_cuda_cluster import cuda_visible_devices
from dask_cuda.utils import get_n_gpus

import rmm
import cupy
import dask_cudf
import dask

from dask_cuda import initialize, LocalCUDACluster
from dask.distributed import Client, LocalCluster

from nvidia.cheminformatics.workflow import CpuWorkflow, GpuWorkflow
from nvidia.cheminformatics.chembldata import ChEmblData
from nvidia.cheminformatics.interactive.chemvisualize import ChemVisualization
from nvidia.cheminformatics.utils.fileio import initialize_logfile, log_results

warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('nvidia.cheminformatics')
formatter = logging.Formatter(
    '%(asctime)s %(name)s [%(levelname)s]: %(message)s')

BATCH_SIZE = 5000

FINGER_PRINT_FILES = 'filter_*.h5'

client = None
cluster = None


@atexit.register
def closing():
    if cluster:
        cluster.close()
    if client:
        client.close()


class Launcher(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Nvidia Cheminformatics',
            usage='''
    start <command> [<args>]

Following commands are supported:
   cache      : Create cache
   analyze    : Start Jupyter notebook in a container

To start dash:
    ./start analyze

To create cache:
    ./start cache -p
''')

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def cache(self):
        """
        Create Cache
        """
        parser = argparse.ArgumentParser(description='Create cache')
        parser.add_argument('-c', '--cache_directory',
                            dest='cache_directory',
                            type=str,
                            default='./.cache_dir',
                            help='Location to create fingerprint cache')
        parser.add_argument('-d', '--debug',
                            dest='debug',
                            action='store_true',
                            default=False,
                            help='Show debug message')

        args = parser.parse_args(sys.argv[2:])

        if args.debug:
            logger.setLevel(logging.DEBUG)

        cluster = LocalCluster(dashboard_address=':9001',
                               n_workers=12,
                               threads_per_worker=4)
        client = Client(cluster)

        with client:
            task_start_time = datetime.now()

            if not os.path.exists(args.cache_directory):
                logger.info('Creating folder %s...' % args.cache_directory)
                os.makedirs(args.cache_directory)

            chem_data = ChEmblData()
            chem_data.save_fingerprints(
                os.path.join(args.cache_directory, FINGER_PRINT_FILES))

            logger.info('Fingerprint generated in (hh:mm:ss.ms) {}'.format(
                datetime.now() - task_start_time))

    def analyze(self):
        """
        Start analysis
        """
        parser = argparse.ArgumentParser(description='Analyze')

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

        parser.add_argument('-n', '--num_clusters',
                            dest='num_clusters',
                            type=int,
                            default=7,
                            help='Numer of clusters (KMEANS)')

        parser.add_argument('-c', '--cache_directory',
                            dest='cache_directory',
                            type=str,
                            default=None,
                            help='Location to pick fingerprint from')

        parser.add_argument('-m', '--n_mol',
                            dest='n_mol',
                            type=int,
                            default=100000,
                            help='Number of molecules for analysis. Use negative numbers for using the whole dataset.')

        parser.add_argument('--n_gpu',
                            dest='n_gpu',
                            type=int,
                            default=-1,
                            help='Number of GPUs to use')

        parser.add_argument('--n_cpu',
                            dest='n_cpu',
                            type=int,
                            default=12,
                            help='Number of CPU workers to use')

        parser.add_argument('-d', '--debug',
                            dest='debug',
                            action='store_true',
                            default=False,
                            help='Show debug message')

        args = parser.parse_args(sys.argv[2:])

        if args.debug:
            logger.setLevel(logging.DEBUG)

        initialize_logfile()

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
            if args.n_gpu == -1:
                n_gpu = get_n_gpus() - 1
            else:
                n_gpu = args.n_gpu

            CUDA_VISIBLE_DEVICES = cuda_visible_devices(1, range(n_gpu)).split(',')
            CUDA_VISIBLE_DEVICES = [int(x) for x in CUDA_VISIBLE_DEVICES]
            logger.info('Using GPUs {} ...'.format(CUDA_VISIBLE_DEVICES))

            cluster = LocalCUDACluster(protocol="ucx",
                                       dashboard_address=':9001',
                                       # TODO: automate visible device list
                                       CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
                                       enable_tcp_over_ucx=enable_tcp_over_ucx,
                                       enable_nvlink=enable_nvlink,
                                       enable_infiniband=enable_infiniband)
        else:
            logger.info('Using {} CPUs ...'.format(args.n_cpu))
            cluster = LocalCluster(dashboard_address=':9001',
                                   n_workers=args.n_cpu,
                                   threads_per_worker=2)

        client = Client(cluster)

        start_time = datetime.now()
        task_start_time = datetime.now()
        chem_data = ChEmblData()
        if args.cache_directory is None:
            logger.info('Reading molecules from database...')
            mol_df = chem_data.fetch_all_props(num_recs=args.n_mol,
                                               batch_size=BATCH_SIZE)
        else:
            hdf_path = os.path.join(args.cache_directory, FINGER_PRINT_FILES)
            logger.info('Reading molecules from %s...' % hdf_path)
            mol_df = dask.dataframe.read_hdf(hdf_path, 'fingerprints')

            if args.n_mol > 0:
                mol_df = mol_df.head(args.n_mol, compute=False, npartitions=-1)

        task_start_time = datetime.now()
        if not args.cpu:
            workflow = GpuWorkflow(client,
                                   pca_comps=args.pca_comps,
                                   n_clusters=args.num_clusters)
        else:
            workflow = CpuWorkflow(client,
                                   pca_comps=args.pca_comps,
                                   n_clusters=args.num_clusters)

        mol_df = workflow.execute(mol_df)

        if args.benchmark:
            if not args.cpu:
                mol_df = mol_df.compute()
                n_cpu, n_gpu = 0, args.n_gpu
            else:
                n_cpu, n_gpu = args.n_cpu, 0

            runtime = datetime.now() - task_start_time
            logger.info('Runtime workflow (hh:mm:ss.ms) {}'.format(runtime))
            log_results(task_start_time, 'gpu', 'workflow', runtime, n_cpu, n_gpu)

            runtime = datetime.now() - start_time
            logger.info('Runtime Total (hh:mm:ss.ms) {}'.format(runtime))
            log_results(task_start_time, 'gpu', 'total', runtime, n_cpu, n_gpu)
        else:

            logger.info("Starting interactive visualization...")
            v = ChemVisualization(
                    mol_df,
                    workflow,
                    gpu=not args.cpu)

            logger.info('navigate to https://localhost:5001')
            v.start('0.0.0.0', port=5001)


def main():
    Launcher()


if __name__ == '__main__':
    main()
