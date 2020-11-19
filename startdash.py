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
import sys
import atexit
import logging

import logging
import warnings
import argparse

from datetime import datetime

import rmm
import cupy
import dask_cudf
import dask

from dask_cuda import initialize, LocalCUDACluster
from dask.distributed import Client, LocalCluster

from nvidia.cheminformatics.workflow import CpuWorkflow, GpuWorkflow
from nvidia.cheminformatics.chembldata import ChEmblData
from nvidia.cheminformatics.chemvisualize import ChemVisualization

warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('nvidia.cheminformatics')
formatter = logging.Formatter(
    '%(asctime)s %(name)s [%(levelname)s]: %(message)s')

# Positive number for # of molecules to select and negative number for using
# all available molecules
MAX_MOLECULES = 100000
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
                            help='Numer of clusters(KMEANS)')

        parser.add_argument('-c', '--cache_directory',
                            dest='cache_directory',
                            type=str,
                            default=None,
                            help='Location to pick fingerprint from')

        parser.add_argument('-d', '--debug',
                            dest='debug',
                            action='store_true',
                            default=False,
                            help='Show debug message')

        args = parser.parse_args(sys.argv[2:])

        if args.debug:
            logger.setLevel(logging.DEBUG)

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
                                       # TODO: automate visible device list
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
        if args.cache_directory is None:
            logger.info('Reading molecules from database...')
            mol_df = chem_data.fetch_all_props(num_recs=MAX_MOLECULES,
                                               batch_size=BATCH_SIZE)
        else:
            hdf_path = os.path.join(args.cache_directory, FINGER_PRINT_FILES)
            logger.info('Reading molecules from %s...' % hdf_path)
            mol_df = dask.dataframe.read_hdf(hdf_path, 'fingerprints')

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
            print(mol_df.head())

            logger.info('Runtime workflow (hh:mm:ss.ms) {}'.format(
                datetime.now() - task_start_time))
            logger.info('Runtime Total (hh:mm:ss.ms) {}'.format(
                datetime.now() - start_time))
        else:

            logger.info("Starting interactive visualization...")
            print('mol_df', mol_df.shape)
            # v = ChemVisualization(
            #         mol_df,
            #         workflow,
            #         gpu=not args.cpu)

            # logger.info('navigate to https://localhost:5000')
            # v.start('0.0.0.0')


def main():
    Launcher()


if __name__ == '__main__':
    main()
