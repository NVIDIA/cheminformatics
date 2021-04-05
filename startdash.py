#!/usr/bin/env python3
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

from dask.distributed import Client, LocalCluster

from nvidia.cheminformatics.utils.dask import initialize_cluster
from nvidia.cheminformatics.data.helper.chembldata import ChEmblData
from nvidia.cheminformatics.data.cluster_wf import FINGER_PRINT_FILES
from nvidia.cheminformatics.wf.cluster.cpukmeansumap import CpuKmeansUmap
from nvidia.cheminformatics.wf.cluster.gpukmeansumap import GpuKmeansUmap
from nvidia.cheminformatics.interactive.chemvisualize import ChemVisualization
from nvidia.cheminformatics.utils.logger import initialize_logfile, log_results
from nvidia.cheminformatics.config import Context
from nvidia.cheminformatics.fingerprint import MorganFingerprint, Embeddings

warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('nvidia.cheminformatics')
formatter = logging.Formatter(
    '%(asctime)s %(name)s [%(levelname)s]: %(message)s')


client = None
cluster = None


@atexit.register
def closing():
    if cluster:
        cluster.close()
    if client:
        client.close()


class Launcher(object):
    """
    Application launcher. This class can execute the workflows in headless (for
    benchmarking and testing) and with UI.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Nvidia Cheminformatics',
            usage='''
    start <command> [<args>]

Following commands are supported:
   cache      : Create cache
   analyze    : Start Jupyter notebook in a container
   service    : Start in service mode
   grpc       : Start in grpc service

To start dash:
    ./start analyze

To create cache:
    ./start cache -p

To start dash:
    ./start service

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

        parser.add_argument('-ct', '--cache_type',
                            dest='cache_type',
                            type=str,
                            default='MorganFingerprint',
                            choices = ['MorganFingerprint','Embeddings'],
                            help='Type of data preprocessing (MorganFingerprint or Embeddings)')

        parser.add_argument('-c', '--cache_directory',
                            dest='cache_directory',
                            type=str,
                            default='./.cache_dir',
                            help='Location to create fingerprint cache')

        parser.add_argument('--batch_size',
                            dest='batch_size',
                            type=int,
                            default=100000,
                            help='Chunksize.')

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

        cluster = LocalCluster(dashboard_address=':9001',
                               n_workers=args.n_cpu,
                               threads_per_worker=4)
        client = Client(cluster)

        with client:
            task_start_time = datetime.now()

            if not os.path.exists(args.cache_directory):
                logger.info('Creating folder %s...' % args.cache_directory)
                os.makedirs(args.cache_directory)

            if (args.cache_type == 'MorganFingerprint'):
                prepocess_type = MorganFingerprint
            elif (args.cache_type == 'Embeddings'):
                prepocess_type = Embeddings

            chem_data = ChEmblData(fp_type=prepocess_type)
            chem_data.save_fingerprints(
                os.path.join(args.cache_directory, FINGER_PRINT_FILES), num_recs = -1,
                batch_size=args.batch_size)

            logger.info('Fingerprint generated in (hh:mm:ss.ms) {}'.format(
                datetime.now() - task_start_time))

    def service(self):
        """
        Start services
        """
        parser = argparse.ArgumentParser(description='Service')
        parser.add_argument('-d', '--debug',
                            dest='debug',
                            action='store_true',
                            default=False,
                            help='Show debug message')

        args = parser.parse_args(sys.argv[2:])

        if args.debug:
            logger.setLevel(logging.DEBUG)

        from waitress import serve
        from nvidia.cheminformatics.api import app

        context = Context()
        # port = context.get_config('plotly_port', 6000)
        port = 8081
        serve(app, host='0.0.0.0', port=port)

    def grpc(self):
        """
        Start services
        """
        parser = argparse.ArgumentParser(description='Service')
        parser.add_argument('-p', '--port',
                            dest='port',
                            type=int,
                            default=50051,
                            help='GRPC server Port')
        parser.add_argument('-d', '--debug',
                            dest='debug',
                            action='store_true',
                            default=False,
                            help='Show debug message')

        args = parser.parse_args(sys.argv[2:])

        if args.debug:
            logger.setLevel(logging.DEBUG)

        sys.path.insert(0, "generated")
        import grpc
        import similaritysampler_pb2_grpc
        from concurrent import futures
        from nvidia.cheminformatics.grpc import SimilaritySampler

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        similaritysampler_pb2_grpc.add_SimilaritySamplerServicer_to_server(SimilaritySampler(), server)
        server.add_insecure_port(f'[::]:{args.port}')
        server.start()
        server.wait_for_termination()


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
                            help='Number of PCA components')

        parser.add_argument('-n', '--num_clusters',
                            dest='num_clusters',
                            type=int,
                            default=7,
                            help='Numer of clusters')

        parser.add_argument('-c', '--cache_directory',
                            dest='cache_directory',
                            type=str,
                            default=None,
                            help='Location to pick fingerprint from')

        parser.add_argument('-m', '--n_mol',
                            dest='n_mol',
                            type=int,
                            default=10000,
                            help='Number of molecules for analysis. Use negative numbers for using the whole dataset.')

        parser.add_argument('--batch_size',
                            dest='batch_size',
                            type=int,
                            default=100000,
                            help='Chunksize.')

        parser.add_argument('-o', '--output_dir',
                            dest='output_dir',
                            default=".",
                            type=str,
                            help='Output directory for benchmark results')

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

        benchmark_file = os.path.join(args.output_dir, 'benchmark.csv')
        initialize_logfile(benchmark_file)

        client = initialize_cluster(not args.cpu,
                                    n_cpu=args.n_cpu,
                                    n_gpu=args.n_gpu)

        # Set the context
        context = Context()
        context.dask_client = client
        context.is_benchmark = args.benchmark
        context.benchmark_file = benchmark_file
        context.cache_directory = args.cache_directory
        context.n_molecule = args.n_mol
        context.batch_size = args.batch_size

        if args.cpu:
            context.compute_type = 'cpu'
        else:
            logger.info('Number of workers %d.', len(client.scheduler_info()['workers'].keys()))

        start_time = datetime.now()
        task_start_time = datetime.now()

        n_molecules = args.n_mol
        if not args.cpu:
            workflow = GpuKmeansUmap(n_molecules=n_molecules,
                                     pca_comps=args.pca_comps,
                                     n_clusters=args.num_clusters)
        else:
            workflow = CpuKmeansUmap(n_molecules=n_molecules,
                                     pca_comps=args.pca_comps,
                                     n_clusters=args.num_clusters)

        mol_df = workflow.cluster()

        if args.benchmark:
            workflow.compute_qa_matric()
            if not args.cpu:
                mol_df = mol_df.compute()
                n_workers = args.n_gpu
            else:
                n_workers = args.n_cpu

            n_molecules = workflow.n_molecules

            runtime = datetime.now() - task_start_time
            logger.info('Runtime workflow (hh:mm:ss.ms) {}'.format(runtime))
            log_results(task_start_time, context.compute_type, 'workflow',
                        runtime, n_molecules, n_workers, metric_name='',
                        metric_value='', benchmark_file=benchmark_file)

            runtime = datetime.now() - start_time
            logger.info('Runtime Total (hh:mm:ss.ms) {}'.format(runtime))
            log_results(task_start_time, context.compute_type, 'total',
                        runtime, n_molecules, n_workers, metric_name='',
                        metric_value='', benchmark_file=benchmark_file)
        else:
            port = context.get_config('plotly_port', 5000)

            logger.info("Starting interactive visualization...")
            v = ChemVisualization(workflow)

            logger.info('navigate to https://localhost: %s' % port)
            v.start('0.0.0.0', port=port)


def main():
    Launcher()


if __name__ == '__main__':
    main()
