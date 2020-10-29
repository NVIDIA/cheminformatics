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
import logging

import logging
from datetime import datetime

from dask import bag
from dask.distributed import Client, LocalCluster

import cudf
import cupy
import cuml
import rmm

from cuml.dask.decomposition import PCA
from cuml.dask.cluster import KMeans

import sklearn.cluster
import sklearn.decomposition
import umap

import chemvisualize

from nvidia.cheminformatics.chemutil import morgan_fingerprint
from nvidia.cheminformatics.chembldata import ChEmblData, fetch_all_props

import warnings
warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('nv_chem_viz')
formatter = logging.Formatter(
        '%(asctime)s %(name)s [%(levelname)s]: %(message)s')

MAX_MOLECULES=2000
ENABLE_GPU = True
PCA_COMPONENTS = 64


if __name__=='__main__':

    rmm.reinitialize(managed_memory=True)
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    enable_tcp_over_ucx = True
    enable_nvlink = False
    enable_infiniband = False

    # initialize.initialize(create_cuda_context=True,
    #                     enable_tcp_over_ucx=enable_tcp_over_ucx,
    #                     enable_nvlink=enable_nvlink,
    #                     enable_infiniband=enable_infiniband)
    # cluster = LocalCUDACluster(protocol="ucx",
    #                         dashboard_address=':9001',
    #                         # We use the number of available GPUs minus device=0
    #                         # (the client) which is oversubscribed w/ UVM
    #                         CUDA_VISIBLE_DEVICES=[0, 1],
    #                         enable_tcp_over_ucx=enable_tcp_over_ucx,
    #                         enable_nvlink=enable_nvlink,
    #                         enable_infiniband=enable_infiniband)
    # client = Client(cluster)

    logger.info('Starting dash cluster...')
    cluster = LocalCluster(dashboard_address=':9001', n_workers=12)
    client = Client(cluster)

    start = time.time()
    chem_data = ChEmblData()

    logger.info('Fetching molecules from database for fingerprints...')
    mol_df = fetch_all_props()

    logger.info('Generating fingerprints...')
    result = mol_df.map_partitions(
            lambda df: df.apply(
                (lambda row: morgan_fingerprint(row.canonical_smiles)),
                axis=1),
        meta=tuple).compute()

    logger.info(time.time() - start)

    # # results = results.compute()
    # logger.info('Copying data into dask_cudf...')

    # df_fingerprints = dask_cudf.from_dask_dataframe(results.to_dataframe())
    # df_fingerprints = df_fingerprints.compute()
    # print(df_fingerprints)
    # print(dir(df_fingerprints))
    # print(type(df_fingerprints))

    if True:
        import sys
        sys.exit()

    # prepare one set of clusters
    if PCA_COMPONENTS:
        logger.info('PCA...')
        task_start_time = datetime.now()
        if ENABLE_GPU:
            pca = PCA(n_components=PCA_COMPONENTS)
        else:
            pca = sklearn.decomposition.PCA(n_components=PCA_COMPONENTS)

        logger.info('PCA fit_transform...')
        df_fingerprints = pca.fit_transform(df_fingerprints)
        logger.info('Runtime PCA time (hh:mm:ss.ms) {}'.format(
            datetime.now() - task_start_time))
    else:
        pca = False
        logger.info('PCA has been skipped')


    task_start_time = datetime.now()
    n_clusters = 7
    logger.info('KMeans...')
    if ENABLE_GPU:
        kmeans_float = KMeans(client=client, n_clusters=n_clusters)
    else:
        kmeans_float = sklearn.cluster.KMeans(n_clusters=n_clusters)
    kmeans_float.fit(df_fingerprints)
    logger.info('Runtime Kmeans time (hh:mm:ss.ms) {}'.format(
        datetime.now() - task_start_time))



    # UMAP
    task_start_time = datetime.now()
    if ENABLE_GPU:
        umap = cuml.UMAP(n_neighbors=100,
                    a=1.0,
                    b=1.0,
                    learning_rate=1.0)
    else:
        umap = umap.UMAP()

    Xt = umap.fit_transform(df_fingerprints)
    logger.info('Runtime UMAP time (hh:mm:ss.ms) {}'.format(
        datetime.now() - task_start_time))

    if ENABLE_GPU:
        df_fingerprints.add_column('x', Xt[0].to_array())
        df_fingerprints.add_column('y', Xt[1].to_array())
        df_fingerprints.add_column('cluster', kmeans_float.labels_)
    else:
        df_fingerprints['x'] = Xt[:,0]
        df_fingerprints['y'] = Xt[:,1]
        df_fingerprints['cluster'] = kmeans_float.labels_

    # start dash
    v = chemvisualize.ChemVisualization(
        df_fingerprints.copy(),
        n_clusters,
        cudf.from_pandas(mol_df),
        enable_gpu=ENABLE_GPU,
        pca_model=PCA_COMPONENTS)

    logger.info('navigate to https://localhost:5000')
    v.start('0.0.0.0')