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

import rmm
from dask_cuda import initialize, LocalCUDACluster
from dask.distributed import Client

import dask_cudf
import cudf
import cupy
import cuml

from cuml.dask.decomposition import PCA
from cuml.dask.cluster import KMeans

import sklearn.cluster
import sklearn.decomposition
import umap

import chemvisualize
from nvidia.cheminformatics.chembldata import ChEmblData

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

    logger.info('Starting dash cluster...')
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
    client = Client(cluster)

    start = time.time()
    chem_data = ChEmblData()
    mol_df = chem_data.fetch_all_props(num_recs=100000, batch_size=10000)

    df_fingerprints = dask_cudf.from_dask_dataframe(mol_df)

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
