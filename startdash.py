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
import multiprocessing

import logging
from datetime import datetime

from dask.distributed import Client, LocalCluster
import dask.dataframe as dd

import cudf
import cupy
import cuml

import numpy as np

import sklearn.cluster
import sklearn.decomposition
import umap

import chemvisualize

from nvidia.cheminformatics.chemutil import morgan_fingerprint
from nvidia.cheminformatics.chembldata import ChEmblData

import warnings
warnings.filterwarnings('ignore', 'Expected ')
warnings.simplefilter('ignore')

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('nv_chem_viz')
formatter = logging.Formatter(
        '%(asctime)s %(name)s [%(levelname)s]: %(message)s')

MAX_MOLECULES=1000
ENABLE_GPU = True
PCA_COMPONENTS = 64


if __name__=='__main__':

    chem_data = ChEmblData()
    # start dask cluster
    logger.info('Starting dash cluster...')
    cluster = LocalCluster(dashboard_address=':9001', n_workers=12)
    client = Client(cluster)

    logger.info('Fetching molecules from database...')
    mol_df = chem_data.fetch_props(1000)
    mol_df = mol_df.astype({'mol_id': np.float32})

    start = time.time()
    logger.info('Initializing Morgan fingerprints...')

    finger_prints = dd.from_pandas(
        mol_df, npartitions=multiprocessing.cpu_count() * 4).map_partitions(
            lambda df: df.apply(
                (lambda row: morgan_fingerprint(row.canonical_smiles, molregno=row.mol_id)),
                axis=1),
        meta=tuple).compute()

    logger.info(time.time() - start)

    start = time.time()
    logger.info('Copying data into cuDF...')
    # fp_arrays = cupy.stack(finger_prints).astype(np.float32)
    df_fingerprints = cudf.DataFrame(finger_prints)
    df_fingerprints.rename(columns={0: 'mol_id'}, inplace=True)
    logger.info(time.time() - start)

    # prepare one set of clusters
    if PCA_COMPONENTS:
        task_start_time = datetime.now()
        if ENABLE_GPU:
            pca = cuml.PCA(n_components=PCA_COMPONENTS)
        else:
            pca = sklearn.decomposition.PCA(n_components=PCA_COMPONENTS)

        df_fingerprints = pca.fit_transform(df_fingerprints)
        logger.info('Runtime PCA time (hh:mm:ss.ms) {}'.format(
            datetime.now() - task_start_time))
    else:
        pca = False
        logger.info('PCA has been skipped')

    task_start_time = datetime.now()
    n_clusters = 7
    if ENABLE_GPU:
        kmeans_float = cuml.KMeans(n_clusters=n_clusters)
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