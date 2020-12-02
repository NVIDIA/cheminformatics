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

import logging
from nvidia.cheminformatics.utils.fileio import log_results

from datetime import datetime

import dask_cudf
import dask_ml

from cuml.manifold import UMAP as cuUMAP
from cuml.dask.decomposition import PCA as cuDaskPCA
from cuml.dask.cluster import KMeans as cuDaskKMeans
from cuml.dask.manifold import UMAP as cuDaskUMAP

import sklearn.cluster
import sklearn.decomposition
import umap


logger = logging.getLogger(__name__)


class CpuWorkflow:

    def __init__(self,
                 client,
                 pca_comps=64,
                 n_clusters=7,):
        self.client = client
        self.pca_comps = pca_comps
        self.n_clusters = n_clusters

    def execute(self, mol_df):
        logger.info("Executing CPU workflow...")

        mol_df = mol_df.persist()

        logger.info('PCA...')
        n_cpu = len(self.client.cluster.workers)
        print('WORKERS', n_cpu)

        if self.pca_comps:
            task_start_time = datetime.now()
            pca = sklearn.decomposition.PCA(n_components=self.pca_comps)
            df_fingerprints = pca.fit_transform(mol_df)
            runtime = datetime.now() - task_start_time
            logger.info('### Runtime PCA time (hh:mm:ss.ms) {}'.format(runtime))
            log_results(task_start_time, 'cpu', 'pca', runtime, n_cpu=n_cpu)
        else:
            df_fingerprints = mol_df.copy()

        logger.info('KMeans...')
        task_start_time = datetime.now()
        # kmeans_float = sklearn.cluster.KMeans(n_clusters=self.n_clusters)
        kmeans_float = dask_ml.cluster.KMeans(n_clusters=self.n_clusters)
        kmeans_float.fit(df_fingerprints)

        runtime = datetime.now() - task_start_time
        logger.info('### Runtime Kmeans time (hh:mm:ss.ms) {}'.format(runtime))
        log_results(task_start_time, 'cpu', 'kmeans', runtime, n_cpu=n_cpu)

        logger.info('UMAP...')
        task_start_time = datetime.now()
        umap_model = umap.UMAP()

        Xt = umap_model.fit_transform(df_fingerprints)
        # TODO: Use dask to distribute umap. https://github.com/dask/dask/issues/5229
        mol_df = mol_df.compute()
        mol_df['x'] = Xt[:, 0]
        mol_df['y'] = Xt[:, 1]
        mol_df['cluster'] = kmeans_float.labels_
        runtime = datetime.now() - task_start_time
        logger.info('### Runtime UMAP time (hh:mm:ss.ms) {}'.format(runtime))
        log_results(task_start_time, 'cpu', 'umap', runtime, n_cpu=n_cpu)

        return mol_df


class GpuWorkflow:

    def __init__(self,
                 client,
                 pca_comps=64,
                 n_clusters=7,):
        self.client = client
        self.pca_comps = pca_comps
        self.n_clusters = n_clusters

    def re_cluster(self, gdf, new_figerprints=None, new_chembl_ids=None):

        # Before reclustering remove all columns that may interfere
        # ids = gdf['id']
        # chembl_ids = gdf['chembl_id']

        # gdf.drop(['x', 'y', 'cluster', 'id', 'chembl_id'],
        #          axis=1, inplace=True)
        gdf = gdf.drop(['x', 'y', 'cluster', 'id', 'filter_col'], axis=1)

        if new_figerprints is not None and new_chembl_ids is not None:
            # Add new figerprints and chEmblIds before reclustering
            if self.pca_comps:
                new_figerprints = self.pca.transform(new_figerprints)

            fp_df = cudf.DataFrame(new_figerprints,
                                    index=[idx for idx in range(
                                        self.orig_df.shape[0], self.orig_df.shape[0] + len(new_figerprints))],
                                    columns=gdf.columns)

            gdf = gdf.append(fp_df, ignore_index=True)
            # Update original dataframe for it to work on reload
            fp_df['id'] = fp_df.index
            self.orig_df = self.orig_df.append(fp_df, ignore_index=True)
            chembl_ids = chembl_ids.append(
                cudf.Series(new_chembl_ids), ignore_index=True)
            ids = ids.append(fp_df['id'], ignore_index=True), 'id', 'chembl_id'
            self.chembl_ids.extend(new_chembl_ids)

            del fp_df

        kmeans_cuml = cuDaskKMeans(client=self.client, 
                                   n_clusters=self.n_clusters)
        kmeans_cuml.fit(gdf)
        kmeans_labels = kmeans_cuml.predict(gdf)

        local_model = cuUMAP()
        X_train = gdf.compute()
        local_model.fit(X_train)

        umap_model = cuDaskUMAP(local_model,
                                 n_neighbors=100,
                                 a=1.0,
                                 b=1.0,
                                 learning_rate=1.0,
                                 client=self.client)
        Xt = umap_model.transform(gdf)


        # Add back the column required for plotting and to correlating data
        # between re-clustering
        gdf['x'] = Xt[0]
        gdf['y'] = Xt[1]
        gdf['cluster'] = kmeans_labels
        gdf['id'] = gdf.index

        # gdf.add_column('y', Xt[1].to_array())
        # gdf.add_column('cluster', kmeans_labels)

        # gdf.add_column('chembl_id', chembl_ids)
        # gdf.add_column('id', ids)
        print('Workflow: recluster ', gdf.head())
        return gdf


    def execute(self, mol_df):
        logger.info("Executing GPU workflow...")
        n_gpu = len(self.client.cluster.workers)
        print('WORKERS', n_gpu)

        mol_df = dask_cudf.from_dask_dataframe(mol_df)
        mol_df = mol_df.persist()

        logger.info('PCA...')
        if self.pca_comps:
            task_start_time = datetime.now()
            pca = cuDaskPCA(client=self.client, n_components=self.pca_comps)
            df_fingerprints = pca.fit_transform(mol_df)
            runtime = datetime.now() - task_start_time
            logger.info('### Runtime PCA time (hh:mm:ss.ms) {}'.format(runtime))
            log_results(task_start_time, 'gpu', 'pca', runtime, n_gpu=n_gpu)
        else:
            df_fingerprints = mol_df.copy()

        logger.info('KMeans...')
        task_start_time = datetime.now()
        kmeans_cuml = cuDaskKMeans(client=self.client, n_clusters=self.n_clusters)
        kmeans_cuml.fit(df_fingerprints)
        kmeans_labels = kmeans_cuml.predict(df_fingerprints)
        runtime = datetime.now() - task_start_time
        logger.info('### Runtime Kmeans time (hh:mm:ss.ms) {}'.format(runtime))
        log_results(task_start_time, 'gpu', 'kmeans', runtime, n_gpu=n_gpu)

        logger.info('UMAP...')
        task_start_time = datetime.now()
        local_model = cuUMAP()
        X_train = df_fingerprints.compute()
        local_model.fit(X_train)

        umap_model = cuDaskUMAP(local_model,
                                 n_neighbors=100,
                                 a=1.0,
                                 b=1.0,
                                 learning_rate=1.0,
                                 client=self.client)
        Xt = umap_model.transform(df_fingerprints)

        mol_df['x'] = Xt[0]
        mol_df['y'] = Xt[1]
        mol_df['cluster'] = kmeans_labels
        runtime = datetime.now() - task_start_time
        logger.info('### Runtime UMAP time (hh:mm:ss.ms) {}'.format(runtime))
        log_results(task_start_time, 'gpu', 'umap', runtime, n_gpu=n_gpu)

        print(mol_df.head())

        return mol_df