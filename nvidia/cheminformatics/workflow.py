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

import numpy
import sklearn.decomposition
from dask_ml.cluster import KMeans as dask_KMeans
import umap

import cudf
import cupy
import dask
import dask_cudf
from functools import singledispatch

from cuml.manifold import UMAP as cuUMAP
from cuml.dask.decomposition import PCA as cuDaskPCA
from cuml.dask.cluster import KMeans as cuDaskKMeans
from cuml.dask.manifold import UMAP as cuDaskUMAP
from cuml.metrics import pairwise_distances

from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores, spearman_rho
from nvidia.cheminformatics.utils.distance import tanimoto_calculate
from nvidia.cheminformatics.data import ClusterWfDAO
from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.utils.fileio import MetricsLogger
from nvidia.cheminformatics.wf.cluster import BaseClusterWorkflow

logger = logging.getLogger(__name__)


class CpuWorkflow(BaseClusterWorkflow):

    def __init__(self,
                 client,
                 n_molecules,
                 dao: ClusterWfDAO = ChemblClusterWfDao(),
                 n_pca=64,
                 n_clusters=7,
                 benchmark_file='./benchmark.csv',
                 benchmark=False):
        self.client = client
        self.dao = dao
        self.n_molecules = n_molecules
        self.n_pca = n_pca
        self.n_clusters = n_clusters
        self.benchmark_file = benchmark_file
        self.benchmark=benchmark

    def cluster(self,
                df_molecular_embedding=None,
                cache_directory=None):

        logger.info("Executing CPU workflow...")

        if df_molecular_embedding is None:
            df_molecular_embedding = self.dao.fetch_molecular_embedding(
                self.n_molecules,
                cache_directory=cache_directory)

        df_molecular_embedding = df_molecular_embedding.persist()

        if self.n_pca:
            with MetricsLogger(self.client, 'pca', 'cpu',
                              self.benchmark_file, self.n_molecules,
                              benchmark=self.benchmark) as ml:

                pca = sklearn.decomposition.PCA(n_components=self.n_pca)
                df_fingerprints = pca.fit_transform(df_molecular_embedding)

        else:
            df_fingerprints = df_molecular_embedding.copy()

        with MetricsLogger(self.client, 'kmeans', 'cpu',
                          self.benchmark_file, self.n_molecules,
                          benchmark=self.benchmark) as ml:

            kmeans_float = dask_KMeans(n_clusters=self.n_clusters)
            kmeans_float.fit(df_fingerprints)
            kmeans_labels = kmeans_float.predict(df_fingerprints)

            ml.metric_name='silhouette_score'
            ml.metric_func = batched_silhouette_scores
            ml.metric_func_args = (df_fingerprints, kmeans_labels)
            ml.metric_func_kwargs = {'on_gpu': False}

        with MetricsLogger(self.client, 'umap', 'gpu',
                          self.benchmark_file, self.n_molecules,
                          benchmark=self.benchmark) as ml:
            umap_model = umap.UMAP()

            Xt = umap_model.fit_transform(df_fingerprints)
            # TODO: Use dask to distribute umap. https://github.com/dask/dask/issues/5229
            df_molecular_embedding = df_molecular_embedding.compute()

        df_molecular_embedding['x'] = Xt[:, 0]
        df_molecular_embedding['y'] = Xt[:, 1]
        df_molecular_embedding['cluster'] = kmeans_float.labels_

        return df_molecular_embedding


@singledispatch
def _gpu_cluster_wrapper(embedding, n_pca, self):
    return NotImplemented

@_gpu_cluster_wrapper.register(dask.dataframe.core.DataFrame)
def _(embedding, n_pca, self):
    embedding = dask_cudf.from_dask_dataframe(embedding)
    return self._cluster(embedding, n_pca)

@_gpu_cluster_wrapper.register(cudf.DataFrame)
def _(embedding, n_pca, self):
    embedding = dask_cudf.from_cudf(embedding)
    return self._cluster(embedding, n_pca)


class GpuWorkflow(BaseClusterWorkflow):

    def __init__(self,
                 n_molecules: int,
                 dao: ClusterWfDAO = ChemblClusterWfDao(),
                 client=None,
                 pca_comps=64,
                 n_clusters=7,
                 benchmark_file='./benchmark.csv',
                 benchmark=False):
        self.client = client
        self.dao = dao
        self.n_molecules = n_molecules
        self.pca_comps = pca_comps
        self.n_clusters = n_clusters
        self.benchmark_file = benchmark_file
        self.benchmark=benchmark

        self.df_embedding = None

    def _compute_spearman_rho(self, embedding, X_train, Xt):
        n_indexes = min(5000, X_train.shape[0])
        indexes = numpy.random.choice(numpy.array(range(X_train.shape[0])),
                                      size=n_indexes,
                                      replace=False)

        X_train_sample = cupy.fromDlpack(embedding.compute().to_dlpack())[indexes]
        Xt_sample = cupy.fromDlpack(Xt.compute().to_dlpack())[indexes]

        dist_array_tani = tanimoto_calculate(X_train_sample, calc_distance=True)
        dist_array_eucl = pairwise_distances(Xt_sample)

        return spearman_rho(dist_array_tani, dist_array_eucl).mean()

    def _cluster(self, embedding, n_pca):

        # Before reclustering remove all columns that may interfere
        for col in ['x', 'y', 'cluster', 'id', 'filter_col']:
            if col in embedding.columns:
                embedding = embedding.drop([col], axis=1)

        if n_pca:
            with MetricsLogger(self.client, 'pca', 'gpu',
                              self.benchmark_file, self.n_molecules,
                              benchmark=self.benchmark) as ml:
                pca = cuDaskPCA(client=self.client, n_components=n_pca)
                embedding = pca.fit_transform(embedding)

        with MetricsLogger(self.client, 'kmeans', 'gpu',
                          self.benchmark_file, self.n_molecules,
                          benchmark=self.benchmark) as ml:
            kmeans_cuml = cuDaskKMeans(client=self.client,
                                       n_clusters=self.n_clusters)
            kmeans_cuml.fit(embedding)
            kmeans_labels = kmeans_cuml.predict(embedding)

            ml.metric_name='silhouette_score'
            ml.metric_func = batched_silhouette_scores
            ml.metric_func_args = (embedding, kmeans_labels)
            ml.metric_func_kwargs = {'on_gpu': True}

        with MetricsLogger(self.client, 'umap', 'gpu',
                          self.benchmark_file, self.n_molecules,
                          benchmark=self.benchmark) as ml:
            X_train = embedding.compute()

            local_model = cuUMAP()
            local_model.fit(X_train)

            umap_model = cuDaskUMAP(local_model,
                                    n_neighbors=100,
                                    a=1.0,
                                    b=1.0,
                                    learning_rate=1.0,
                                    client=self.client)
            Xt = umap_model.transform(embedding)

            ml.metric_name='spearman_rho'
            ml.metric_func = self._compute_spearman_rho
            ml.metric_func_args = (embedding, X_train, Xt)

        # Add back the column required for plotting and to correlating data
        # between re-clustering
        embedding['x'] = Xt[0]
        embedding['y'] = Xt[1]
        embedding['cluster'] = kmeans_labels
        embedding['id'] = embedding.index

        return embedding

    def cluster(self,
                df_molecular_embedding=None,
                cache_directory=None):

        logger.info("Executing GPU workflow...")

        if df_molecular_embedding is None:
            df_molecular_embedding = self.dao.fetch_molecular_embedding(
                self.n_molecules,
                cache_directory=cache_directory)
            df_molecular_embedding = df_molecular_embedding.persist()

        self.df_embedding = _gpu_cluster_wrapper(df_molecular_embedding,
                                                 self.pca_comps,
                                                 self)
        return self.df_embedding

    def re_cluster(self, mol_df, gdf,
                   new_figerprints=None,
                   new_chembl_ids=None,
                   n_clusters=None):

        if n_clusters is not None:
            self.n_clusters = n_clusters

        # Before reclustering remove all columns that may interfere
        for col in ['x', 'y', 'cluster', 'id', 'filter_col']:
            if col in gdf.columns:
                gdf = gdf.drop([col], axis=1)

        if new_figerprints is not None and new_chembl_ids is not None:
            # Add new figerprints and chEmblIds before reclustering
            mol_df.append(new_figerprints, ignore_index=True)
            if self.pca_comps:
                new_figerprints = self.pca.transform(new_figerprints)

            fp_df = cudf.DataFrame(
                new_figerprints,
                index=[idx for idx in range(self.orig_df.shape[0],
                                            self.orig_df.shape[0] + len(new_figerprints))],
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

        self.df_embedding = self._cluster_wrapper(self.df_embedding)
        return self.df_embedding

