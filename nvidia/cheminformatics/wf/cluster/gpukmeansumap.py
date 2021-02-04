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
from nvidia.cheminformatics.config import Context

import numpy

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

from . import BaseClusterWorkflow
from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores, spearman_rho
from nvidia.cheminformatics.utils.distance import tanimoto_calculate
from nvidia.cheminformatics.data import ClusterWfDAO
from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.utils.logger import MetricsLogger

logger = logging.getLogger(__name__)




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


class GpuKmeansUmap(BaseClusterWorkflow):

    def __init__(self,
                 n_molecules: int,
                 dao: ClusterWfDAO = ChemblClusterWfDao(),
                 pca_comps=64,
                 n_clusters=7,
                 seed=0):

        self.dao = dao
        self.n_molecules = n_molecules
        self.pca_comps = pca_comps
        self.n_clusters = n_clusters

        self.df_embedding = None
        self.seed = seed
        self.n_spearman = 5000

    def _compute_spearman_rho(self, embedding, X_train, Xt):
        n_indexes = min(self.n_spearman, X_train.shape[0])
        numpy.random.seed(self.seed)
        indexes = numpy.random.choice(numpy.array(range(X_train.shape[0])),
                                      size=n_indexes,
                                      replace=False)
        fp_sample = cupy.fromDlpack(embedding.compute().to_dlpack())[indexes]
        Xt_sample = cupy.fromDlpack(Xt.compute().to_dlpack())[indexes]

        dist_array_tani = tanimoto_calculate(fp_sample, calc_distance=True)
        dist_array_eucl = pairwise_distances(Xt_sample)

        return spearman_rho(dist_array_tani, dist_array_eucl, top_k=100)

    def _cluster(self, embedding, n_pca):
        """
        Generates UMAP transformation on Kmeans labels generated from
        molecular fingerprints.
        """

        client = Context().dask_client

        # Before reclustering remove all columns that may interfere
        for col in ['x', 'y', 'cluster', 'id', 'filter_col']:
            if col in embedding.columns:
                embedding = embedding.drop([col], axis=1)

        if n_pca:
            with MetricsLogger('pca', self.n_molecules) as ml:
                pca = cuDaskPCA(client=client, n_components=n_pca)
                embedding = pca.fit_transform(embedding)

        with MetricsLogger('kmeans', self.n_molecules) as ml:
            kmeans_cuml = cuDaskKMeans(client=client,
                                       n_clusters=self.n_clusters)
            kmeans_cuml.fit(embedding)
            kmeans_labels = kmeans_cuml.predict(embedding)

            ml.metric_name = 'silhouette_score'
            ml.metric_func = batched_silhouette_scores
            ml.metric_func_args = (embedding, kmeans_labels)
            ml.metric_func_kwargs = {'on_gpu': True,  'seed': self.seed}

        with MetricsLogger('umap', self.n_molecules) as ml:
            X_train = embedding.compute()

            local_model = cuUMAP()
            local_model.fit(X_train)

            umap_model = cuDaskUMAP(local_model,
                                    n_neighbors=100,
                                    a=1.0,
                                    b=1.0,
                                    learning_rate=1.0,
                                    client=client)
            Xt = umap_model.transform(embedding)

            ml.metric_name = 'spearman_rho'
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

        self.n_molecules = df_molecular_embedding.compute().shape[0]
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
            self.n_molecules = len(self.chembl_ids)

        self.df_embedding = self._cluster_wrapper(self.df_embedding)
        return self.df_embedding
