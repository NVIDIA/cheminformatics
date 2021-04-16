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
from nvidia.cheminformatics.data.helper.chembldata import ADDITIONAL_FEILD, IMP_PROPS

from dask_ml.decomposition import PCA as dask_PCA
from dask_ml.cluster import KMeans as dask_KMeans
import dask.array
import dask
import umap
import pandas as pd
import sklearn.cluster

from . import BaseClusterWorkflow
from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores
from nvidia.cheminformatics.data import ClusterWfDAO
from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.utils.logger import MetricsLogger
from nvidia.cheminformatics.config import Context


logger = logging.getLogger(__name__)


class CpuKmeansUmap(BaseClusterWorkflow):

    def __init__(self,
                 n_molecules=None,
                 dao: ClusterWfDAO = ChemblClusterWfDao(),
                 pca_comps=64,
                 n_clusters=7,
                 seed=0):
        super(CpuKmeansUmap, self).__init__()

        self.dao = dao
        self.n_molecules = n_molecules
        self.n_pca = pca_comps
        self.n_clusters = n_clusters

        self.seed = seed
        self.context = Context()
        self.n_spearman = 5000
        self.n_silhouette = 500000

    def is_gpu_enabled(self):
        return False

    def cluster(self,
                df_molecular_embedding=None):
        logger.info("Executing CPU workflow...")
        cache_directory = self.context.cache_directory

        if df_molecular_embedding is None:
            self.n_molecules = self.context.n_molecule
            df_molecular_embedding = self.dao.fetch_molecular_embedding(
                self.n_molecules,
                cache_directory=cache_directory)

        ids = df_molecular_embedding['id']
        df_molecular_embedding = df_molecular_embedding.persist()
        # self.n_molecules = df_molecular_embedding.compute().shape[0]
        self.n_molecules = self.context.n_molecule

        for col in ['id', 'index', 'molregno']:
            if col in df_molecular_embedding.columns:
                df_molecular_embedding = df_molecular_embedding.drop([col], axis=1)

        other_props = IMP_PROPS + ADDITIONAL_FEILD
        df_molecular_embedding = df_molecular_embedding.drop(other_props, axis=1)

        if self.context.is_benchmark:
            molecular_embedding_sample, spearman_index = self._random_sample_from_arrays(
                df_molecular_embedding, n_samples=self.n_spearman)

        if self.n_pca:
            with MetricsLogger('pca', self.n_molecules) as ml:
                pca = dask_PCA(n_components=self.n_pca)
                df_embedding = pca.fit_transform(df_molecular_embedding.to_dask_array(lengths=True))
        else:
            df_embedding = df_molecular_embedding

        with MetricsLogger('kmeans', self.n_molecules,) as ml:

            # kmeans_float = dask_KMeans(n_clusters=self.n_clusters)
            kmeans_float = sklearn.cluster.KMeans(n_clusters=self.n_clusters)

            kmeans_float.fit(df_embedding)
            kmeans_labels = kmeans_float.labels_

            ml.metric_name = 'silhouette_score'
            ml.metric_func = batched_silhouette_scores
            ml.metric_func_kwargs = {}
            ml.metric_func_args = (None, None)
            if self.context.is_benchmark:
                (embedding_sample, kmeans_labels_sample), _ = self._random_sample_from_arrays(
                    df_embedding, kmeans_labels, n_samples=self.n_silhouette)
                ml.metric_func_args = (embedding_sample, kmeans_labels_sample)

        with MetricsLogger('umap', self.n_molecules) as ml:
            df_molecular_embedding = df_molecular_embedding.compute()
            umap_model = umap.UMAP()  # TODO: Use dask to distribute umap. https://github.com/dask/dask/issues/5229
            X_train = umap_model.fit_transform(df_embedding)

            ml.metric_name = 'spearman_rho'
            ml.metric_func = self._compute_spearman_rho
            ml.metric_func_args = (None, None)
            if self.context.is_benchmark:
                X_train_sample, _ = self._random_sample_from_arrays(
                    X_train, index=spearman_index)
                ml.metric_func_args = (molecular_embedding_sample, X_train_sample)

        df_molecular_embedding['x'] = X_train[:, 0]
        df_molecular_embedding['y'] = X_train[:, 1]
        # df_molecular_embedding['cluster'] = kmeans_labels.compute()
        df_molecular_embedding['cluster'] = kmeans_labels
        df_molecular_embedding['id'] = ids

        self.df_embedding = df_molecular_embedding
        return self.df_embedding
