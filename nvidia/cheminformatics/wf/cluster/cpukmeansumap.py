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

import sklearn.decomposition
from dask_ml.cluster import KMeans as dask_KMeans
import umap

from . import BaseClusterWorkflow
from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores
from nvidia.cheminformatics.data import ClusterWfDAO
from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.utils.logger import MetricsLogger


logger = logging.getLogger(__name__)


class CpuKmeansUmap(BaseClusterWorkflow):

    def __init__(self,
                 n_molecules,
                 dao: ClusterWfDAO = ChemblClusterWfDao(),
                 n_pca=64,
                 n_clusters=7,
                 benchmark_file='./benchmark.csv',
                 benchmark=False):
        self.dao = dao
        self.n_molecules = n_molecules
        self.n_pca = n_pca
        self.n_clusters = n_clusters
        self.benchmark_file = benchmark_file
        self.benchmark=benchmark

    def cluster(self,
                df_molecular_embedding=None,
                cache_directory=None):
        """
        Generates UMAP transformation on Kmeans labels generated from
        molecular fingerprints.
        """

        logger.info("Executing CPU workflow...")

        if df_molecular_embedding is None:
            df_molecular_embedding = self.dao.fetch_molecular_embedding(
                self.n_molecules,
                cache_directory=cache_directory)

        df_molecular_embedding = df_molecular_embedding.persist()

        if self.n_pca:
            with MetricsLogger('pca', self.n_molecules) as ml:

                pca = sklearn.decomposition.PCA(n_components=self.n_pca)
                df_fingerprints = pca.fit_transform(df_molecular_embedding)

        else:
            df_fingerprints = df_molecular_embedding.copy()

        with MetricsLogger('kmeans',self.n_molecules,) as ml:

            kmeans_float = dask_KMeans(n_clusters=self.n_clusters)
            kmeans_float.fit(df_fingerprints)
            kmeans_labels = kmeans_float.predict(df_fingerprints)

            ml.metric_name='silhouette_score'
            ml.metric_func = batched_silhouette_scores
            ml.metric_func_args = (df_fingerprints, kmeans_labels)
            ml.metric_func_kwargs = {'on_gpu': False}

        with MetricsLogger('umap', self.n_molecules) as ml:
            umap_model = umap.UMAP()

            Xt = umap_model.fit_transform(df_fingerprints)
            # TODO: Use dask to distribute umap. https://github.com/dask/dask/issues/5229
            df_molecular_embedding = df_molecular_embedding.compute()

        df_molecular_embedding['x'] = Xt[:, 0]
        df_molecular_embedding['y'] = Xt[:, 1]
        df_molecular_embedding['cluster'] = kmeans_float.labels_

        return df_molecular_embedding
