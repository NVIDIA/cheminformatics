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
from nvidia.cheminformatics.config import Context

import sklearn.decomposition
from dask_ml.cluster import KMeans as dask_KMeans
import umap
import numpy
import cupy
from cuml.metrics import pairwise_distances

from . import BaseClusterWorkflow
from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores, spearmanr
from nvidia.cheminformatics.utils.distance import tanimoto_calculate
from nvidia.cheminformatics.data import ClusterWfDAO
from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.utils.logger import MetricsLogger


logger = logging.getLogger(__name__)

class CpuKmeansUmap(BaseClusterWorkflow):

    def __init__(self,
                 n_molecules = None,
                 dao: ClusterWfDAO = ChemblClusterWfDao(),
                 n_pca=64,
                 n_clusters=7,
                 seed=0):
        self.dao = dao
        self.n_molecules = n_molecules
        self.n_pca = n_pca
        self.n_clusters = n_clusters

        self.seed = seed
        self.n_spearman = 5000

    def is_gpu_enabled(self):
        return False

    def _compute_spearman_rho(self, mol_df, X_train):
        n_indexes = min(self.n_spearman, X_train.shape[0])
        numpy.random.seed(self.seed)
        indexes = numpy.random.choice(numpy.array(range(X_train.shape[0])),
                                      size=n_indexes,
                                      replace=False)

        fp_sample = cupy.array(mol_df.iloc[indexes])
        Xt_sample = cupy.array(X_train[indexes])

        dist_array_tani = tanimoto_calculate(fp_sample, calc_distance=True)
        dist_array_eucl = pairwise_distances(Xt_sample)
        return cupy.nanmean(spearmanr(dist_array_tani, dist_array_eucl, top_k=100))

    def cluster(self,
                df_molecular_embedding=None,
                cache_directory=None):

        logger.info("Executing CPU workflow...")

        if df_molecular_embedding is None:
            self.n_molecules = Context().n_molecule
            df_molecular_embedding = self.dao.fetch_molecular_embedding(
                self.n_molecules,
                cache_directory=cache_directory)

        ids =  df_molecular_embedding['id']
        df_molecular_embedding = df_molecular_embedding.persist()
        self.n_molecules = df_molecular_embedding.compute().shape[0]

        for col in ['id', 'index', 'molregno']:
            if col in df_molecular_embedding.columns:
                df_molecular_embedding = df_molecular_embedding.drop([col], axis=1)

        other_props = IMP_PROPS + ADDITIONAL_FEILD
        df_molecular_embedding = df_molecular_embedding.drop(other_props, axis=1)

        if self.n_pca:
            with MetricsLogger('pca', self.n_molecules) as ml:

                pca = sklearn.decomposition.PCA(n_components=self.n_pca)
                df_fingerprints = pca.fit_transform(df_molecular_embedding)

        else:
            df_fingerprints = df_molecular_embedding.copy()

        with MetricsLogger('kmeans', self.n_molecules,) as ml:

            kmeans_float = dask_KMeans(n_clusters=self.n_clusters)
            kmeans_float.fit(df_fingerprints)
            kmeans_labels = kmeans_float.predict(df_fingerprints)

            ml.metric_name = 'silhouette_score'
            ml.metric_func = batched_silhouette_scores
            ml.metric_func_args = (df_fingerprints, kmeans_labels)
            ml.metric_func_kwargs = {'on_gpu': False, 'seed': self.seed}

        with MetricsLogger('umap', self.n_molecules) as ml:
            umap_model = umap.UMAP()

            X_train = umap_model.fit_transform(df_fingerprints)
            # TODO: Use dask to distribute umap. https://github.com/dask/dask/issues/5229
            # Sample to calculate spearman's rho
            # Currently this converts indexes to
            df_molecular_embedding = df_molecular_embedding.compute()

            ml.metric_name = 'spearman_rho'
            ml.metric_func = self._compute_spearman_rho
            ml.metric_func_args = (df_molecular_embedding, X_train)

        df_molecular_embedding['x'] = X_train[:, 0]
        df_molecular_embedding['y'] = X_train[:, 1]
        df_molecular_embedding['cluster'] = kmeans_float.labels_
        df_molecular_embedding['id'] = ids

        self.df_embedding = df_molecular_embedding
        return self.df_embedding
