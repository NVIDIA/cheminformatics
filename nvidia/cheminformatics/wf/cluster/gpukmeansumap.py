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
from . import BaseClusterWorkflow
from nvidia.cheminformatics.utils.singleton import Singleton
from nvidia.cheminformatics.config import Context
from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores, spearman_rho
from nvidia.cheminformatics.utils.distance import tanimoto_calculate
from nvidia.cheminformatics.data import ClusterWfDAO
from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.utils.logger import MetricsLogger

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


logger = logging.getLogger(__name__)


@singledispatch
def _gpu_cluster_wrapper(embedding, n_pca, self):
    return NotImplemented


@_gpu_cluster_wrapper.register(dask.dataframe.core.DataFrame)
def _(embedding, n_pca, self):
    embedding = dask_cudf.from_dask_dataframe(embedding)
    return self._cluster(embedding, n_pca)


@_gpu_cluster_wrapper.register(dask_cudf.core.DataFrame)
def _(embedding, n_pca, self):
    return self._cluster(embedding, n_pca)


@_gpu_cluster_wrapper.register(cudf.DataFrame)
def _(embedding, n_pca, self):
    embedding = dask_cudf.from_cudf(embedding,
                                    int(chunksize=embedding.shape * 0.1))
    return self._cluster(embedding, n_pca)


class GpuKmeansUmap(BaseClusterWorkflow, metaclass=Singleton):

    def __init__(self,
                 n_molecules: int = None,
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

        dask_client = Context().dask_client
        embedding = embedding.reset_index()

        # Before reclustering remove all columns that may interfere
        ids =  embedding['id']
        for col in ['x', 'y', 'cluster', 'id', 'filter_col', 'index', 'molregno']:
            if col in embedding.columns:
                embedding = embedding.drop([col], axis=1)

        other_props = IMP_PROPS + ADDITIONAL_FEILD
        # Tempraryly store columns not required during processesing
        prop_series = {}
        for col in other_props:
            if col in embedding.columns:
                prop_series[col] = embedding[col]
        if len(prop_series) > 0:
            embedding = embedding.drop(other_props, axis=1)

        if n_pca:
            with MetricsLogger('pca', self.n_molecules) as ml:
                pca = cuDaskPCA(client=dask_client, n_components=n_pca)
                embedding = pca.fit_transform(embedding)

        with MetricsLogger('kmeans', self.n_molecules) as ml:
            kmeans_cuml = cuDaskKMeans(client=dask_client,
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
                                    client=dask_client)
            Xt = umap_model.transform(embedding)

            ml.metric_name = 'spearman_rho'
            ml.metric_func = self._compute_spearman_rho
            ml.metric_func_args = (embedding, X_train, Xt)

        # Add back the column required for plotting and to correlating data
        # between re-clustering
        embedding['cluster'] = kmeans_labels
        embedding['x'] = Xt[0]
        embedding['y'] = Xt[1]
        embedding['id'] = ids

        # Add back the prop columns
        for col in other_props:
            embedding[col] = prop_series[col]

        return embedding

    def cluster(self, df_mol_embedding=None):

        logger.info("Executing GPU workflow...")

        if df_mol_embedding is None:
            self.n_molecules = Context().n_molecule

            cache_directory = Context().cache_directory

            df_mol_embedding = self.dao.fetch_molecular_embedding(
                self.n_molecules,
                cache_directory=cache_directory)

            df_mol_embedding = df_mol_embedding.persist()

        self.n_molecules = df_mol_embedding.compute().shape[0]
        self.df_embedding = _gpu_cluster_wrapper(df_mol_embedding,
                                                 self.pca_comps,
                                                 self)
        return self.df_embedding

    def re_cluster(self, filter_column, filter_values,
                   new_figerprints=None,
                   new_chembl_ids=None,
                   n_clusters=None):

        df_embedding = self.df_embedding
        if filter_values is not None:
            filter = df_embedding[filter_column].isin(filter_values)

            df_embedding['filter_col'] = filter
            df_embedding = df_embedding.query('filter_col == True')

        if n_clusters is not None:
            self.n_clusters = n_clusters

        self.df_embedding = _gpu_cluster_wrapper(df_embedding, None, self)

        return self.df_embedding
