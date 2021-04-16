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
from functools import singledispatch
from nvidia.cheminformatics.utils.singleton import Singleton
from nvidia.cheminformatics.wf.cluster import BaseClusterWorkflow
from typing import List

import cupy
import cudf
import dask
import pandas
import dask_cudf

from cuml import SparseRandomProjection, KMeans

from nvidia.cheminformatics.utils.logger import MetricsLogger
from nvidia.cheminformatics.data import ClusterWfDAO
from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.config import Context
from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores


logger = logging.getLogger(__name__)


@singledispatch
def _gpu_random_proj_wrapper(embedding, self):
    return NotImplemented


@_gpu_random_proj_wrapper.register(dask.dataframe.core.DataFrame)
def _(embedding, self):
    logger.info('Converting from dask.dataframe.core.DataFrame...')
    embedding = embedding.compute()
    return _gpu_random_proj_wrapper(embedding, self)


@_gpu_random_proj_wrapper.register(dask_cudf.core.DataFrame)
def _(embedding, self):
    logger.info('Converting from dask_cudf.core.DataFrame...')
    embedding = embedding.compute()
    return _gpu_random_proj_wrapper(embedding, self)


@_gpu_random_proj_wrapper.register(pandas.DataFrame)
def _(embedding, self):
    logger.info('Converting from pandas.DataFrame...')
    embedding = cudf.from_pandas(embedding)
    return _gpu_random_proj_wrapper(embedding, self)


@_gpu_random_proj_wrapper.register(cudf.DataFrame)
def _(embedding, self):
    return self._cluster(embedding)


class GpuWorkflowRandomProjection(BaseClusterWorkflow, metaclass=Singleton):

    def __init__(self,
                 n_molecules: int = None,
                 dao: ClusterWfDAO = ChemblClusterWfDao(),
                 n_clusters=7,
                 seed=0):
        super(GpuWorkflowRandomProjection, self).__init__()

        self.dao = dao
        self.n_molecules = n_molecules
        self.n_clusters = n_clusters
        self.pca = None
        self.seed = seed
        self.n_silhouette = 500000
        self.context = Context()
        self.srp_embedding = SparseRandomProjection(n_components=2)

    def rand_jitter(self, arr):
        """
        Introduces random displacements to spread the points
        """
        stdev = .023 * cupy.subtract(cupy.max(arr), cupy.min(arr))
        for i in range(arr.shape[1]):
            rnd = cupy.multiply(cupy.random.randn(len(arr)), stdev)
            arr[:, i] = cupy.add(arr[:, i], rnd)

        return arr

    def _cluster(self, embedding):
        logger.info('Computing cluster...')
        embedding = embedding.reset_index()
        n_molecules = embedding.shape[0]

        # Before reclustering remove all columns that may interfere
        embedding, prop_series = self._remove_non_numerics(embedding)

        with MetricsLogger('random_proj', n_molecules) as ml:
            srp = self.srp_embedding.fit_transform(embedding.values)

            ml.metric_name = 'spearman_rho'
            ml.metric_func = self._compute_spearman_rho
            ml.metric_func_args = (embedding, embedding, srp)

        with MetricsLogger('kmeans', n_molecules) as ml:
            kmeans_cuml = KMeans(n_clusters=self.n_clusters)
            kmeans_cuml.fit(srp)
            kmeans_labels = kmeans_cuml.predict(srp)

            ml.metric_name = 'silhouette_score'
            ml.metric_func = batched_silhouette_scores
            ml.metric_func_kwargs = {}
            ml.metric_func_args = (None, None)
            if self.context.is_benchmark:
                (srp_sample, kmeans_labels_sample), _ = self._random_sample_from_arrays(
                    srp, kmeans_labels, n_samples=self.n_silhouette)
                ml.metric_func_args = (srp_sample, kmeans_labels_sample)

        # Add back the column required for plotting and to correlating data
        # between re-clustering
        srp = self.rand_jitter(srp)
        embedding['cluster'] = kmeans_labels
        embedding['x'] = srp[:, 0]
        embedding['y'] = srp[:, 1]

        # Add back the prop columns
        for col in prop_series.keys():
            embedding[col] = prop_series[col]

        return embedding

    def cluster(self, df_mol_embedding=None):
        logger.info("Executing GPU workflow...")

        if df_mol_embedding is None:
            self.n_molecules = self.context.n_molecule

            df_mol_embedding = self.dao.fetch_molecular_embedding(
                self.n_molecules,
                cache_directory=self.context.cache_directory)
            df_mol_embedding = df_mol_embedding.persist()

        self.df_embedding = _gpu_random_proj_wrapper(df_mol_embedding, self)
        return self.df_embedding

    def recluster(self,
                  filter_column=None,
                  filter_values=None,
                  n_clusters=None):

        if filter_values is not None:
            self.df_embedding['filter_col'] = self.df_embedding[filter_column].isin(filter_values)
            self.df_embedding = self.df_embedding.query('filter_col == True')

        if n_clusters is not None:
            self.n_clusters = n_clusters

        self.df_embedding = _gpu_random_proj_wrapper(self.df_embedding, self)
        return self.df_embedding

    def add_molecules(self, chemblids: List):

        chem_mol_map = {row[0]: row[1] for row in self.dao.fetch_id_from_chembl(chemblids)}
        molregnos = list(chem_mol_map.keys())

        self.df_embedding['id_exists'] = self.df_embedding['id'].isin(molregnos)

        ldf = self.df_embedding.query('id_exists == True')
        if hasattr(ldf, 'compute'):
            ldf = ldf.compute()

        self.df_embedding = self.df_embedding.drop(['id_exists'], axis=1)
        missing_mol = set(molregnos).difference(ldf['id'].to_array())

        chem_mol_map = {id: chem_mol_map[id] for id in missing_mol}
        missing_molregno = chem_mol_map.keys()
        if len(missing_molregno) > 0:
            new_fingerprints = self.dao.fetch_molecular_embedding_by_id(missing_molregno)
            new_fingerprints = new_fingerprints.compute()

            self.df_embedding = self._remove_ui_columns(self.df_embedding)
            self.df_embedding = self.df_embedding.append(new_fingerprints)

        return chem_mol_map, molregnos, self.df_embedding
