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

import math
import cudf
import cupy
import pandas
import numpy

import dask
import dask_cudf
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

import logging
logger = logging.getLogger(__name__)


def batched_silhouette_scores(embeddings, clusters, batch_size=5000, seed=0, downsample_size=500000, on_gpu=True):
    """Calculate silhouette score in batches on the CPU. Compatible with data on GPU or CPU

    Args:
        embeddings (cudf.DataFrame or cupy.ndarray): input features to clustering
        clusters (cudf.DataFrame or cupy.ndarray): cluster values for each data point
        batch_size (int, optional): Size for batching. Defaults to 5000.
        seed (int, optional): Random seed. Defaults to 0.
        downsample_size (int, optional): Limit on size of data used for silhouette score. Defaults to 500000.
        on_gpu (bool, optional): Input data is on GPU. Defaults to True.

    Returns:
        float: mean silhouette score from batches
    """

    if on_gpu:
        arraylib = cupy
        dflib = cudf
        AsArray = cupy.asnumpy
    else:
        arraylib = numpy
        dflib = pandas
        AsArray = numpy.asarray

    # Function to calculate results
    def _silhouette_scores(input_data):
        embeddings, clusters = input_data
        return silhouette_score(AsArray(embeddings), AsArray(clusters))

    # Compute dask objects
    if isinstance(embeddings, dask_cudf.core.DataFrame) | isinstance(embeddings, dask.array.core.Array):
        embeddings = embeddings.compute()

    if isinstance(clusters, dask_cudf.core.Series) | isinstance(clusters, dask.array.core.Array):
        clusters = clusters.compute()

    # Shuffle
    combined = dflib.DataFrame(embeddings) if not isinstance(embeddings, dflib.DataFrame) else embeddings
    embeddings_columns = combined.columns
    cluster_column = 'clusters'

    clusters = dflib.Series(clusters, name=cluster_column)
    combined[cluster_column] = clusters

    # Drop null values
    mask = combined.notnull().any(axis=1)
    combined = combined[mask]

    n_data = min(len(combined), downsample_size)
    logger.info('Calculating silhouette score on {} molecules with batch size of {}...'.format(n_data, batch_size))
    combined = combined.sample(n=n_data, replace=False, random_state=seed) # shuffle via sampling

    embeddings = combined[embeddings_columns]
    clusters = combined[cluster_column]

    # Chunk arrays
    if on_gpu:
        embeddings = cupy.fromDlpack(embeddings.to_dlpack())
        clusters = cupy.fromDlpack(clusters.to_dlpack())

    n_chunks = int(math.ceil(len(embeddings) / batch_size))
    embeddings_chunked = arraylib.array_split(embeddings, n_chunks)
    clusters_chunked = arraylib.array_split(clusters, n_chunks)

    # Calculate scores on batches and return the average
    scores = list(map(_silhouette_scores, zip(embeddings_chunked, clusters_chunked)))
    return numpy.array(scores).mean()


def spearman_rho(data_matrix1, data_matrix2, top_k=10):
    """Calculate spearman's Rho, ranked correlation coefficient

    Args:
        data_matrix1 (2D array or dataframe): matrix with samples as rows, the reference matrix
        data_matrix2 (2D array or dataframe): matrix with samples as rows

    Returns:
        matrix: ranked correlation coeffcients for data
    """

    data_matrix1 = cupy.asnumpy(data_matrix1)
    data_matrix2 = cupy.asnumpy(data_matrix2)

    n_samples = data_matrix1.shape[0]
    data_matrix_argsort = data_matrix1.argsort(axis=1)
    mask_top_k = (data_matrix_argsort > 0) & (data_matrix_argsort <= top_k).reshape(n_samples, -1)
    
    data_matrix1_top_k = data_matrix1[mask_top_k].reshape(n_samples, -1)
    data_matrix2_top_k = data_matrix2[mask_top_k].reshape(n_samples, -1)

   # Includes Dask and cupy and cudf
    if hasattr(data_matrix1_top_k, 'device'):
        data_matrix1_top_k = cupy.asnumpy(data_matrix1_top_k)

    if hasattr(data_matrix2_top_k, 'device'):
        data_matrix2_top_k = cupy.asnumpy(data_matrix2_top_k)

    rho_value = numpy.array([spearmanr(x, y).correlation 
                              for x,y in zip(data_matrix1_top_k, data_matrix2_top_k)]).mean()
    return rho_value
