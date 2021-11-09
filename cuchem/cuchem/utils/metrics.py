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
import math

import cupy
import numpy
from cuchemcommon.data.helper.chembldata import BATCH_SIZE
from numba import cuda
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def batched_silhouette_scores(embeddings, clusters, batch_size=BATCH_SIZE):
    """Calculate silhouette score in batches on the CPU. Compatible with data on GPU or CPU

    Args:
        embeddings (cudf.DataFrame or cupy.ndarray): input features to clustering
        clusters (cudf.DataFrame or cupy.ndarray): cluster values for each data point
        batch_size (int, optional): Size for batching.

    Returns:
        float: mean silhouette score from batches
    """

    # Function to calculate batched results
    def _silhouette_scores(input_data):
        embeddings, clusters = input_data
        return silhouette_score(cupy.asnumpy(embeddings), cupy.asnumpy(clusters))

    if hasattr(embeddings, 'values'):
        embeddings = embeddings.values
    embeddings = cupy.asarray(embeddings)

    if hasattr(clusters, 'values'):
        clusters = clusters.values
    clusters = cupy.asarray(clusters)

    n_data = len(embeddings)
    msg = 'Calculating silhouette score on {} molecules'.format(n_data)
    if batch_size < n_data:
        msg += ' with batch size of {}'.format(batch_size)
    logger.info(msg + ' ...')

    n_chunks = int(math.ceil(n_data / batch_size))
    embeddings_chunked = cupy.array_split(embeddings, n_chunks)
    clusters_chunked = cupy.array_split(clusters, n_chunks)

    # Calculate scores on batches and return the average
    scores = list(map(_silhouette_scores, zip(embeddings_chunked, clusters_chunked)))
    return numpy.nanmean(numpy.array(scores))
