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
from sklearn.metrics import silhouette_score


def batched_silhouette_scores(embeddings, clusters, batch_size=5000, seed=0, on_gpu=True):
    """Calculate silhouette score in batches on the CPU. Compatible with data on GPU or CPU

    Args:
        embeddings (cudf.DataFrame or cupy.ndarray): input features to clustering
        clusters (cudf.DataFrame or cupy.ndarray): cluster values for each data point
        batch_size (int, optional): Size for batching. Defaults to 5000.
        seed (int, optional): Random seed. Defaults to 0.
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

    # Shuffle on GPU
    combined = dflib.DataFrame(embeddings) if not isinstance(embeddings, dflib.DataFrame) else embeddings
    embeddings_columns = combined.columns
    cluster_column = 'clusters'

    clusters = dflib.Series(clusters, name=cluster_column)
    combined[cluster_column] = clusters
    combined = combined.sample(n=len(combined), replace=False, random_state=seed) # shuffle via sampling

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
