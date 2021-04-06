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
import pandas
import numpy
from math import isnan
import cudf
import cupy
from numba import cuda

import dask
import dask_cudf
from sklearn.metrics import silhouette_score
from nvidia.cheminformatics.data.helper.chembldata import BATCH_SIZE

import logging
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


def rankdata(data, method='average', na_option='keep', axis=1, is_symmetric=False):
    """Rank observations for a series of samples, with tie handling
    NOTE: due to a bug with cudf ranking, data will be transposed if row-wise ranking is
    selected

    Parameters
    ----------
    data : array_like
        The array of values to be ranked.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        The method used to assign ranks to tied elements.
        The following methods are available (default is 'average'):
          * 'average': The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.
          * 'min': The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
          * 'max': The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.
          * 'dense': Like 'min', but the rank of the next highest element is
            assigned the rank immediately after those assigned to the tied
            elements.
          * 'ordinal': All values are given a distinct rank, corresponding to
            the order that the values occur in `a`.
    axis : {None, int}, optional
        Axis along which to perform the ranking. Default is 1 -- samples in rows,
        observations in columns
    is_symmetric : {False, bool}, optional
        Will be used to avoid additional data transpose steps if axis = 1

    Returns
    -------
    ranks : cupy ndarray
         An array of size equal to the size of `a`, containing rank
         scores.

    See also scipy.stats.rankdata, for which this function is a replacement
    """

    dtype = cupy.result_type(data.dtype, cupy.float64)
    data = cupy.asarray(data, dtype=dtype)

    if is_symmetric:
        assert data.ndim == 2
        assert data.shape[0] == data.shape[1]

    if data.ndim < 2:
        data = data[:, None]
    elif (data.ndim == 2) & (axis == 1) & (not is_symmetric):
        data = data.T

    ranks = cudf.DataFrame(data).rank(axis=0, method=method, na_option=na_option)
    ranks = ranks.values

    if axis == 1:
        ranks = ranks.T
    return ranks


@cuda.jit()
def _get_kth_unique_kernel(data, kth_values, k, axis):
    """Numba kernel to get the kth unique rank from a sorted array"""

    i = cuda.grid(1)

    if axis == 1:
        vector = data[i, :]
    else:
        vector = data[:, i]

    pos = 0
    prev_val = cupy.NaN

    for val in vector:
        if not isnan(val):
            if val != prev_val:
                prev_val = val
                pos += 1

        if pos == k:
            break

    kth_values[i] = prev_val


def get_kth_unique_value(data, k, axis=1):
    """Find the kth value along an axis of a matrix on the GPU

    Parameters
    ----------
    data : array_like
        The array of values to be ranked.
    k : {int} kth unique value to be found
    axis : {None, int}, optional
        Axis along which to perform the ranking. Default is 1 -- samples in rows,
        observations in columns

    Returns
    -------
    kth_values : cupy ndarray
         An array of kth values.
    """

    # Coerce data into array -- make a copy since it needs to be sorted
    # TODO -- should the sort be done in Numba kernel (and how to do it)?
    dtype = cupy.result_type(data, cupy.float64)
    data_id = id(data)
    data = cupy.ascontiguousarray(data, dtype=dtype)

    if data_id == id(data):  # Ensure sort is being done on a copy
        data = data.copy()

    assert data.ndim <= 2

    if data.ndim < 2:
        if axis == 0:
            data = data[:, None]
        else:
            data = data[None, :]

    if axis == 0:
        n_obs, n_samples = data.shape
    else:
        n_samples, n_obs = data.shape

    data.sort(axis=axis)
    kth_values = cupy.zeros(n_samples, dtype=data.dtype)
    _get_kth_unique_kernel.forall(n_samples, 1)(data, kth_values, k, axis)

    if axis == 0:
        kth_values = kth_values[None, :]
    else:
        kth_values = kth_values[:, None]

    return kth_values


def corr_pairwise(x, y, return_pearson=False):
    """Covariance and Pearson product-moment correlation coefficients on the GPU for paired data with tolerance of NaNs.
       Curently only supports rows as samples and columns as observations.

    Parameters
    ----------
    x : array_like
        The baseline array of values.
    y : array_like
        The comparison array of values.

    Returns
    -------
    corr : cupy ndarray
         Array of correlation values
    """

    def _cov_pairwise(x1, x2, factor):
        return cupy.nansum(x1 * x2, axis=1, keepdims=True) * cupy.true_divide(1, factor)

    # Coerce arrays into 2D format and set dtype
    dtype = cupy.result_type(x, y, cupy.float64)
    x = cupy.asarray(x, dtype=dtype)
    y = cupy.asarray(y, dtype=dtype)

    assert x.shape == y.shape
    if x.ndim < 2:
        x = x[None, :]
        y = y[None, :]
    n_samples, n_obs = x.shape

    # Calculate degrees of freedom for each sample pair
    ddof = 1
    nan_count = (cupy.isnan(x) | cupy.isnan(y)).sum(axis=1, keepdims=True)
    fact = n_obs - nan_count - ddof

    # Mean normalize
    x -= cupy.nanmean(x, axis=1, keepdims=True)
    y -= cupy.nanmean(y, axis=1, keepdims=True)

    # Calculate covariance matrix
    corr = _cov_pairwise(x, y, fact)

    if return_pearson:
        x_corr = _cov_pairwise(x, x, fact)
        y_corr = _cov_pairwise(y, y, fact)
        auto_corr = cupy.sqrt(x_corr) * cupy.sqrt(y_corr)
        corr = corr / auto_corr
        corr = cupy.clip(corr.real, -1, 1, out=corr.real)
        return corr

    return corr.squeeze()


def spearmanr(x, y, axis=1, top_k=None):
    """GPU implementation of Spearman R correlation coefficient for paired data with NaN support

    Parameters
    ----------
    x : array_like
        The baseline array of values.
    y : array_like
        The comparison array of values.
    axis : {None, int}, optional
        Axis along which to perform the ranking. Default is 1 -- samples in rows,
        observations in columns
    top_k : {int} kth unique value to be found

    Returns
    -------
    spearmanr_array : cupy ndarray
         Array of spearmanr rank correlation values
    """

    if hasattr(x, 'values'):
        x = x.values
    x = cupy.array(x, copy=True)

    if hasattr(y, 'values'):
        y = y.values
    y = cupy.array(y, copy=True)

    assert x.ndim <= 2
    assert x.shape == y.shape

    if x.ndim < 2:
        if axis == 0:
            x = x[:, None]
            y = y[:, None]
        else:
            x = x[None, :]
            y = y[None, :]

    if axis == 0:
        n_obs, n_samples = x.shape
    else:
        n_samples, n_obs = x.shape

    n_obs -= 1
    assert n_obs > 2

    msg = 'Calculating Spearman correlation coefficient on {} molecules'.format(n_samples)
    if top_k is not None:
        msg += ' with selection of top {} molecules'.format(top_k)
    logger.info(msg + ' ...')

    # Force diagonal to be last in ranking so it can be ignored
    cupy.fill_diagonal(x, cupy.NaN)
    cupy.fill_diagonal(y, cupy.NaN)

    ranks_x = rankdata(x, axis=axis, method='average', na_option='keep')
    ranks_y = rankdata(y, axis=axis, method='average', na_option='keep')

    # cudf does not currently preserve the NaNs, even with na_option='keep' so add them back
    cupy.fill_diagonal(ranks_x, cupy.NaN)
    cupy.fill_diagonal(ranks_y, cupy.NaN)

    # Filter out values above top k
    if top_k is not None:
        if top_k <= n_obs:
            top_k_values = get_kth_unique_value(ranks_x, top_k, axis=axis)
            mask = ranks_x > top_k_values
            ranks_x[mask] = cupy.NaN
            ranks_y[mask] = cupy.NaN

    spearmanr_array = corr_pairwise(ranks_x, ranks_y, return_pearson=True).squeeze()
    return spearmanr_array
