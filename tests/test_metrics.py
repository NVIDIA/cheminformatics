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
import pytest
import sys
import os

import pandas as pd
import numpy as np
from math import isnan

from scipy.stats import rankdata as rankdata_cpu
from scipy.stats import spearmanr as spearmanr_cpu
from sklearn.metrics import silhouette_score

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataManip.Metric import GetTanimotoDistMat

import cupy
import cudf
from cuml.metrics import pairwise_distances

# Define paths
_this_directory = os.path.dirname(os.path.realpath(__file__))
_parent_directory = os.path.dirname(_this_directory)
sys.path.insert(0, _parent_directory)  # TODO is there a better way to add nvidia directory to the path

from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores, rankdata, get_kth_unique_value, corr_pairwise, spearmanr
from nvidia.cheminformatics.utils.distance import tanimoto_calculate

_data_dir = os.path.join(_this_directory, 'data')
benchmark_approved_drugs_path = os.path.join(_data_dir, 'benchmark_approved_drugs.csv')
fingerprint_approved_drugs_path = os.path.join(_data_dir, 'fingerprints_approved_drugs.csv')
pca_approved_drugs_path = os.path.join(_data_dir, 'pca_approved_drugs.csv')

# Test parameters
run_tanimoto_params = [(benchmark_approved_drugs_path, 'canonical_smiles')]
run_silhouette_score_params = [(pca_approved_drugs_path, 'clusters')]
run_rankdata_params = [(10, 10, 5, 0), (10, 10, 5, 1), (10, 20, 10, 0), (10, 20, 10, 1)]
run_corr_pairwise = [(10, 10, 5, 0), (10, 10, 5, 2), (10, 20, 10, 0), (10, 20, 10, 5)]
run_get_kth_unique_value_params = [(10, 10, 5, 2, 0), (10, 10, 5, 2, 1),
                                   (10, 20, 10, 5, 0), (10, 20, 10, 5, 1), (10, 20, 10, 100, 1)]
run_spearman_rho_params = [(pca_approved_drugs_path, fingerprint_approved_drugs_path, 'clusters', 2, 100)]


# Accessory functions
def _random_nans(data1, data2, num_nans):
    """Randomly add NaNs in identical positions to two numpy arrays"""
    n_rows, n_cols = data1.shape
    row_array = np.random.choice(np.arange(0, n_rows), num_nans)
    col_array = np.random.choice(np.arange(0, n_cols), num_nans)
    data1[row_array, col_array] = np.NaN
    data2[row_array, col_array] = np.NaN

    return data1, data2


def _rowwise_numpy_corr(data1, data2, func):
    """Pariwise correlation function on CPU"""
    corr_array = []
    for d1, d2 in zip(data1, data2):
        mask = np.invert(np.isnan(d1) | np.isnan(d2))
        val = func(d1[mask], d2[mask])
        if hasattr(val, 'correlation'):
            val = val.correlation
        if hasattr(val, '__len__'):
            val = val[1, 0]
        corr_array.append(val)

    return np.array(corr_array)


def _get_kth_unique_array_cpu(data, k, axis):
    """Return kth unique values for a sorted array along row/column"""
    data = data.T if axis == 0 else data
    kth_values = []

    for vector in data:
        pos = 0
        prev_val = np.NaN

        for val in vector:
            if not isnan(val):
                if val != prev_val:
                    prev_val = val
                    pos += 1

            if pos == k:
                break

        kth_values.append(prev_val)
    return np.array(kth_values)


# The unit tests
@pytest.mark.parametrize('benchmark_data_csv, column_name', run_tanimoto_params)
def test_run_tanimoto(benchmark_data_csv, column_name):
    """Validate tanimoto distance calculation"""

    # Load data and calculate Morgan Fingerprints
    smiles_data = pd.read_csv(benchmark_data_csv)[column_name]
    mol_data = [Chem.MolFromSmiles(x) for x in smiles_data]
    morganfp_data = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in mol_data]

    # RDKit Tanimoto distance on CPU is the baseline
    tanimoto_dist_rdkit = GetTanimotoDistMat(morganfp_data)

    # Compare to GPU version
    idx = np.tril_indices(len(morganfp_data), k=-1)
    fparray = [x.ToBitString() for x in morganfp_data]
    fparray = [np.array(list(x)).astype(np.int) for x in fparray]
    tanimoto_dist_gpu = tanimoto_calculate(cupy.array(fparray), calc_distance=True)
    tanimoto_dist_gpu = cupy.asnumpy(tanimoto_dist_gpu)[idx]

    assert np.allclose(cupy.asnumpy(tanimoto_dist_gpu), tanimoto_dist_rdkit)


@pytest.mark.parametrize('pca_approved_csv, cluster_column', run_silhouette_score_params)
def test_run_silhouette_score(pca_approved_csv, cluster_column):
    """Validate the silhouette score"""

    pca_data = pd.read_csv(pca_approved_csv).set_index('molregno')
    clusters = pca_data[cluster_column]
    pca_data.drop(cluster_column, axis=1, inplace=True)
    score_cpu = silhouette_score(pca_data, clusters)

    # TODO copy pca_data or ensure it doesn't modify original
    n_data = pca_data.shape[0]
    score_gpu1 = batched_silhouette_scores(pca_data, clusters, batch_size=n_data)
    score_gpu2 = batched_silhouette_scores(cudf.DataFrame(
        pca_data), cudf.Series(clusters), batch_size=n_data)

    assert np.allclose(score_cpu, score_gpu1) & np.allclose(score_cpu, score_gpu2)


@pytest.mark.parametrize('n_rows, n_cols, max_int, axis', run_rankdata_params)
def test_run_rankdata(n_rows, n_cols, max_int, axis):
    """Test the GPU ranking function relative to the CPU baseline"""
    # TODO Add tests for ranking with NaNs once it's fixed in cuDF

    # Use integers to ensure there will be ties
    data = np.random.randint(0, max_int, (n_rows, n_cols)).astype(np.float)
    rank_cpu = rankdata_cpu(data, axis=axis)
    rank_gpu = rankdata(cupy.asarray(data), axis=axis)
    assert np.allclose(rank_cpu, cupy.asnumpy(rank_gpu))

    if n_rows == n_cols:
        data2 = data * data.T
        rank_cpu2 = rankdata_cpu(data2, axis=axis)
        rank_gpu2 = rankdata(cupy.asarray(data2), axis=axis, is_symmetric=True)
        assert np.allclose(rank_cpu2, cupy.asnumpy(rank_gpu2))


@pytest.mark.parametrize('n_rows, n_cols, max_int, num_nans', run_corr_pairwise)
def test_run_corr_pairwise(n_rows, n_cols, max_int, num_nans):
    """Test the pairwise covariance matrix calculation and the Pearson correlation coefficient"""

    data1c = np.random.randint(0, max_int, (n_rows, n_cols)).astype(np.float)
    data2c = np.random.randint(0, max_int, (n_rows, n_cols)).astype(np.float)

    if num_nans > 0:
        data1c, data2c = _random_nans(data1c, data2c, num_nans)

    data1g = cupy.array(data1c)
    data2g = cupy.array(data2c)

    # Covariance matrix
    cov_cpu = _rowwise_numpy_corr(data1c, data2c, np.cov)
    cov_gpu = corr_pairwise(data1g, data2g, False).squeeze()
    assert np.allclose(cov_cpu, cupy.asnumpy(cov_gpu), equal_nan=True)

    # Pearson correlation
    corcoef_cpu = _rowwise_numpy_corr(data1c, data2c, np.corrcoef)
    corcoef_gpu = corr_pairwise(data1g, data2g, True).squeeze()
    assert np.allclose(corcoef_cpu, cupy.asnumpy(corcoef_gpu), equal_nan=True)


@pytest.mark.parametrize('n_rows, n_cols, max_int, top_k, axis', run_get_kth_unique_value_params)
def test_run_get_kth_unique_value(n_rows, n_cols, max_int, top_k, axis):
    """Test the GPU function to get the kth unique value relative to the CPU baseline"""

    data = np.random.randint(0, max_int, (n_rows, n_cols)).astype(np.float)
    data = rankdata_cpu(data, axis=axis)
    data.sort(axis=axis)

    # Test without NaNs
    kth_values_cpu = _get_kth_unique_array_cpu(data, top_k, axis)
    kth_values_gpu = get_kth_unique_value(cupy.array(data), top_k, axis=axis).squeeze()
    assert np.allclose(kth_values_cpu, cupy.asnumpy(kth_values_gpu), equal_nan=True)

    # And with NaNs
    np.fill_diagonal(data, np.NaN)
    data[2, :] = np.NaN

    kth_values_cpu = _get_kth_unique_array_cpu(data, top_k, axis)
    kth_values_gpu = get_kth_unique_value(cupy.array(data), top_k, axis=axis).squeeze()
    assert np.allclose(kth_values_cpu, cupy.asnumpy(kth_values_gpu), equal_nan=True)


@pytest.mark.parametrize('pca_approved_drugs_csv, fingerprint_approved_drugs_csv, cluster_column, n_dims_eucl_data, top_k', run_spearman_rho_params)
def test_run_spearman_rho(pca_approved_drugs_csv, fingerprint_approved_drugs_csv, cluster_column, n_dims_eucl_data, top_k):
    """Validate the spearman rho scoring"""

    # Load PCA data to use as Euclidean distances
    pca_data = pd.read_csv(pca_approved_drugs_csv).set_index('molregno').drop(cluster_column, axis=1)
    float_data = pca_data[pca_data.columns[:n_dims_eucl_data]]
    euclidean_dist = pairwise_distances(cupy.array(float_data))

    # Load fingerprints and calculate tanimoto distance
    fp_data = pd.read_csv(fingerprint_approved_drugs_csv).set_index('molregno')
    tanimoto_dist = tanimoto_calculate(cupy.array(fp_data), calc_distance=True)

    # Check all data compared to the CPU version
    all_data_gpu = spearmanr(tanimoto_dist, euclidean_dist)

    euclidean_dist_cpu = cupy.asnumpy(euclidean_dist)
    tanimoto_dist_cpu = cupy.asnumpy(tanimoto_dist)
    all_data_cpu = _rowwise_numpy_corr(tanimoto_dist_cpu, euclidean_dist_cpu, spearmanr_cpu)

    cupy.allclose(cupy.array(all_data_cpu), all_data_gpu, atol=0.005, equal_nan=True)

    # Check using top k calculation compared to the CPU version
    top_k_data_gpu = spearmanr(tanimoto_dist, euclidean_dist, top_k=top_k, axis=1)

    cupy.fill_diagonal(tanimoto_dist, cupy.NaN)
    kth_lim = get_kth_unique_value(tanimoto_dist, top_k, axis=1)
    mask = tanimoto_dist > kth_lim
    tanimoto_dist[mask] = cupy.NaN
    euclidean_dist[mask] = cupy.NaN
    euclidean_dist_cpu = cupy.asnumpy(euclidean_dist)
    tanimoto_dist_cpu = cupy.asnumpy(tanimoto_dist)
    top_k_data_cpu = _rowwise_numpy_corr(tanimoto_dist_cpu, euclidean_dist_cpu, spearmanr_cpu)

    cupy.allclose(cupy.array(top_k_data_cpu), top_k_data_gpu, atol=0.005, equal_nan=True)
