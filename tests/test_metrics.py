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
import cupy
import cudf

from scipy.stats import rankdata as rankdata_cpu
from scipy.stats import spearmanr as spearmanr_cpu_base
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.stats import rankdata as rankdata_cpu
from scipy.stats import spearmanr as spearmanr_cpu_base

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataManip.Metric import GetTanimotoDistMat

# Define paths
# TODO FIX THIS
# _this_directory = os.path.dirname(os.path.realpath(__file__))
_this_directory = '/workspace/tests'
_parent_directory = os.path.dirname(_this_directory)
sys.path.insert(0, _parent_directory)  # TODO better way to add this directory to the path

from nvidia.cheminformatics.utils.distance import tanimoto_calculate
from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores, rankdata, get_kth_unique_value, spearmanr

_data_dir = os.path.join(_this_directory, 'data')
benchmark_approved_drugs_path = os.path.join(_data_dir, 'benchmark_approved_drugs.csv')
fingerprint_approved_drugs_path = os.path.join(_data_dir, 'fingerprints_approved_drugs.csv')
pca_approved_drugs_path = os.path.join(_data_dir, 'pca_approved_drugs.csv')

# Test parameters
run_tanimoto_params = [(benchmark_approved_drugs_path, 'canonical_smiles')]
run_silhouette_score_params = [(pca_approved_drugs_path, 'clusters')]
run_rankdata_params = [(10, 10, 5, 0), (10, 10, 5, 1), (10, 20, 10, 0), (10, 20, 10, 1)]
run_spearman_rho_params = [(pca_approved_drugs_path, fingerprint_approved_drugs_path, 'clusters', 2, 10, 0.99547)]
run_get_kth_unique_value_params = [(2, 0), (2, 1), (10, 0), (10, 1)]


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
    score_gpu1 = batched_silhouette_scores(pca_data, clusters, batch_size=n_data, on_gpu=False)
    score_gpu2 = batched_silhouette_scores(cudf.DataFrame(
        pca_data), cudf.Series(clusters), batch_size=n_data, on_gpu=True)

    assert np.allclose(score_cpu, score_gpu1) & np.allclose(score_cpu, score_gpu2)


@pytest.mark.parametrize('n_rows, n_cols, max_int, axis', run_rankdata_params)
def test_run_rankdata(n_rows, n_cols, max_int, axis):
    """Test the GPU ranking function relative to the CPU baseline"""

    # Use integers to ensure there will be ties
    data = np.random.randint(0, max_int, (n_rows, n_cols))
    rank_cpu = rankdata_cpu(data, axis=axis)
    rank_gpu = rankdata(cupy.asarray(data), axis=axis)
    assert np.allclose(rank_cpu, cupy.asnumpy(rank_gpu))

    if n_rows == n_cols:
        data2 = data * data.T
        rank_cpu2 = rankdata_cpu(data2, axis=axis)
        rank_gpu2 = rankdata(cupy.asarray(data2), axis=axis, is_symmetric=True)
        assert np.allclose(rank_cpu2, cupy.asnumpy(rank_gpu2))


# TODO DOES NOT WORK YET
@pytest.mark.parametrize('top_k, axis', run_get_kth_unique_value_params)
def test_run_get_kth_unique_value(top_k, axis):
    """Test the GPU function to get the kth unique value relative to the CPU baseline"""

    # TODO UPDATE THIS TO USE ANY ARRAY
    # Use integers to ensure there will be ties
    data = np.array([[3., 4., 4., 1., 0.],
                     [0., 2., 4., 2., 1.],
                     [2., 3., 0., 0., 1.],
                     [2., 0., 2., 4., 3.],
                     [1., 3., 0., 2., 2.]])

    rank_cpu = rankdata_cpu(data, axis=axis)
    rank_gpu = rankdata(cupy.asarray(data), axis=axis, is_symmetric=False)

    if axis == 0:
        rank_cpu_iter = rank_cpu.T
    else:
        rank_cpu_iter = rank_cpu

    # Calculate CPU kth rank
    kth_values_cpu = []
    for x in rank_cpu_iter:
        unique_vals = sorted(set(x))
        if top_k >= len(unique_vals):
            val = unique_vals[-1]
        else:
            val = unique_vals[top_k]
        kth_values_cpu.append(val)

    kth_values_cpu = np.array(kth_values_cpu)

    # GPU kth rank
    kth_values_gpu = get_kth_unique_value(rank_gpu, top_k, axis=axis).squeeze()
    print(kth_values_cpu, kth_values_gpu)

    # print(kth_values_cpu, kth_values_gpu)


# TODO WRITE SPEARMAN R TEST
# n = 5
# d1c = np.random.randint(0, n, (n,n)).astype(np.float)
# d2c = np.random.randint(0, n, (n,n)).astype(np.float)
# d1g = cupy.array(d1c)
# d2g = cupy.array(d2c)
# ranks1 = rankdata(d1g, axis=1, method='average', na_option='keep')
# ranks2 = rankdata(d2g, axis=1, method='average', na_option='keep')
# print(corr_pairwise(ranks1, ranks2, return_pearson=True))
# print([spearmanr_cpu(cupy.asnumpy(ranks1[x]), cupy.asnumpy(ranks2[x])).correlation for x in range(ranks1.shape[0])])
    




# @pytest.mark.parametrize('pca_approved_drugs_csv, fingerprint_approved_drugs_csv, cluster_column, n_dims_eucl_data, top_k, top_k_value', run_spearman_rho_params)
# def test_run_spearman_rho(pca_approved_drugs_csv, fingerprint_approved_drugs_csv, cluster_column, n_dims_eucl_data, top_k, top_k_value):
#     """Validate the spearman rho score"""

#     # Load PCA data to use as Euclidean distances
#     pca_data = pd.read_csv(pca_approved_drugs_csv).set_index('molregno').drop(cluster_column, axis=1)
#     float_data = pca_data[pca_data.columns[:n_dims_eucl_data]]
#     pairwise_eucl_dist = pairwise_distances(float_data)
#     np.fill_diagonal(pairwise_eucl_dist, np.inf)

#     # Load fingerprints for tanimoto distances
#     fp_data = pd.read_csv(fingerprint_approved_drugs_csv).set_index('molregno')
#     tanimoto_dist = tanimoto_calculate(cupy.array(fp_data), calc_distance=True)
#     cupy.fill_diagonal(tanimoto_dist, cupy.inf)

#     # Check small amount of data (top_k) for absolute value
#     top_k_value_check = spearmanr(pairwise_eucl_dist, tanimoto_dist, top_k=top_k)
#     assert np.allclose(top_k_value_check, np.round(top_k_value, 5))

#     # Check all data compared to the CPU version
#     top_k_all_data = pairwise_eucl_dist.shape[1]  # - 1
#     all_data_gpu = spearmanr(pairwise_eucl_dist, tanimoto_dist, top_k=top_k_all_data)

#     tanimoto_dist_cpu = cupy.asnumpy(tanimoto_dist)
#     all_data_cpu = np.array([spearmanr(x, y).correlation for x, y in zip(pairwise_eucl_dist, tanimoto_dist_cpu)])
#     all_data_cpu = np.nanmean(all_data_cpu)
#     # assert np.allclose(all_data_gpu, all_data_cpu) # TODO debug this


