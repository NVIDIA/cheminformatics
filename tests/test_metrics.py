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
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem#, DataStructs
from rdkit.DataManip.Metric import GetTanimotoDistMat
import pandas as pd
import numpy as np
import cupy
import cudf

# Define paths
# _this_directory = os.path.dirname(os.path.realpath(__file__))
_this_directory = '/workspace/tests'
_parent_directory = os.path.dirname(_this_directory)
sys.path.insert(0, _parent_directory) # TODO better way to add this directory to the path

from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores, spearman_rho
from nvidia.cheminformatics.utils.distance import tanimoto_calculate

_data_dir = os.path.join(_this_directory, 'data')
benchmark_approved_drugs_path = os.path.join(_data_dir, 'benchmark_approved_drugs.csv')
fingerprint_approved_drugs_path = os.path.join(_data_dir, 'fp_approved_drugs.csv')
pca_approved_drugs_path = os.path.join(_data_dir, 'pca_approved_drugs.csv')

# Test parameters
run_tanimoto_params = [(benchmark_approved_drugs_path, 'canonical_smiles')]
run_silhouette_score_params = [(pca_approved_drugs_path, 'clusters')]


@pytest.mark.parametrize('benchmark_data_csv, column_name', run_tanimoto_params)
def test_run_tanimoto(benchmark_data_csv, column_name):
    """Validate tanimoto distance calculation"""

    # RDKit Tanimoto distance on CPU is the baseline
    smiles_data = pd.read_csv(benchmark_data_csv)[column_name]
    mol_data = [Chem.MolFromSmiles(x) for x in smiles_data]
    morganfp_data = [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in mol_data]
    tanimoto_dist_rdkit = GetTanimotoDistMat(morganfp_data)

    # Compare to GPU version
    idx = np.tril_indices(len(morganfp_data), k=-1)
    fparray = [x.ToBitString() for x in morganfp_data]
    fparray = [np.array(list(x)).astype(np.int) for x in fparray]
    tanimoto_dist_gpu = tanimoto_calculate(cupy.array(fparray), calc_distance=True)
    tanimoto_dist_gpu = cupy.asnumpy(tanimoto_dist_gpu)[idx]
    # TODO add value

    assert np.allclose(cupy.asnumpy(tanimoto_dist_gpu), tanimoto_dist_rdkit)


@pytest.mark.parametrize('pca_approved_csv, cluster_column', run_silhouette_score_params)
def test_run_silhouette_score(pca_approved_csv, cluster_column):
    """Validate the silhouette score"""
    pca_data = pd.read_csv(pca_approved_drugs_path).set_index('molregno')
    clusters = pca_data[cluster_column]
    pca_data.drop(cluster_column, axis=1, inplace=True)
    score_cpu = silhouette_score(pca_data, clusters)

    # TODO copy pca_data or ensure it doesn't modify original
    n_data = pca_data.shape[0]
    score_gpu1 = batched_silhouette_scores(pca_data, clusters, batch_size=n_data, on_gpu=False)
    score_gpu2 = batched_silhouette_scores(cudf.DataFrame(pca_data), cudf.Series(clusters), batch_size=n_data, on_gpu=True)

    assert np.allclose(score_cpu, score_gpu1) & np.allclose(score_cpu, score_gpu2)


cluster_column = 'cluster'
pca_data = pd.read_csv(pca_approved_drugs_path).set_index('molregno').drop(cluster_column, axis=1)
float_data = pca_data[:, :2]
fp_data = pd.read_csv(fingerprint_approved_drugs_path).set_index('molregno')

pairwise_eucl_dist = pairwise_distances(float_data)
tanimoto_dist = tanimoto_calculate(cupy.array(fp_data), calc_distance=True)
spearman_rho(data_matrix1, data_matrix2, top_k=10)

# import unittest
# import tempfile
# import shutil
# import sys
# import shlex
# import glob

# # Output directory
# temp_dir = tempfile.mkdtemp()

# # Parameter lists
# run_benchmark_params = [ ([{'test_type': 'gpu', 'n_workers':  1, 'n_mol': -1}, 
#                            {'test_type': 'cpu', 'n_workers': 19, 'n_mol': -1}], _data_dir, temp_dir) ]
# load_benchmark_params = [(temp_dir)]


# @pytest.mark.parametrize('benchmark_config_list, data_dir, output_dir', run_benchmark_params)
# def test_run_benchmark(benchmark_config_list, data_dir, output_dir):

#     output_file = os.path.join(output_dir, 'benchmark.csv')
#     if os.path.exists(output_file):
#         os.remove(output_file)

#     for config in benchmark_config_list:
#         test_type = config['test_type']
#         n_workers = config['n_workers']
#         n_mol = config['n_mol']
        
#         # Create run command and inject into sys.argv
#         command = f'startdash.py analyze -b --cache {data_dir} '
#         if test_type == 'cpu':
#             command += f'--{test_type} '
#         command += f'--n_{test_type} {n_workers} --n_mol {n_mol} --output_dir {output_dir}'

#         sys_argv = shlex.split(command)
#         with unittest.mock.patch('sys.argv', sys_argv):
#             Launcher()

#     # Filename is set in workflow -- move to create randomized name
#     temp_file = tempfile.NamedTemporaryFile(prefix='benchmark_', suffix='.csv', dir=output_dir, delete=False).name
#     shutil.move(output_file, temp_file)
#     assert os.path.exists(temp_file)
    
#     benchmark_results = pd.read_csv(temp_file, comment='#')
#     nrows, ncols = benchmark_results.shape
#     assert nrows == len(benchmark_config_list) * 5
#     assert ncols == 8
