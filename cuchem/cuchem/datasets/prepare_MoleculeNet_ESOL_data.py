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

import os
import numpy as np
import pandas as pd
import pathlib
from cuchemcommon.fingerprint import calc_morgan_fingerprints

# Data location: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
# Download to DATA_BENCHMARK_DIR

DATA_BENCHMARK_DIR = os.path.join(pathlib.Path(__file__).absolute().parent.parent,
                                'data')
columns = ['smiles'] 
physchem_columns = ['measured log solubility in mols per litre']

if __name__ == '__main__':
    data_benchmark_path = os.path.join(DATA_BENCHMARK_DIR, 'benchmark_MoleculeNet_ESOL.csv')
    benchmark_df = pd.read_csv(data_benchmark_path, usecols=columns+physchem_columns)
    # benchmark_df = benchmark_df.rename(columns={phychem_columns[0]: 'log_solubility_(mol_per_L)'})

    # TODO: benchmark SMILES have not been explicitly canonicalized with RDKit. Should this be done?
    fp = calc_morgan_fingerprints(benchmark_df, smiles_col=columns[0])
    fp.columns = fp.columns.astype(np.int64)
    for col in fp.columns:
        fp[col] = fp[col].astype(np.float32)
    fp.index = benchmark_df.index.astype(np.int64)
    fp.index.rename('index')
    fp = fp.reset_index()
    
    # Write results
    fp.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'fingerprints_MoleculeNet_ESOL.csv'), index=False)
