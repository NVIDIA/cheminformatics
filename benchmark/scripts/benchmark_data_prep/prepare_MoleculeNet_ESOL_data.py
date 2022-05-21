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
import logging
import numpy as np
import pandas as pd
import pathlib
from chembench.utils.smiles import calc_morgan_fingerprints

logger = logging.getLogger(__name__)

# Data as provided by AstraZeneca
# Location of original data: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv

DATA_BENCHMARK_DIR = '/workspace/benchmark/cuchembench/csv_data'

columns = ['SMILES']
physchem_columns = ['measured log solubility in mols per litre']

if __name__ == '__main__':
    data_benchmark_path = os.path.join(DATA_BENCHMARK_DIR, 'benchmark_MoleculeNet_ESOL.csv')
    benchmark_df = pd.read_csv(data_benchmark_path, usecols=columns+physchem_columns)

    # TODO: benchmark SMILES have not been explicitly canonicalized with RDKit. Should this be done?
    fp = calc_morgan_fingerprints(benchmark_df, smiles_col=columns[0])
    # fp = fp.to_pandas()
    fp.columns = fp.columns.astype(np.int64) # TODO may not be needed since formatting fixed
    for col in fp.columns: # TODO why are these floats
        fp[col] = fp[col].astype(np.float32)
    fp.index = benchmark_df.index.astype(np.int64)

    assert len(benchmark_df) == len(fp)
    assert benchmark_df.index.equals(fp.index)

    # Write results
    fp = fp.reset_index()
    fp.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'fingerprints_MoleculeNet_ESOL.csv'), index=False)
