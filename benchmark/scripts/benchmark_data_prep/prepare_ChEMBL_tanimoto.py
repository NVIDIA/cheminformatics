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
from chembench.utils.chembldata import ChEmblData
from chembench.utils.smiles import calc_morgan_fingerprints

logger = logging.getLogger(__name__)

DATA_BENCHMARK_DIR = '/workspace/benchmark/scripts/data'
DEFAULT_MAX_SEQ_LEN = 512

if __name__ == '__main__':

    num_samples = 20000
    benchmark_df = pd.DataFrame(ChEmblData().fetch_random_samples(num_samples, DEFAULT_MAX_SEQ_LEN))
    benchmark_df.rename(columns={'len': 'length'}, inplace=True)

    # TODO: benchmark SMILES have not been canonicalized. Should this be done?
    print(benchmark_df.head())
    fp = calc_morgan_fingerprints(benchmark_df, smiles_col=1)
    fp.columns = fp.columns.astype(np.int64)
    fp.index = benchmark_df.index.astype(np.int64) # TODO is this needed?

    for col in fp.columns:
        fp[col] = fp[col].astype(np.int) # TODO is this needed?
        # fp[col] = fp[col].astype(np.float32)

    assert len(benchmark_df) == len(fp)
    assert benchmark_df.index.equals(fp.index)

    # Write results
    benchmark_df.reset_index(inplace=True)  # For consistency with approved drugs, only one has index reset
    benchmark_df.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'benchmark_ChEMBL_random_sampled_drugs.csv'))
    fp.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'fingerprints_ChEMBL_random_sampled_drugs.csv'))
