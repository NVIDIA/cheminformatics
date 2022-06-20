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
from chembench.utils.chembldata import ChEmblData
from chembench.utils.smiles import calc_morgan_fingerprints

logger = logging.getLogger(__name__)

DATA_BENCHMARK_DIR = os.path.join(pathlib.Path(__file__).absolute().parent.parent.parent,
                                  'scripts', 'data')

if __name__ == '__main__':
    results = ChEmblData().fetch_approved_drugs(with_labels=True)
    benchmark_df = pd.DataFrame(results[0], columns=results[1])

    # TODO: benchmark SMILES have not been explicitly canonicalized with RDKit. Should this be done?
    fp = calc_morgan_fingerprints(benchmark_df)
    fp.index = benchmark_df.index.astype(np.int64)
    fp.columns = fp.columns.astype(np.int64) # TODO may not be needed since formatting fixed
    for col in fp.columns: # TODO why are these floats
        fp[col] = fp[col].astype(np.float32)

    assert len(benchmark_df) == len(fp)
    assert benchmark_df.index.equals(fp.index)

    # Write results
    benchmark_df = benchmark_df.reset_index()
    fp = fp.reset_index()
    benchmark_df.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'benchmark_approved_drugs.csv'), index=False)
    fp.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'fingerprints_approved_drugs.csv'), index=False)
    # fp_hdf5 = cudf.DataFrame(fp)
    # fp_hdf5.to_hdf(os.path.join(DATA_BENCHMARK_DIR, 'filter_00.h5', 'fingerprints', format='table'))
