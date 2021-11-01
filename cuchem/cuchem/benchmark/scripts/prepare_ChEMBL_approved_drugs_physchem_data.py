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
from cuchemcommon.data.helper.chembldata import ChEmblData
from cuchemcommon.fingerprint import calc_morgan_fingerprints # TODO RAJESH convert to calc_morgan_fingerprints from datasets.utils

logger = logging.getLogger(__name__)

DATA_BENCHMARK_DIR = os.path.join(pathlib.Path(__file__).absolute().parent.parent,
                                'data')
columns = ['molregno', 'canonical_smiles', 'max_phase_for_ind'] 
physchem_columns = ['mw_freebase', 'alogp', 'hba', 'hbd', 'psa', 'rtb', \
                    'ro3_pass', 'num_ro5_violations', 'cx_logp', 'cx_logd', \
                    'full_mwt', 'aromatic_rings', 'heavy_atoms', 'qed_weighted', \
                    'mw_monoisotopic', 'hba_lipinski', 'hbd_lipinski', 'num_lipinski_ro5_violations']

if __name__ == '__main__':
    results = ChEmblData(fp_type=None).fetch_approved_drugs_physchem(with_labels=True)
    benchmark_df = pd.DataFrame(results[0], columns=results[1])
    benchmark_df = benchmark_df[columns + physchem_columns]
    benchmark_df = benchmark_df.rename(columns={'molregno': 'index'})

    keep_mask = benchmark_df['alogp'].notnull()
    benchmark_df = benchmark_df[keep_mask]

    # TODO: benchmark SMILES have not been explicitly canonicalized with RDKit. Should this be done?
    fp = calc_morgan_fingerprints(benchmark_df)
    fp.columns = fp.columns.astype(np.int64)
    for col in fp.columns:
        fp[col] = fp[col].astype(np.float32)
    fp.index = benchmark_df.index.astype(np.int64)

    assert len(benchmark_df) == len(fp)
    assert benchmark_df.index.equals(fp.index)
    
    # Write results
    benchmark_df = benchmark_df.reset_index()
    fp = fp.reset_index()
    benchmark_df.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'benchmark_ChEMBL_approved_drugs_physchem.csv'), index=False)
    fp.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'fingerprints_ChEMBL_approved_drugs_physchem.csv'), index=False)
    # fp_hdf5 = cudf.DataFrame(fp)
    # fp_hdf5.to_hdf(os.path.join(DATA_BENCHMARK_DIR, 'filter_00.h5', 'fingerprints', format='table'))
