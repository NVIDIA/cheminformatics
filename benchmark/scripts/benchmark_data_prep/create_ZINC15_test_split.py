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
import glob
import pandas as pd
import numpy as np
import multiprocessing as mp
import pathlib

from cuchembench.utils.smiles import calc_morgan_fingerprints

from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem

from CDDD_train_data_overlap import MOLECULES_IN_CDDD_TRAIN_DATA

DATA_BENCHMARK_DIR = os.path.join(pathlib.Path(__file__).absolute().parent.parent.parent,
                                'cuchembench', 'csv_data')
NUM_PROCESSES = (mp.cpu_count() * 2) - 1 # --> max num proceses, but needs more memory
NUM_DATA = 20000
ZINC_DATA_DIR = '/data/zinc_csv_split'

def calc_properties(smiles_list):
    logp_vals, mw_vals = [], []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        logp_vals.append(MolLogP(mol, True))
        mw_vals.append(ExactMolWt(mol))
    return pd.DataFrame({'logp': logp_vals, 'mw': mw_vals})


if __name__ == '__main__':

    # Read data
    zinc_test_filelist = glob.glob(os.path.join(ZINC_DATA_DIR, 'test', 'x*.csv'))
    benchmark_df = list()
    for fil in zinc_test_filelist:
        benchmark_df.append(pd.read_csv(fil, names=['canonical_smiles'], skiprows=[0]))
    benchmark_df = pd.concat(benchmark_df, axis=0)
    mask = benchmark_df['canonical_smiles'].isin(MOLECULES_IN_CDDD_TRAIN_DATA).pipe(np.invert)
    benchmark_df = benchmark_df[mask].reset_index(drop=True)

    benchmark_df = benchmark_df.sample(n=NUM_DATA, replace=False, random_state=0).reset_index(drop=True)
    assert NUM_DATA <= len(benchmark_df)
    benchmark_df['length'] = benchmark_df['canonical_smiles'].map(len)

    # Calculate properties
    # pool = mp.Pool(processes=NUM_PROCESSES)
    # chunks = benchmark_df['canonical_smiles'].to_numpy()
    # chunks = np.array_split(chunks, NUM_PROCESSES)
    # outputs = pool.map(calc_properties, chunks)
    # outputs = pd.concat(outputs, axis=0).reset_index(drop=True)
    outputs = calc_properties(benchmark_df['canonical_smiles'].tolist())
    benchmark_df = pd.concat([benchmark_df, outputs], axis=1)
    benchmark_df.index.name = 'index'


    fp = calc_morgan_fingerprints(benchmark_df[['canonical_smiles']])
    fp.index = fp.index.astype(np.int64)
    fp.index.name = 'index'
    fp.columns = fp.columns.astype(np.int64) # TODO may not be needed since formatting fixed
    for col in fp.columns: # TODO why are these floats
        fp[col] = fp[col].astype(np.float32)

    # Write results
    benchmark_df.reset_index().to_csv(os.path.join(DATA_BENCHMARK_DIR, 'benchmark_zinc15_test.csv'), index=False)
    fp.reset_index().to_csv(os.path.join(DATA_BENCHMARK_DIR, 'fingerprints_zinc15_test.csv'), index=False)
    # fp_hdf5 = cudf.DataFrame(fp)
    # fp_hdf5.to_hdf(os.path.join(DATA_BENCHMARK_DIR, 'filter_00.h5', 'fingerprints', format='table'))
