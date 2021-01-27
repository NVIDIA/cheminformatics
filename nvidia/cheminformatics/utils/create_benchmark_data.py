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
import pandas as pd
import sqlite3
import cudf
import numpy as np
from nvidia.cheminformatics.fingerprint import MorganFingerprint

CHEMBL_DB_PATH = '/data/db/chembl_27.db'
DATA_BENCHMARK_DIR = '/workspace/tests/data'


def fetch_approved_drugs(chembl_db_path=CHEMBL_DB_PATH):
    """Fetch approved drugs with phase >=3 as dataframe

    Args:

        chembl_db_path (string): path to chembl sqlite database
    Returns:
        pd.DataFrame: dataframe containing SMILES strings and molecule index
    """

    sql_query = """SELECT
        di.molregno,
        cs.canonical_smiles,
        di.max_phase_for_ind
    FROM
        drug_indication AS di
    LEFT JOIN compound_structures AS cs ON di.molregno = cs.molregno
    WHERE
        di.max_phase_for_ind >= 3
        AND cs.canonical_smiles IS NOT NULL;"""

    chembl_db_url = 'file:%s?mode=ro' % chembl_db_path
    conn = sqlite3.connect(chembl_db_url, uri=True)
    benchmark_df = pd.read_sql_query(sql_query, conn).drop_duplicates(subset='molregno').set_index('molregno')

    return benchmark_df


def calc_morgan_figerprints(benchmark_df, smiles_col='canonical_smiles'):
    """Calculate Morgan fingerprints on SMILES strings

    Args:
        benchmark_df (pd.DataFrame): dataframe containing a SMILES column for calculation

    Returns:
        pd.DataFrame: new dataframe containing fingerprints
    """
    mf = MorganFingerprint()
    fp = mf.transform(benchmark_df[[smiles_col]])
    fp.index = benchmark_df.index

    return fp


if __name__ == '__main__':

    benchmark_df = fetch_approved_drugs().set_index('molregno').sort_index()
    benchmark_df.index = benchmark_df.index.astype(np.int64)

    # TODO: benchmark SMILES have not been canonicalized. Should this be done?
    fp = calc_morgan_figerprints(benchmark_df)
    fp.columns = fp.columns.astype(np.int64)
    fp.index = fp.index.astype(np.int64)
    for col in fp.columns:
        fp[col] = fp[col].astype(np.float32)

    # Write results
    benchmark_df.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'benchmark_approved_drugs.csv'))
    fp.to_csv(os.path.join(DATA_BENCHMARK_DIR, 'fingerprints_approved_drugs.csv'))
    fp_hdf5 = cudf.DataFrame(fp)
    fp_hdf5.to_hdf(os.path.join(DATA_BENCHMARK_DIR, 'filter_00.h5', 'fingerprints', format='table'))
