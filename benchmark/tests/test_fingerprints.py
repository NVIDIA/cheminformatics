import logging
import pandas as pd

from cuchembench.utils.smiles import calc_morgan_fingerprints

logger = logging.getLogger(__name__)


def test_smiles_fingerprint():

    df_smiles = pd.read_csv('cuchembench/csv_data/benchmark_MoleculeNet_FreeSolv.csv')

    fp = calc_morgan_fingerprints(df_smiles, smiles_col='SMILES')
    shape = fp.shape

    # Ensure not all values are True or False
    assert (fp.sum() == 0).sum() != shape[0]
    assert (fp.sum() == shape[1]).sum() != shape[0]
