#!/usr/bin/env python3
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd


# TODO RAJESH this is a version of function that is decoupled from MorganFingerprint Transformer -- happy to discuss other options for fector
def calc_morgan_fingerprints(dataframe, smiles_col='canonical_smiles'):
    """Calculate Morgan fingerprints on SMILES strings

    Args:
        dataframe (pd.DataFrame): dataframe containing a SMILES column for calculation

    Returns:
        pd.DataFrame: new dataframe containing fingerprints
    """
    default_args = {'radius': 2, 'nBits': 512}
    data = dataframe[smiles_col]
    fp_array = []
    for smiles in data:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, **default_args)
            fp = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
        else:
            fp = None
        fp_array.append(fp)
    fp_array = np.asarray(fp_array)
    return pd.DataFrame(fp_array)

    
