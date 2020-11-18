from rdkit import Chem
from rdkit.Chem import AllChem

import cupy
import numpy as np


def morgan_fingerprint(smiles, molregno=None, radius=2, nBits=512):
    m = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    ar = np.array(fp)
    
    if molregno:
        ar = np.concatenate((np.array([molregno]), ar))

    return ar

# def morgan_fingerprint(smiles, molregno=None, radius=2, nBits=512):
#     m = Chem.MolFromSmiles(smiles)
#     fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
#     ar = cupy.array(fp)

#     if molregno:
#         ar = cupy.concatenate((cupy.array([molregno]), ar))

#     return ar
