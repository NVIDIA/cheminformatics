from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np


def morgan_fingerprint(smiles, radius=2, nBits=512):
    # print('------>', smiles['canonical_smiles'])
    # print('------>', smiles)
    # print('------>', dir(smiles) )
    # print('------>', smiles.columns )

    m = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    return np.array(fp)