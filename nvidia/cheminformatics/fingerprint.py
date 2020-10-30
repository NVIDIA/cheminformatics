import logging

from rdkit import Chem
from rdkit.Chem import AllChem


logger = logging.getLogger(__name__)


def morgan_fingerprint(smiles, radius=2, nBits=512):
    m = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)

    return ', '.join(list(fp.ToBitString()))
