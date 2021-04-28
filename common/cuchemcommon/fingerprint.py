import numpy as np

import logging
from abc import ABC
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


class BaseTransformation(ABC):
    def __init__(self, **kwargs):
        self.name = None
        self.kwargs = None
        self.func = None

    def transform(self, data):
        return NotImplemented

    def transform_many(self, data):
        return list(map(self.transform, data))

    def __len__(self):
        return NotImplemented

class MorganFingerprint(BaseTransformation):

    DEFAULTS = {'radius':2, 'nBits':512}

    def __init__(self, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = MorganFingerprint.DEFAULTS.value
        self.kwargs.update(kwargs)
        self.func = AllChem.GetMorganFingerprintAsBitVect

    def transform(self, data):
        data = data['transformed_smiles']
        fp_array = []
        for mol in data:
            m = Chem.MolFromSmiles(mol)
            fp = self.func(m, **self.kwargs)
            fp_array.append(list(fp.ToBitString()))
        fp_array = np.asarray(fp_array)
        return fp_array

    def __len__(self):
        return self.kwargs['nBits']
