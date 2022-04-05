import logging
import os
from abc import ABC
from enum import Enum

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from cuchemcommon.utils.smiles import calculate_morgan_fingerprint


logger = logging.getLogger(__name__)

import cupy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class TransformationDefaults(Enum):
    MorganFingerprint = {'radius': 2, 'nBits': 512}
    Embeddings = {}


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

    def __init__(self, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = TransformationDefaults[self.name].value
        self.kwargs.update(kwargs)
        self.func = AllChem.GetMorganFingerprintAsBitVect

    def transform_single(self, smiles):
        """Process single molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = self.func(mol, **self.kwargs)
            fp = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
        else:
            logger.warn(f'WARNING: Invalid SMILES identified {smiles}')
            fp = np.array([0 for _ in range(self.kwargs['nBits'])], dtype=np.uint8)

        fp = cupy.asarray(fp)
        return fp

    def transform(self, data, col_name='transformed_smiles'):
        """Single threaded processing of list"""
        fp = calculate_morgan_fingerprint(data[col_name].values,
                                          self.kwargs['radius'],
                                          self.kwargs['nBits'])
        return fp

    def __len__(self):
        return self.kwargs['nBits']
