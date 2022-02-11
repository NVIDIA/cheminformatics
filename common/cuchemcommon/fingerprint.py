import logging
import os
from abc import ABC
from enum import Enum

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from math import ceil


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

    def transform(self, data, smiles_column = 'transformed_smiles'):
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

    def transform(self, data, col_name='transformed_smiles', return_fp=False, raw=False):
        """Single threaded processing of list"""
        data = data[col_name]
        fp_array = []
        self.n_fp_integers = ceil(self.kwargs['nBits'] / INTEGER_NBITS)
        if raw:
            raw_fp_array = []
        else:
            raw_fp_array = [[] for i in range(0, self.kwargs['nBits'], INTEGER_NBITS)]
        for smiles in data:
            fp = self.transform_single(smiles)
            fp_array.append(fp)
            fp_bs = fp.ToBitString()
            if return_fp:
                if raw:
                    raw_fp_array.append(fp)
                else:
                    for i in range(0, self.kwargs['nBits'], INTEGER_NBITS):
                        raw_fp_array[i // INTEGER_NBITS].append(int(fp_bs[i: i + INTEGER_NBITS], 2))
        fp_array = cupy.stack(fp_array)
        if return_fp:
            if raw:
                return fp_array, raw_fp_array
            else:
                return fp_array, np.asarray(raw_fp_array, dtype=np.uint64)        
        return fp_array

    def __len__(self):
        return self.kwargs['nBits']
