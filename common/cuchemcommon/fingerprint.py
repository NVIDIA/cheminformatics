import logging
import os
from abc import ABC
from enum import Enum

import numpy as np
import pandas as pd
from cddd.inference import InferenceModel
from cuchem.utils.data_peddler import download_cddd_models
from rdkit import Chem
from rdkit.Chem import AllChem
from math import ceil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logging.getLogger(__name__)
INTEGER_NBITS = 64 # Maximum number of bits in an integer column in a cudf Series


def calc_morgan_fingerprints(dataframe, smiles_column='canonical_smiles'):
    """Calculate Morgan fingerprints on SMILES strings

    Args:
        dataframe (pd.DataFrame): dataframe containing a SMILES column for calculation

    Returns:
        pd.DataFrame: new dataframe containing fingerprints
    """
    mf = MorganFingerprint()
    fp = mf.transform(dataframe, smiles_column=smiles_column)
    fp = pd.DataFrame(fp)
    fp.index = dataframe.index
    return fp


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

    def transform(self, data, smiles_column='transformed_smiles', return_fp=False, raw=False):
        data = data[smiles_column]
        fp_array = []
        self.n_fp_integers = ceil(self.kwargs['nBits'] / INTEGER_NBITS)
        if raw:
            raw_fp_array = []
        else:
            raw_fp_array = [[] for i in range(0, self.kwargs['nBits'], INTEGER_NBITS)]
        for mol_smiles in data:
            m = Chem.MolFromSmiles(mol_smiles)
            fp = self.func(m, **self.kwargs)
            fp_bs = fp.ToBitString()
            fp_array.append(list(fp_bs))
            if return_fp:
                if raw:
                    raw_fp_array.append(fp)
                else:
                    for i in range(0, self.kwargs['nBits'], INTEGER_NBITS):
                        raw_fp_array[i // INTEGER_NBITS].append(int(fp_bs[i: i + INTEGER_NBITS], 2))
        fp_array = np.asarray(fp_array)
        if return_fp:
            if raw:
                return fp_array, raw_fp_array
            else:
                return fp_array, np.asarray(raw_fp_array, dtype=np.uint64)
        return fp_array

    def __len__(self):
        return self.kwargs['nBits']


class Embeddings(BaseTransformation):

    def __init__(self, use_gpu=True, cpu_threads=5, model_dir=None, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = TransformationDefaults[self.name].value
        self.kwargs.update(kwargs)
        model_dir = download_cddd_models()
        self.func = InferenceModel(model_dir, use_gpu=use_gpu, cpu_threads=cpu_threads)

    def transform(self, data):
        data = data['transformed_smiles']
        return self.func.seq_to_emb(data).squeeze()

    def inverse_transform(self, embeddings):
        "Embedding array -- individual compound embeddings are in rows"
        embeddings = np.asarray(embeddings)
        return self.func.emb_to_seq(embeddings)

    def __len__(self):
        return self.func.hparams.emb_size
