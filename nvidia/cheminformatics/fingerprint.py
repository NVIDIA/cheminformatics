import logging

import pandas as pd

from abc import ABC
from enum import Enum
from rdkit import Chem
from rdkit.Chem import AllChem
from cddd.inference import InferenceModel

logger = logging.getLogger(__name__)

class TransformationDefaults(Enum):
    morgan_fingerprint = {'radius':2, 'nBits':512}
    MorganFingerprint = {'radius':2, 'nBits':512}
    Embeddings = {}


class BaseTransformation(ABC):
    def __init__(self, **kwargs):
        self.name = None
        self.kwargs = None
        self.func = None

    def transform(self, data):
        return NotImplemented

    def transform_many(self, data):
        return map(self.transform, data)

    def __len__(self):
        return NotImplemented


class MorganFingerprint(BaseTransformation):

    def __init__(self, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = TransformationDefaults[self.name].value
        self.kwargs.update(kwargs)

    def _morgan_fingerprint(self, smiles, radius=2, nBits=512):
        m = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
        return ', '.join(list(fp.ToBitString()))

    def transform(self, df):
        df['fp'] = df.apply(
            lambda row:
            self._morgan_fingerprint(row.canonical_smiles, **self.kwargs),
            axis=1)
        return df['fp'].str.split(pat=', ',
                                  n=len(self)+1,
                                  expand=True).astype('float32')

    def __len__(self):
        return self.kwargs['nBits']


class Embeddings(BaseTransformation):
    MODEL_DIR = '/opt/nvidia/cheminfomatics/cddd/default_model'

    def __init__(self, use_gpu=True, cpu_threads=12, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = TransformationDefaults[self.name].value
        self.kwargs.update(kwargs)
        self.func = InferenceModel(self.MODEL_DIR, use_gpu=use_gpu, cpu_threads=cpu_threads)

    def transform(self, df):
        smiles = df['canonical_smiles'].tolist()
        smiles_embedding = self.func.seq_to_emb(smiles)

        result_df = pd.DataFrame.from_records(smiles_embedding)
        result_df.index = df.index
        return result_df

    def __len__(self):
        return self.func.hparams.emb_size
