from nvidia.cheminformatics.utils.data_peddler import CDDD_DEFAULT_MODLE_LOC
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import pandas as pd
import numpy as np

import logging
from abc import ABC
from enum import Enum
from rdkit import Chem
from rdkit.Chem import AllChem
from cddd.inference import InferenceModel

from nvidia.cheminformatics.utils.data_peddler import download_cddd_models

logger = logging.getLogger(__name__)

class TransformationDefaults(Enum):
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
        return list(map(self.transform, data))

    def __len__(self):
        return NotImplemented

class MorganFingerprint(BaseTransformation):

    def __init__(self, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = TransformationDefaults[self.name].value
        self.kwargs.update(kwargs)
        self.func = AllChem.GetMorganFingerprintAsBitVect

    def transform(self, data):
        smile = data['transformed_smiles']
        m = Chem.MolFromSmiles(smile)
        fp = self.func(m, **self.kwargs)
        return list(fp.ToBitString())

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
        smile = data['transformed_smiles']
        if isinstance(smile, str):
            smile = [smile]
        return self.func.seq_to_emb(smile).squeeze()

    def inverse_transform(self, embeddings):
        "Embedding array -- individual compound embeddings are in rows"
        embeddings = np.asarray(embeddings)
        return self.func.emb_to_seq(embeddings)

    def __len__(self):
        return self.func.hparams.emb_size
