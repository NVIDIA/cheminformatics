import logging
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
        self.func = AllChem.GetMorganFingerprintAsBitVect

    def transform(self, data):
        m = Chem.MolFromSmiles(data)
        fp = self.func(m, **self.kwargs)
        return ', '.join(list(fp.ToBitString()))

    def __len__(self):
        return self.kwargs['nBits']
        

class Embeddings(BaseTransformation):
    MODEL_DIR = '/workspace/cddd/default_model'

    def __init__(self, use_gpu=True, cpu_threads=5, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = TransformationDefaults[self.name].value
        self.kwargs.update(kwargs)
        self.func = InferenceModel(self.MODEL_DIR, use_gpu=use_gpu, cpu_threads=cpu_threads)

    def transform(self, data):
        if isinstance(data, str):
            data = [data]
        return self.func.seq_to_emb(data)

    def __len__(self):
        return self.func.hparams.emb_size


### DEPRECATED ###
def morgan_fingerprint(smiles, radius=2, nBits=512):
    m = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)

    return ', '.join(list(fp.ToBitString()))

