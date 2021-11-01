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

logger = logging.getLogger(__name__)

try:
    import cupy # TODO is there a better way to check for RAPIDS?
except:
    logger.info('RAPIDS installation not found. Numpy and pandas will be used instead.')
    import numpy as xpy
    import pandas as xdf
else:
    logger.info('RAPIDS installation found. Using cupy and cudf where possible.')
    import cupy as xpy
    import cudf as xdf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TODO RAJESH the morgan fingerprint class and this function will need to be separated from those which require TF here
def calc_morgan_fingerprints(dataframe, smiles_col='canonical_smiles', remove_invalid=True):
    """Calculate Morgan fingerprints on SMILES strings

    Args:
        dataframe (pandas/cudf.DataFrame): dataframe containing a SMILES column for calculation
        remove_invalid (bool): remove fingerprints from failed SMILES conversion, default: True

    Returns:
        pandas/cudf.DataFrame: new dataframe containing fingerprints
    """
    mf = MorganFingerprint()
    fp = mf.transform(dataframe, col_name=smiles_col)
    fp = xdf.DataFrame(fp)
    fp.index = dataframe.index

    # Check for invalid smiles
    # Prune molecules which failed to convert or throw error and exit
    valid_molecule_mask = (fp.sum(axis=1) > 0)
    num_invalid_molecules = int(valid_molecule_mask.shape[0] - valid_molecule_mask.sum())
    if num_invalid_molecules > 0:
        logger.warn(f'WARNING: Found {num_invalid_molecules} invalid fingerprints due to invalid SMILES during fingerprint creation')
        if remove_invalid:
            logger.info(f'Removing {num_invalid_molecules} invalid fingerprints due to invalid SMILES during fingerprint creation')
            fp = fp[valid_molecule_mask]

    return fp


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

        fp = xpy.asarray(fp)
        return fp

    def transform(self, data, col_name='transformed_smiles'):
        """Single threaded processing of list"""
        data = data[col_name]
        fp_array = []
        for smiles in data:
            fp = self.transform_single(smiles)
            fp_array.append(fp)
        fp_array = xpy.stack(fp_array)
        return fp_array

    def __len__(self):
        return self.kwargs['nBits']


# TODO RAJESH this will have to be moved to a different file during refactor due to TensorFlow requirement
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
