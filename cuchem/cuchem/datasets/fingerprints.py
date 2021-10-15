import logging
import os
import pathlib
import cudf
from .base import GenericFingerprintDataset

__all__ = ['ChEMBLApprovedDrugsFingerprints', 'MoleculeNetLipophilicityFingerprints', 'MoleculeNetESOLFingerprints', 'MoleculeNetFreeSolvFingerprints', 'ZINC15TestSplitFingerprints']

logger = logging.getLogger(__name__)

class ChEMBLApprovedDrugsFingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'ChEMBL Approved Drugs (Phase III/IV) Fingerprints'

        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_ChEMBL_approved_drugs_physchem.csv')
        assert os.path.exists(self.data_path)


class MoleculeNetLipophilicityFingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'MoleculeNet Lipophilicity Fingerprints'
        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_MoleculeNet_Lipophilicity.csv')
        assert os.path.exists(self.data_path)


class MoleculeNetESOLFingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'MoleculeNet ESOL Fingerprints'
        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_MoleculeNet_ESOL.csv')
        assert os.path.exists(self.data_path)


class MoleculeNetFreeSolvFingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'MoleculeNet FreeSolv Fingerprints'
        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_MoleculeNet_FreeSolv.csv')
        assert os.path.exists(self.data_path)


class ZINC15TestSplitFingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'ZINC15 Test Split 20K Fingerprints'

        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_ZINC15_test_split.csv')
        assert os.path.exists(self.data_path)


### DEPRECATED ###
class ChEMBL20KFingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='molregno'):
        self.name = 'ChEMBL 20K Fingerprints'
        logger.warn(f'Class {self.name} is deprecated.')
        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_ChEMBL_random_sampled_drugs.csv')
        assert os.path.exists(self.data_path)