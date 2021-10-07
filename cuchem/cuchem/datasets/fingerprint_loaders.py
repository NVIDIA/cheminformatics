import logging
import os
import pathlib

import cudf

logger = logging.getLogger(__name__)


class GenericFingerprintDataset():
    def __init__(self):
        self.name = None
        self.index_col = None
        self.data = None
        self.data_path = None

    def load(self, index=None):
        data = cudf.read_csv(self.data_path)
        if self.index_col:
            data = data.set_index(self.index_col).sort_index()

        if index is not None:
            data = data.loc[index]
        self.data = data
        return


class ChEMBL_Approved_Drugs_Fingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'ChEMBL Approved Drugs (Phase III/IV) Fingerprints'

        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_ChEMBL_approved_drugs_physchem.csv')
        assert os.path.exists(self.data_path)


class MoleculeNet_Lipophilicity_Fingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'MoleculeNet Lipophilicity Fingerprints'
        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_MoleculeNet_Lipophilicity.csv')
        assert os.path.exists(self.data_path)


class MoleculeNet_ESOL_Fingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'MoleculeNet ESOL Fingerprints'
        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_MoleculeNet_ESOL.csv')
        assert os.path.exists(self.data_path)


class MoleculeNet_FreeSolv_Fingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'MoleculeNet FreeSolv Fingerprints'
        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_MoleculeNet_FreeSolv.csv')
        assert os.path.exists(self.data_path)


class ZINC15_TestSplit_20K_Fingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='index'):
        self.name = 'ZINC15 Test Split 20K Fingerprints'

        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_ZINC15_test_split.csv')
        assert os.path.exists(self.data_path)


### DEPRECATED ###
class ChEMBL_20K_Fingerprints(GenericFingerprintDataset):
    def __init__(self, index_col='molregno'):
        self.name = 'ChEMBL 20K Fingerprints'
        logger.warn(f'Class {self.name} is deprecated.')
        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),
                                      'data',
                                      'fingerprints_ChEMBL_random_sampled_drugs.csv')
        assert os.path.exists(self.data_path)