import logging
from .base import GenericCSVDataset


logger = logging.getLogger(__name__)


__all__ = ['ChEMBLApprovedDrugs',
           'MoleculeNetLipophilicity',
           'MoleculeNetESOL',
           'MoleculeNetFreeSolv',
           'ZINC15TestSplit',
           'PHYSCHEM_TABLE_LIST']

# must match datasets table_names, could cycle through classes to get them
PHYSCHEM_TABLE_LIST = ['chembl', 'lipophilicity',
                       'esol', 'freesolv',
                       'zinc15_test']


class ChEMBLApprovedDrugs(GenericCSVDataset):
    def __init__(self, **kwargs):
        super().__init__(data_filename='benchmark_ChEMBL_approved_drugs_physchem.csv',
                         fp_filename='fingerprints_ChEMBL_approved_drugs_physchem.csv',
                         **kwargs)
        self.name = 'ChEMBL Approved Drugs (Phase III/IV)'
        self.table_name = 'chembl'
        self.index_col = 'index'
        self.properties_cols = ['max_phase_for_ind', 'mw_freebase',
                                'alogp', 'hba', 'hbd', 'psa', 'rtb', 'ro3_pass', 'num_ro5_violations',
                                'cx_logp', 'cx_logd', 'full_mwt', 'aromatic_rings', 'heavy_atoms',
                                'qed_weighted', 'mw_monoisotopic', 'hba_lipinski', 'hbd_lipinski',
                                'num_lipinski_ro5_violations']


class MoleculeNetLipophilicity(GenericCSVDataset):
    def __init__(self, **kwargs):
        super().__init__(data_filename='benchmark_MoleculeNet_Lipophilicity.csv',
                         fp_filename='fingerprints_MoleculeNet_Lipophilicity.csv',
                         **kwargs)
        self.name = 'MoleculeNet Lipophilicity'
        self.table_name = 'lipophilicity'
        self.smiles_col = 0
        self.properties_cols = ['logD']
        self.orig_property_name = ['exp']


class MoleculeNetESOL(GenericCSVDataset):
    def __init__(self, **kwargs):
        super().__init__(data_filename='benchmark_MoleculeNet_ESOL.csv',
                         fp_filename='fingerprints_MoleculeNet_ESOL.csv',
                         **kwargs)
        self.name = 'MoleculeNet ESOL'
        self.table_name = 'esol'
        self.smiles_col = 0
        self.properties_cols = ['log_solubility_(mol_per_L)']
        self.orig_property_name = ['measured log solubility in mols per litre']


class MoleculeNetFreeSolv(GenericCSVDataset):
    def __init__(self, **kwargs):
        super().__init__(data_filename='benchmark_MoleculeNet_FreeSolv.csv',
                         fp_filename='fingerprints_MoleculeNet_FreeSolv.csv',
                         **kwargs)
        self.name = 'MoleculeNet FreeSolv'
        self.table_name = 'freesolv'
        self.smiles_col = 0
        self.properties_cols = ['hydration_free_energy']
        self.orig_property_name = ['y']


class ZINC15TestSplit(GenericCSVDataset):
    def __init__(self, **kwargs):
        super().__init__(data_filename='benchmark_ZINC15_test_split.csv',
                         fp_filename='fingerprints_ZINC15_test_split.csv',
                         **kwargs)
        self.name = 'ZINC15 Test Split 20K Samples'
        self.table_name = 'zinc15_test'
        self.properties_cols = ['logp', 'mw']
        self.index_col = 'index'

    def load(self, columns=['canonical_smiles'], length_column='length', data_len=None):
        self.data, _ = self._load_csv(columns, length_column, return_remaining=False, data_len=None)

        # if data_len:
        #     self.data = self.data.iloc[:data_len]

        if self.data.shape[-1] == 1: # TODO this is a fix for an issue with SQL data cache, should improve datacache
            self.data = self.data[columns[0]]
