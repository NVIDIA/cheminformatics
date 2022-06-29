import logging
from .base import GenericCSVDataset

__all__ = ['ExCAPEDataset']


logger = logging.getLogger(__name__)


class ExCAPEDataset(GenericCSVDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'ExCAPE'
        self.table_name = 'excape'
        self.smiles_col = 'SMILES'
        self.index_col = 'index'
        self.properties_cols = ['pXC50']

    @staticmethod
    def _truncate_data(data, data_len):
        orig_index = data.index.name
        genes = data['Gene_Symbol'].unique()[:data_len]
        dat = data.reset_index().set_index('Gene_Symbol').loc[genes]
        return dat.reset_index().set_index(orig_index)

    def load(self,
             columns=['canonical_smiles'],
             length_column='length',
             data_len=None,
             nbits = 512):

        super().load(columns, length_column, data_len, nbits)
        self.smiles = self.smiles.rename(columns={'Gene_Symbol': 'gene'}).set_index('gene', append=True)
