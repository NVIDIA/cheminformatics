import logging
import os
import pandas as pd
from .base import GenericCSVDataset
from chembench.utils.smiles import calculate_morgan_fingerprint
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except:
    DASK_AVAILABLE = False


logger = logging.getLogger(__name__)

__all__ = ['ExCAPEDataset', 'BIOACTIVITY_TABLE_LIST']
BIOACTIVITY_TABLE_LIST = ['excape_activity', 'excape_fp']

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

# #### Deprecated ####

# GENE_SYMBOLS = ["ABL1", "ACHE", "ADAM17", "ADORA2A", "ADORA2B", "ADORA3", "ADRA1A", "ADRA1D",
#              "ADRB1", "ADRB2", "ADRB3", "AKT1", "AKT2", "ALK", "ALOX5", "AR", "AURKA",
#              "AURKB", "BACE1", "CA1", "CA12", "CA2", "CA9", "CASP1", "CCKBR", "CCR2",
#              "CCR5", "CDK1", "CDK2", "CHEK1", "CHRM1", "CHRM2", "CHRM3", "CHRNA7", "CLK4",
#              "CNR1", "CNR2", "CRHR1", "CSF1R", "CTSK", "CTSS", "CYP19A1", "DHFR", "DPP4",
#              "DRD1", "DRD3", "DRD4", "DYRK1A", "EDNRA", "EGFR", "EPHX2", "ERBB2", "ESR1",
#              "ESR2", "F10", "F2", "FAAH", "FGFR1", "FLT1", "FLT3", "GHSR", "GNRHR", "GRM5",
#              "GSK3A", "GSK3B", "HDAC1", "HPGD", "HRH3", "HSD11B1", "HSP90AA1", "HTR2A",
#              "HTR2C", "HTR6", "HTR7", "IGF1R", "INSR", "ITK", "JAK2", "JAK3", "KCNH2",
#              "KDR", "KIT", "LCK", "MAOB", "MAPK14", "MAPK8", "MAPK9", "MAPKAPK2", "MC4R",
#              "MCHR1", "MET", "MMP1", "MMP13", "MMP2", "MMP3", "MMP9", "MTOR", "NPY5R",
#              "NR3C1", "NTRK1", "OPRD1", "OPRK1", "OPRL1", "OPRM1", "P2RX7", "PARP1", "PDE5A",
#              "PDGFRB", "PGR", "PIK3CA", "PIM1", "PIM2", "PLK1", "PPARA", "PPARD", "PPARG",
#              "PRKACA", "PRKCD", "PTGDR2", "PTGS2", "PTPN1", "REN", "ROCK1", "ROCK2", "S1PR1",
#              "SCN9A", "SIGMAR1", "SLC6A2", "SLC6A3", "SRC", "TACR1", "TRPV1", "VDR"]
