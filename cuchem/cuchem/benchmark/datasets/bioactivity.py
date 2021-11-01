import logging
import os
import pandas as pd
import dask.dataframe as dd

from cuchemcommon.fingerprint import calc_morgan_fingerprints

logger = logging.getLogger(__name__)

__all__ = ['ExCAPEBioactivity', 'ExCAPEFingerprints', 'BIOACTIVITY_TABLE_LIST']
BIOACTIVITY_TABLE_LIST = ['excape_activity', 'excape_fp']

GENE_SYMBOLS = ["ABL1", "ACHE", "ADAM17", "ADORA2A", "ADORA2B", "ADORA3", "ADRA1A", "ADRA1D",
             "ADRB1", "ADRB2", "ADRB3", "AKT1", "AKT2", "ALK", "ALOX5", "AR", "AURKA",
             "AURKB", "BACE1", "CA1", "CA12", "CA2", "CA9", "CASP1", "CCKBR", "CCR2",
             "CCR5", "CDK1", "CDK2", "CHEK1", "CHRM1", "CHRM2", "CHRM3", "CHRNA7", "CLK4",
             "CNR1", "CNR2", "CRHR1", "CSF1R", "CTSK", "CTSS", "CYP19A1", "DHFR", "DPP4",
             "DRD1", "DRD3", "DRD4", "DYRK1A", "EDNRA", "EGFR", "EPHX2", "ERBB2", "ESR1",
             "ESR2", "F10", "F2", "FAAH", "FGFR1", "FLT1", "FLT3", "GHSR", "GNRHR", "GRM5",
             "GSK3A", "GSK3B", "HDAC1", "HPGD", "HRH3", "HSD11B1", "HSP90AA1", "HTR2A",
             "HTR2C", "HTR6", "HTR7", "IGF1R", "INSR", "ITK", "JAK2", "JAK3", "KCNH2",
             "KDR", "KIT", "LCK", "MAOB", "MAPK14", "MAPK8", "MAPK9", "MAPKAPK2", "MC4R",
             "MCHR1", "MET", "MMP1", "MMP13", "MMP2", "MMP3", "MMP9", "MTOR", "NPY5R",
             "NR3C1", "NTRK1", "OPRD1", "OPRK1", "OPRL1", "OPRM1", "P2RX7", "PARP1", "PDE5A",
             "PDGFRB", "PGR", "PIK3CA", "PIM1", "PIM2", "PLK1", "PPARA", "PPARD", "PPARG",
             "PRKACA", "PRKCD", "PTGDR2", "PTGS2", "PTPN1", "REN", "ROCK1", "ROCK2", "S1PR1",
             "SCN9A", "SIGMAR1", "SLC6A2", "SLC6A3", "SRC", "TACR1", "TRPV1", "VDR"]


class ExCAPEDataset():
    def __init__(self,
                 data_dir='/data/ExCAPE',
                 name = 'ExCAPE',
                 table_name = 'excape',
                 properties_cols = ['pXC50'],
                 max_seq_len = None,
                 ):
        self.name = name
        self.table_name = table_name
        self.properties_cols = properties_cols
        self.max_seq_len = max_seq_len
        self.raw_data_path = os.path.join(data_dir, 'raw_data.csv')
        self.filter_data_path = os.path.join(data_dir, 'filtered_data.csv')

    def filter_data(self, filter_col='Gene_Symbol', filter_list=GENE_SYMBOLS):
        data = dd.read_csv(self.raw_data_path,
                            delimiter='\t',
                            usecols=['SMILES', filter_col] + self.properties_cols)

        filtered_df = data[data[filter_col].isin(filter_list)]

        filtered_df = filtered_df.compute()
        filtered_df.index.name = 'index'
        filtered_df = filtered_df.sort_values([filter_col] + self.properties_cols).reset_index(drop=True)
        filtered_df = filtered_df.rename(columns={'SMILES': 'canonical_smiles', 'Gene_Symbol': 'gene'})

        # Save for later use.
        logger.info('Saving filtered database records...')
        filtered_df.reset_index().to_csv(self.filter_data_path, index=False)
        return filtered_df

    def load(self, data_len=None):
        if os.path.exists(self.filter_data_path):
            data = pd.read_csv(self.filter_data_path)
            data = data.set_index('index')
        else:
            logger.info('Filtered data not found, loading from RAW data')
            data = self.filter_data()

        if self.max_seq_len:
            data = data[data['canonical_smiles'].str.len() <= self.max_seq_len]
        else:
            self.max_seq_len = data['canonical_smiles'].str.len().max()

        if data_len:
            data = data.iloc[:data_len]

        self.data = data

class ExCAPEBioactivity(ExCAPEDataset):
    def __init__(self, data_dir='/data/ExCAPE', max_seq_len=None):
        super().__init__(data_dir=data_dir,
                         name = 'ExCAPE Bioactivity',
                         table_name = 'excape_activity',
                         max_seq_len = max_seq_len)

        self.raw_data_path = os.path.join(data_dir, 'pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv')
        self.filter_data_path = os.path.join(data_dir, 'ExCAPE_filtered_data.csv')

    def load(self, data_len=None):
        super().load(data_len=data_len)
        self.properties = self.data[['pXC50', 'gene']].reset_index().set_index(['gene', 'index'])
        self.data = self.data[['canonical_smiles', 'gene']].reset_index().set_index(['gene', 'index'])

    def remove_invalids_by_index(self, fingerprint_dataset):
        mask = self.data.index.isin(fingerprint_dataset.data.index)
        num_invalid_molecules = len(self.data) - mask.sum()
        if num_invalid_molecules > 0:
            logger.info(f'Removing {num_invalid_molecules} entry from dataset based on index matching.')
            self.data = self.data[mask]
            self.properties = self.properties[mask]

class ExCAPEFingerprints(ExCAPEDataset):
    def __init__(self, data_dir='/data/ExCAPE', max_seq_len=None):
        super().__init__(data_dir=data_dir,
                         name = 'ExCAPE Fingerprints',
                         table_name = 'excape_fp',
                         max_seq_len = max_seq_len)

        self.raw_data_path = os.path.join(data_dir, 'pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv')
        self.filter_data_path = os.path.join(data_dir, 'ExCAPE_filtered_data.csv')

    def load(self, data_len=None):
        super().load(data_len=data_len)

        # very few repeated SMILES, so probably not worth making unique and then merging
        fp = calc_morgan_fingerprints(self.data, smiles_col='canonical_smiles', remove_invalid=False)

        index_name = self.data.index.name
        fp[index_name] = self.data.index.values
        fp['gene'] = self.data['gene']
        fp = fp.set_index(['gene', index_name])

        # Prune molecules which failed to convert
        valid_molecule_mask = (fp.sum(axis=1) > 0)
        num_invalid_molecules = int(valid_molecule_mask.shape[0] - valid_molecule_mask.sum())
        if num_invalid_molecules > 0:
            logger.warn(f'WARNING: fingerprint dataset length does not match that of SMILES dataset. Run `remove_invalids_by_index` on SMILES dataset to ensure they match.')
            fp = fp[valid_molecule_mask]

        if not isinstance(fp, pd.DataFrame):
            fp = fp.to_pandas()
        self.data = fp
