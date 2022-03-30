import logging
import os
import pandas as pd
from .base import GenericCSVDataset
from cuchembench.utils.smiles import calculate_morgan_fingerprint
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
        super().__init__(data_filename='benchmark_ExCAPE_Bioactivity.csv',
                         fp_filename='fingerprints_ExCAPE_Bioactivity.csv',
                         **kwargs)
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
             data_len=None):

        super().load(columns, length_column, data_len)
        self.smiles = self.smiles.rename(columns={'Gene_Symbol': 'gene'}).set_index('gene', append=True)

#### Deprecated ####

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

class FullExCAPEDataset():
    def __init__(self,
                 data_dir='/data/ExCAPE',
                 name = 'ExCAPE',
                 table_name = 'excape',
                 properties_cols = ['pXC50'],
                 max_seq_len = None,
                 ):
        assert DASK_AVAILABLE, AssertionError(f'Dask is not available and is required for processing full ExCAPE dataset.')
        logger.warn(f'Download and processing of full ExCAPE db data is deprecated and will produce different results from included data tranche.')

        self.name = name
        self.table_name = table_name
        self.properties_cols = properties_cols
        self.max_seq_len = max_seq_len

        # All directories and files to be used for ExCAPE DB.
        self.excape_dir = data_dir
        self.raw_data_path = os.path.join(self.excape_dir, 'publication_inchi_smiles_v2.tsv')
        self.filter_data_path = os.path.join(self.excape_dir, 'filtered_data.csv')
        if not os.path.exists(self.excape_dir):
            os.makedirs(self.excape_dir)

        # Relavent data loaded and computed using input ExCAPE DB.
        self.smiles = None
        self.properties = None
        self.fingerprints = None

    def _download(self):
        """
        Download ExCAPE database if not already downloaded.
        """
        if not os.path.exists(self.raw_data_path):
            excape_file = os.path.join(self.excape_dir, 'publication_inchi_smiles_v2.tsv.xz')
            if not os.path.exists(excape_file):
                logger.info('Downloading ExCAPE data...')
                os.system(f'wget -O {excape_file} https://zenodo.org/record/2543724/files/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz?download=1')

            logger.info(f'Extracting ExCAPE db {excape_file}...')
            os.system(f'cd {self.excape_dir} && xz -k -d {excape_file}')

    def filter_data(self, filter_col='Gene_Symbol', filter_list=GENE_SYMBOLS):
        """
        Filter ExCAPE data to only include genes of interest.
        """

        if os.path.exists(self.filter_data_path):
            logger.debug('Using existing filtered ExCAPE data...')
            return pd.read_csv(self.filter_data_path)

        self._download()
        logger.info(f'Loading ExCAPE file from {self.raw_data_path}...')
        data = dd.read_csv(self.raw_data_path,
                            delimiter='\t',
                            usecols=['SMILES', filter_col] + self.properties_cols)

        filtered_df = data[data[filter_col].isin(filter_list)]

        filtered_df = filtered_df.compute()
        filtered_df = filtered_df.sort_values([filter_col] + self.properties_cols).reset_index(drop=True)
        filtered_df = filtered_df.rename(columns={'SMILES': 'canonical_smiles', 'Gene_Symbol': 'gene'})

        # Save for later use.
        logger.info('Saving filtered database records...')
        filtered_df.reset_index().to_csv(self.filter_data_path, index=False)
        filtered_df['index'] = filtered_df.index
        return filtered_df

    def _remove_invalids_by_index(self):

        if self.fingerprints is None:
            raise ValueError('Fingerprint data not loaded. Run `load` first.')

        mask = self.smiles.index.isin(self.fingerprints.index)
        num_invalid_molecules = len(self.smiles) - mask.sum()

        if num_invalid_molecules > 0:
            logger.info(f'Removing {num_invalid_molecules} entry from dataset based on index matching.')
            self.smiles = self.smiles[mask]
            self.properties = self.properties[mask]

    def load(self, data_len=None):
        """
        Load ExCAPE data.
        """
        data = self.filter_data()
        data = data.set_index(['gene', 'index'])

        if self.max_seq_len:
            data = data[data['canonical_smiles'].str.len() <= self.max_seq_len]
        else:
            self.max_seq_len = data['canonical_smiles'].str.len().max()

        if data_len:
            data = data.groupby(level='gene', as_index=False).apply(lambda x: x.iloc[:data_len])
            index_names = data.index.names
            if len(index_names) > 2: # pandas adds dummy index
                droplevel = index_names.index(None)
                data = data.droplevel(droplevel)

        # Bioactivity data.
        logger.info('Setting properties and smiles...')
        self.properties = data[['pXC50']]
        self.smiles = data[['canonical_smiles']]

        # Fingerprint data.
        fp = calculate_morgan_fingerprint(data['canonical_smiles'].values, 2, 512)
        fp = pd.DataFrame(fp, index=data.index)

        # Prune molecules which failed to convert
        valid_molecule_mask = (fp.sum(axis=1) > 0)
        num_invalid_molecules = int(valid_molecule_mask.shape[0] - valid_molecule_mask.sum())
        if num_invalid_molecules > 0:
            logger.warn(f'WARNING: fingerprint dataset length does not match that of SMILES dataset. Run `remove_invalids_by_index` on SMILES dataset to ensure they match.')
            fp = fp[valid_molecule_mask]

        if not isinstance(fp, pd.DataFrame):
            fp = fp.to_pandas()

        self.fingerprints = fp
        self._remove_invalids_by_index()

        assert len(self.smiles) == len(self.fingerprints)
        assert self.smiles.index.equals(self.fingerprints.index)
