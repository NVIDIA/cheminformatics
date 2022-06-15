import os
import pathlib
import logging
import pandas as pd
import numpy as np
import tempfile
from cuchembench.utils.smiles import calc_morgan_fingerprints

logger = logging.getLogger(__name__)


class GenericCSVDataset():
    def __init__(self,
                 name=None,
                 properties_cols=None,
                 index_col=None,
                 index_selection=None,
                 max_seq_len=None,
                 data_filename=None,
                 fp_filename=None):
        self.name = name

        # Files
        self.data_filename = data_filename
        self.fp_filename = fp_filename

        self.prop_data_path = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(), 'csv_data', data_filename)
        assert os.path.exists(self.prop_data_path)

        # prop_data = pd.DataFrame(columns=properties_cols)
        # prop_data.to_csv(self.prop_data_path, index=False)
        # #TODO cols must be passed in
        fp_data_path = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(), 'csv_data', fp_filename)
        # if not os.path.exists(fp_data_path):
        #     logger.info(f'Fingerprint path {fp_data_path} does not exist, generating temporary fingerprints file.')
        #     temp_dir = '/tmp' #tempfile.TemporaryDirectory().name
        #     fp_data_path = os.path.join(temp_dir, self.fp_filename)

        self.fp_data_path = fp_data_path

        # Metadata - Physchem: Many of these are set in the base classes
        self.max_seq_len = max_seq_len
        self.index_col = index_col
        self.index_selection = index_selection
        self.properties_cols = properties_cols # TODO most of these should be passed during load
        self.orig_property_name = None
        self.smiles_col = None

        # Data
        self.smiles = None
        self.properties = None
        self.fingerprints = None

    def _generate_fingerprints(self, data, columns, nbits = 512):
        fp = calc_morgan_fingerprints(data, smiles_col=columns[0], nbits = nbits)
        # fp = fp.to_pandas()
        fp.columns = fp.columns.astype(np.int64) # TODO may not be needed since formatting fixed
        for col in fp.columns: # TODO why are these floats
            fp[col] = fp[col].astype(np.float32)
        fp.index = data.index.astype(np.int64)

        assert len(data) == len(fp)
        assert data.index.equals(fp.index)
        fp = fp.reset_index()
        fp.to_csv(self.fp_data_path, index=False)
        assert os.path.exists(self.fp_data_path), AssertionError(f'Failed to create temporary fingerprint file {self.fp_data_path}')

    @staticmethod
    def _truncate_data(data, data_len):
        return data.iloc[:data_len]

    def _load_csv(self,
                  columns,
                  length_column=None,
                  return_remaining=True,
                  data_len=None,
                  nbits = 512):
        columns = [columns] if not isinstance(columns, list) else columns
        data = pd.read_csv(self.prop_data_path)

        if self.index_col:
            data = data.set_index(self.index_col).sort_index()
        else:
            data.index.name = 'index'

        if self.index_selection:
            data = data.loc[self.index_selection]

        if self.max_seq_len:
            mask = data[columns[0]].str.len() <= self.max_seq_len
            data = data[mask]
        else:
            self.max_seq_len = data[columns[0]].str.len().max()

        if data_len:
            data = self._truncate_data(data, data_len)

        generate_fingerprints_file = True
        if os.path.exists(self.fp_data_path):
            logger.info(f'Fingerprints file {self.fp_data_path} exists. Checking if indexes match data.')

            # Check index
            fp_subset = pd.read_csv(self.fp_data_path, usecols=[self.index_col])
            if data.index.isin(fp_subset[self.index_col]).all():
                generate_fingerprints_file = False
                logger.info(f'Indexes in data are all contained in fingerprints file {self.fp_data_path}. Using existing file.')
            else:
                logger.info(f'Indexes in data are not all contained in fingerprints file {self.fp_data_path} Regenerating.')

        if generate_fingerprints_file:
            # Generate here so that column names are consistent with inputs
            logger.info(f'Creating temporary fingerprints file {self.fp_data_path} for {data.shape} input size.')
            # TODO
            self._generate_fingerprints(data, columns, nbits)

        cleaned_data = data[columns]
        if return_remaining:
            remain_columns = [x for x in data.columns if (x not in columns)]
            other_data = data[remain_columns]
        else:
            other_data = None
        return cleaned_data, other_data

    #TODO: rename 'columns' argument to match its usage.
    def load(self,
             columns=['canonical_smiles'],
             length_column='length',
             data_len=None,
             nbits = 512):

        # Load physchem properties
        logger.info(f'Loading data from {self.prop_data_path}')
        self.smiles, self.properties = self._load_csv(columns, length_column, data_len=data_len, nbits = nbits)

        if self.smiles_col is not None:
            self.smiles = self.smiles.rename(columns={self.smiles_col: 'canonical_smiles'})

        if self.orig_property_name:
            self.properties = self.properties.rename(columns=dict(zip(self.orig_property_name,
                                                                      self.properties_cols)))

        if self.properties_cols:
            self.properties = self.properties[self.properties_cols]

        # Load fingerprint properties
        logger.info(f'Loading fingerprints from {self.fp_data_path}')
        self.fingerprints = pd.read_csv(self.fp_data_path)

        # Set column names and check for correctness
        if self.index_col:
            self.fingerprints = self.fingerprints.set_index(self.index_col).sort_index()

        assert len(self.fingerprints.columns) == nbits, AssertionError(f'Fingerprint dataframe appears to contain incorrect number of column(s)')
        try:
            self.fingerprints.columns.astype(int)
        except:
            raise ValueError(f'Fingerprint dataframe appears to contain incorrect (non integer) column name(s)')

        # Slice fingerprints if needed and ensure data set indexes are identical
        if self.smiles is not None:
            self.fingerprints = self.fingerprints.loc[self.smiles.index]

        assert len(self.fingerprints) == len(self.smiles) == len(self.properties), AssertionError('Dataframes for SMILES, properties, and fingerprints are not identical length.')
        assert self.smiles.index.equals(self.fingerprints.index) & self.smiles.index.equals(self.properties.index), AssertionError(f'Dataframe indexes are not equivalent')

