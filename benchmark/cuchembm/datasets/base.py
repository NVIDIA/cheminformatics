import os
import pathlib
import logging
import pandas as pd

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
        self.fp_data_path = os.path.join(
            pathlib.Path(__file__).parent.parent.absolute(), 'csv_data', fp_filename)

        assert os.path.exists(self.prop_data_path)
        assert os.path.exists(self.fp_data_path)

        # Metadata - PhyChem: Many of these are set in the base classes
        self.max_seq_len = max_seq_len
        self.physchem_index_col = index_col
        self.index_selection = index_selection
        self.properties_cols = properties_cols # TODO most of these should be passed during load
        self.orig_property_name = None
        self.smiles_col = None

        # Metadata - Fingerprints
        self.fp_index_col = index_col

        # Data
        self.data = None
        self.smiles = None
        self.properties = None
        self.fingerprints = None

    def _load_csv(self,
                  columns,
                  length_column=None,
                  return_remaining=True,
                  data_len=None):
        columns = [columns] if not isinstance(columns, list) else columns
        data = pd.read_csv(self.prop_data_path)

        if self.physchem_index_col:
            data = data.set_index(self.physchem_index_col).sort_index()
        else:
            data.index.name = 'index'

        if self.index_selection:
            data = data.loc[self.index_selection]

        if self.max_seq_len:
            # if length_column:
            #     mask = data[length_column].str.len() <= self.max_seq_len
            # elif len(columns) == 1:
            #     mask = data[columns[0]].str.len() <= self.max_seq_len
            mask = data[columns[0]].str.len() <= self.max_seq_len
            data = data[mask]
        else:
            # if length_column:
            #     self.max_seq_len = data[length_column].str.len().max()
            # elif len(columns) == 1:
            #     self.max_seq_len = data[columns[0]].str.len().max()
            self.max_seq_len = data[columns[0]].str.len().max()

        cleaned_data = data[columns]
        if data_len:
            cleaned_data = cleaned_data.iloc[:data_len]

        if return_remaining:
            # if length_column:
            #     remain_columns = [x for x in data.columns if (x not in columns) & (x not in [length_column])]
            # else:
            #     remain_columns = [x for x in data.columns if (x not in columns)]
            remain_columns = [x for x in data.columns if (x not in columns)]
            other_data = data[remain_columns]
        else:
            other_data = None
        return cleaned_data, other_data

    #TODO: rename 'columns' argument to match its usage.
    def load(self,
             columns=['canonical_smiles'],
             length_column='length',
             data_len=None):
        # Load phyChem properties
        logger.info(f'Loading phyChem properties from {self.prop_data_path}')
        self.data, self.properties = self._load_csv(columns, length_column, data_len=data_len)

        if self.smiles_col is not None:
            self.smiles = self.data.rename(columns={columns[self.smiles_col]: 'canonical_smiles'})
        else:
            self.smiles = self.data

        if self.orig_property_name:
            self.properties = self.properties.rename(columns=dict(zip(self.orig_property_name,
                                                                      self.properties_cols)))

        if self.properties_cols:
            self.properties = self.properties[self.properties_cols]

        # Load fingerprint properties
        logger.info(f'Loading phyChem properties from {self.fp_data_path}')
        self.fingerprints = pd.read_csv(self.fp_data_path)

        if self.physchem_index_col:
            self.fingerprints = self.fingerprints.set_index(self.physchem_index_col).sort_index()

        if self.smiles is not None:
            self.fingerprints = self.fingerprints.loc[self.smiles.index]

        print(self.smiles.shape)
        print(self.fingerprints.shape)
        assert len(self.fingerprints) == len(self.smiles)
        assert self.smiles.index.equals(self.fingerprints.index)
