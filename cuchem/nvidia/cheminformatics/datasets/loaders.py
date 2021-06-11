import pathlib
import pandas as pd
import numpy as np
import cudf
import os
import logging

logger = logging.getLogger(__name__)

class GenericBenchmarkDataset():
    def __init__(self):
        self.name = None
        self.index_col = None
        self.index = None
        self.max_len = None
        self.data_path = None
        self.data = None

    def _load_csv(self, columns, length_columns=None):
        columns = [columns] if not isinstance(columns, list) else columns
        data = cudf.read_csv(self.data_path).drop_duplicates(subset=columns)

        if self.index_col:
            data = data.set_index(self.index_col).sort_index()

        if self.index is not None:
            data = data.loc[self.index]
        elif self.max_len:
            length_columns = [length_columns] if not isinstance(length_columns, list) else length_columns
            assert len(length_columns) == len(columns)
            mask = data[length_columns].max(axis=1) <= self.max_len
            data = data[mask]
  
        out_col = ['smiles1'] if len(columns) == 1 else ['smiles1', 'smiles2']
        renamer = dict(zip(columns, out_col))
        data.rename(columns=renamer, inplace=True)

        if len(out_col) == 1:
            data = data[out_col[0]] # Series
        else:
            data = data[out_col] # DataFrame
        self.data = data

        return

    def load(self):
        raise NotImplemented


class ChEMBL_Approved_Drugs(GenericBenchmarkDataset):
    def __init__(self, index_col='molregno', max_len=None, index=None):
        self.name = 'ChEMBL Approved Drugs (Phase III/IV)'

        assert (max_len is None) | (index is None)
        self.index_col = index_col
        self.max_len = max_len
        self.index = index
        self.length = None

        self.data_path = '/workspace/cuchem/tests/data/benchmark_approved_drugs.csv'
        assert os.path.exists(self.data_path)
        
    def load(self, columns=['canonical_smiles'], length_columns=['length']):
        self._load_csv(columns, length_columns)


class ChEMBL_20K_Samples(GenericBenchmarkDataset):

    def __init__(self, index_col='molregno', max_len=None, index=None):
        self.name = 'ChEMBL 20K Samples'

        assert (max_len is None) | (index is None)
        self.index_col = index_col
        self.max_len = max_len
        self.index = index
        self.data = None
        self.length = None

        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), 
                                      'data', 
                                      'benchmark_ChEMBL_random_sampled_drugs.csv')
        assert os.path.exists(self.data_path)

    def load(self, columns=['canonical_smiles'], length_columns=['length']):
        self._load_csv(columns, length_columns)


class ChEMBL_20K_Fingerprints(GenericBenchmarkDataset):
    def __init__(self, index_col='molregno'):
        self.name = 'ChEMBL 20K Fingerprints'

        self.index_col = index_col
        self.data = None
        self.data_path = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), 
                                      'data', 
                                      'fingerprints_ChEMBL_random_sampled_drugs.csv')
        assert os.path.exists(self.data_path)

    def load(self, index=None):
        data = cudf.read_csv(self.data_path)
        if self.index_col:
            data = data.set_index(self.index_col).sort_index()

        if index:
            data = data.loc[index]
        self.data = data
        return
