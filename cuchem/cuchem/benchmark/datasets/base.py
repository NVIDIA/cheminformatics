import logging
import cudf

logger = logging.getLogger(__name__)


class GenericCSVDataset():
    def __init__(self,
                 name=None,
                 properties_cols=None,
                 index_col=None,
                 index_selection=None,
                 data_path=None,
                 max_seq_len=None):
        self.name = name
        self.data_path = data_path
        self.data = None
        self.max_seq_len = max_seq_len
        self.properties = None

        self.properties_cols = properties_cols # TODO most of these should be passed during load
        self.index_col = index_col
        self.index_selection = index_selection

    def _load_csv(self, columns, length_column=None, return_remaining=True):
        columns = [columns] if not isinstance(columns, list) else columns
        data = cudf.read_csv(self.data_path)

        if self.index_col:
            data = data.set_index(self.index_col).sort_index()
        else:
            data.index.name = 'index'

        if self.index_selection:
            data = data.loc[self.index_selection]

        if self.max_seq_len:
            if length_column:
                mask = data[length_column] <= self.max_seq_len
            elif len(columns) == 1:
                mask = data[columns[0]].str.len() <= self.max_seq_len
            data = data[mask]
        else:
            if length_column:
                self.max_seq_len = data[length_column].max()
            elif len(columns) == 1:
                self.max_seq_len = data[columns[0]].str.len().max()

        cleaned_data = data[columns]

        if return_remaining:
            if length_column:
                remain_columns = [x for x in data.columns if (x not in columns) & (x not in [length_column])]
            else:
                remain_columns = [x for x in data.columns if (x not in columns)]
            other_data = data[remain_columns]
        else:
            other_data = None
        return cleaned_data, other_data

    def load(self, columns=['canonical_smiles'], length_column='length', data_len=None):
        self.data, _ = self._load_csv(columns, length_column)
        if data_len:
            self.data = self.data.iloc[:data_len]


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
