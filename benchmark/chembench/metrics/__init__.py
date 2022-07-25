import os
import numpy as np
import cupy as cp

from pydoc import locate


class BaseMetric():
    name = None

    """Base class for metrics based on sampling for a single SMILES string"""
    def __init__(self, metric_name, metric_spec, cfg):
        self.name = metric_name
        self.metric_spec = metric_spec
        self.cfg = cfg
        self.dataset_spec = metric_spec.datasets[0]
        self.label = self.dataset_spec['label']
        self.data_file = self.dataset_spec['file']
        self.total_molecules = 0

    def __len__(self):
        return self.total_molecules

    def _calculate_metric(self, metric_array, num_array):
        return np.nanmean(metric_array / num_array)

    def is_prediction(self):
        return False

    def variations(self):
        radius_list = list(self.cfg.sampling.radius)
        sample_size = self.cfg.sampling.sample_size
        return [{'radius': float(x), 'num_samples': sample_size} for x in radius_list]

    def compute_metrics(self, num_samples, radius):
        return NotImplemented

    def calculate(self, **kwargs):
        radius = kwargs['radius']
        num_samples = kwargs['num_samples']

        metric_array, num_array = self.compute_metrics(num_samples, radius)
        metric = self._calculate_metric(metric_array, num_array)

        return {'name': self.name,
                'value': metric,
                'radius': radius,
                'num_samples': num_samples}

    def cleanup(self):
        pass


class BaseEmbeddingMetric(BaseMetric):
    name = None

    """Base class for metrics based on embedding datasets"""
    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)

        fp_filename = f'fp_{os.path.splitext(os.path.basename(self.data_file))[0]}_{metric_spec["nbits"]}.csv'

        self.dataset = locate(self.dataset_spec.impl)(data_filename=self.data_file,
                                                      fp_filename=fp_filename,
                                                      max_seq_len=self.cfg.sampling.max_seq_len)
        self.dataset.index_col = self.dataset_spec['index_col']
        self.dataset.smiles_col = self.dataset_spec['smis_col']
        self.dataset.properties_cols = list(self.dataset_spec['properties_cols'])
        self.dataset.orig_property_name = self.dataset_spec.get('orig_property_name')

        if 'load_cols' in self.dataset_spec:
            self.dataset.load_cols = list(self.dataset_spec['load_cols'])
        else:
            self.dataset.load_cols = self.dataset.smiles_col

        self.dataset.load(columns=self.dataset.load_cols,
                          data_len=self.dataset_spec.input_size,
                          nbits=metric_spec['nbits'])

        self.smiles_dataset = self.dataset.smiles
        self.fingerprint_dataset = self.dataset.fingerprints
        self.smiles_properties = self.dataset.properties

        self.conn = sqlite3.connect(self.cfg.sampling.db,
                                    uri=True,
                                    check_same_thread=False)

    def __len__(self):
        return len(self.dataset.smiles)

    def is_prediction(self):
        return True

    def _find_embedding(self, smiles):
        # Check db for results from a previous run
        result = self.conn.execute('''
                SELECT ss.embedding, ss.embedding_dim
                FROM smiles s, smiles_samples ss
                WHERE s.id = ss.input_id
                    AND ss.is_generated = 0
                    AND s.processed = 1
                    AND s.smiles = ?
                LIMIT 1
                ''',
                [smiles]).fetchone()

        if not result:
            raise Exception(f'No record for smiles {smiles} found')

        embedding = pickle.loads(result[0])
        embedding_dim = pickle.loads(result[1])
        return (embedding, embedding_dim)

    def encode(self, smiles, zero_padded_vals, average_tokens, max_seq_len=None):
        """Encode a single SMILES to embedding from model"""
        embedding, dim = self._find_embedding(smiles)
        embedding = cp.array(embedding).reshape(dim).squeeze()
        n_dim = embedding.ndim

        if zero_padded_vals:
            if n_dim == 2:
                embedding[len(smiles):, :] = 0.0
            else:
                embedding[len(smiles):] = 0.0

        if n_dim == 2:
            if average_tokens:
                embedding = embedding[:len(smiles)].mean(axis=0).squeeze()
            else:
                embedding = embedding.flatten()

        return embedding

    def _calculate_metric(self):
        raise NotImplementedError

    def encode_many(self, smis, max_seq_len=None, zero_padded_vals=True, average_tokens=False):
        # Calculate pairwise distances for embeddings

        if not max_seq_len:
            max_seq_len = self.cfg.sampling.max_seq_len

        embeddings = []
        for smiles in smis:
            embedding = self.encode(smiles, zero_padded_vals, average_tokens, max_seq_len=max_seq_len)
            embeddings.append(embedding)

        return embeddings

    def calculate(self):
        raise NotImplementedError


from .embedding import *
from .sampling import *
