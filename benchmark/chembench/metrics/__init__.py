import numpy as np


class BaseMetric():
    name = None

    """Base class for metrics based on sampling for a single SMILES string"""
    def __init__(self, metric_name, metric_spec, cfg):
        self.name = metric_name
        self.metric_spec = metric_spec
        self.cfg = cfg
        self.dataset = metric_spec.datasets[0]
        self.label = self.dataset['label']
        self.data_file = self.dataset['file']
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


from .embedding import *
from .sampling import *