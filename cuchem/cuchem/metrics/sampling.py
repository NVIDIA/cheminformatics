#!/usr/bin/env python3

import logging
import pickle

import cupy
import numpy as np
import pandas as pd
from rdkit import Chem

from cuml.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid, KFold
from cuml.metrics.regression import mean_squared_error
from cuchem.utils.metrics import spearmanr
from cuchem.utils.distance import tanimoto_calculate
from cuchem.benchmark.data import ZINC15TestSamplingData, ZINC15TrainData


logger = logging.getLogger(__name__)

__all__ = ['Validity', 'Unique', 'Novelty']


class BaseSampleMetric():
    name = None

    """Base class for metrics based on sampling for a single SMILES string"""
    def __init__(self, inferrer):
        self.inferrer = inferrer
        self.benchmark_data = ZINC15TestSamplingData()

    def _find_similars_smiles(self,
                              smiles,
                              num_samples,
                              scaled_radius,
                              force_unique,
                              sanitize):
        # Check db for results from a previous run
        generated_smiles = self.benchmark_data.fetch_sampling_data(self.inferrer.__class__.__name__,
                                                                   smiles,
                                                                   num_samples,
                                                                   scaled_radius,
                                                                   force_unique,
                                                                   sanitize)
        if not generated_smiles:
            # Generate new samples and update the database
            result = self.inferrer.find_similars_smiles(smiles,
                                                        num_samples,
                                                        scaled_radius=scaled_radius,
                                                        force_unique=force_unique,
                                                        sanitize=sanitize)
            # Result from sampler includes the input SMILES. Removing it.
            # result = result[result.Generated == True]
            generated_smiles = result['SMILES'].to_list()

            embeddings = result['embeddings'].to_list()
            embeddings_dim = result['embeddings_dim'].to_list()

            # insert generated smiles into a database for use later.
            self.benchmark_data.insert_sampling_data(self.inferrer.__class__.__name__,
                                                     smiles,
                                                     num_samples,
                                                     scaled_radius,
                                                     force_unique,
                                                     sanitize,
                                                     generated_smiles,
                                                     embeddings,
                                                     embeddings_dim)
        return generated_smiles


    def _calculate_metric(self, metric_array, num_samples):
        total_samples = len(metric_array) * num_samples
        return np.nansum(metric_array) / float(total_samples)

    def variations(self, cfg, model_dict=None):
        return NotImplemented

    def sample(self):
        return NotImplemented

    def sample_many(self, smiles_dataset, num_samples, radius):
        metric_result = list()

        for index in range(len(smiles_dataset.data)):
            smiles = smiles_dataset.data.iloc[index]
            logger.debug(f'Sampling around {smiles}...')
            result = self.sample(smiles, num_samples, radius)
            metric_result.append(result)

        return np.array(metric_result)

    def calculate(self, **kwargs):
        smiles_dataset = kwargs['smiles_dataset']
        num_samples = kwargs['num_samples']
        radius = kwargs['radius']

        metric_array = self.sample_many(smiles_dataset, num_samples, radius)
        metric = self._calculate_metric(metric_array, num_samples)

        return pd.Series({'name': self.__class__.name,
                          'value': metric,
                          'radius': radius,
                          'num_samples': num_samples})


class Validity(BaseSampleMetric):
    name = 'validity'

    def __init__(self, inferrer):
        super().__init__(inferrer)

    def variations(self, cfg, model_dict=None):
        return cfg.metric.validity.radius

    def sample(self, smiles, num_samples, radius):
        generated_smiles = self._find_similars_smiles(smiles,
                                                      num_samples,
                                                      scaled_radius=radius,
                                                      force_unique=False,
                                                      sanitize=False)
        valid_ctr = 0
        for new_smiles in generated_smiles[1:]:
            m = Chem.MolFromSmiles(new_smiles)
            if m:
                valid_ctr += 1

        return valid_ctr


class Unique(BaseSampleMetric):
    name = 'uniqueness'

    def __init__(self, inferrer):
        super().__init__(inferrer)

    def variations(self, cfg, model_dict=None):
        return cfg.metric.unique.radius

    def sample(self, smiles, num_samples, radius):
        generated_smiles = self._find_similars_smiles(smiles,
                                                      num_samples,
                                                      scaled_radius=radius,
                                                      force_unique=False,
                                                      sanitize=False)
        # Get the unquie ones
        generated_smiles = set(generated_smiles[1:])
        return len(generated_smiles)


class Novelty(BaseSampleMetric):
    name = 'novelty'

    def __init__(self, inferrer):
        super().__init__(inferrer)
        self.training_data = ZINC15TrainData()

    def variations(self, cfg, model_dict=None):
        return cfg.metric.novelty.radius

    def smiles_in_train(self, smiles):
        in_train = self.training_data.is_known_smiles(smiles)
        return in_train

    def sample(self, smiles, num_samples, radius):
        generated_smiles = self._find_similars_smiles(smiles,
                                                      num_samples,
                                                      scaled_radius=radius,
                                                      force_unique=False,
                                                      sanitize=False)

        result = sum([self.smiles_in_train(x) for x in generated_smiles])
        return result

