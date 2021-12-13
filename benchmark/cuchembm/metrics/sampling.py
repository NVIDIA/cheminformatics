#!/usr/bin/env python3

import logging

import numpy as np
import pandas as pd
from rdkit import Chem


logger = logging.getLogger(__name__)

__all__ = ['Validity', 'Unique', 'Novelty']


class BaseSampleMetric():
    name = None

    """Base class for metrics based on sampling for a single SMILES string"""
    def __init__(self,
                 inferrer,
                 sample_cache,
                 smiles_dataset,
                 remove_invalid=False):
        self.inferrer = inferrer
        self.sample_cache = sample_cache
        self.dataset = smiles_dataset
        self.remove_invalid = remove_invalid
        self.name = self.__class__.__name__

    def _find_similars_smiles(self,
                              smiles,
                              num_samples,
                              scaled_radius,
                              force_unique,
                              sanitize):
        # Check db for results from a previous run
        logger.debug(f'Checking cache for {smiles}...')
        generated_smiles = self.sample_cache.fetch_sampling_data(self.inferrer.__class__.__name__,
                                                                   smiles,
                                                                   num_samples,
                                                                   scaled_radius,
                                                                   force_unique,
                                                                   sanitize)
        if not generated_smiles:
            logger.debug(f'Sampling for {smiles}...')
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
            self.sample_cache.insert_sampling_data(self.inferrer.__class__.__name__,
                                                     smiles,
                                                     num_samples,
                                                     scaled_radius,
                                                     force_unique,
                                                     sanitize,
                                                     generated_smiles,
                                                     embeddings,
                                                     embeddings_dim)
        return generated_smiles

    def _calculate_metric(self, metric_array, num_array):
        return np.nanmean(metric_array / num_array)

    def variations(self):
        return NotImplemented

    def sample(self):
        return NotImplemented

    def sample_many(self, smiles_dataset, num_samples, radius):
        metric_result = list()
        num_result = list()

        for index in range(len(smiles_dataset.smiles)):
            smiles = smiles_dataset.smiles.iloc[index]
            logger.debug(f'Sampling around {smiles}...')
            result, num_molecules = self.sample(smiles, num_samples, radius)
            metric_result.append(result)
            num_result.append(num_molecules)

        return np.array(metric_result), np.array(num_result)

    def calculate(self, radius, num_samples, **kwargs):
        metric_array, num_array = self.sample_many(self.dataset, num_samples, radius)
        metric = self._calculate_metric(metric_array, num_array)

        return {'name': self.name,
                'value': metric,
                'radius': radius,
                'num_samples': num_samples}

    @staticmethod
    def get_valid_molecules(smiles_list, canonicalize=False):
        valid_smiles_list = list()
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                if canonicalize:
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    valid_smiles_list.append(canonical_smiles)
                else:
                    valid_smiles_list.append(smiles)
            else:
                logger.debug(f'Molecule {smiles} is invalid')
        return valid_smiles_list


class Validity(BaseSampleMetric):
    name = 'validity'

    def __init__(self, inferrer, sample_cache, smiles_dataset, **kwargs):
        super().__init__(inferrer, sample_cache, smiles_dataset)
        self.name = Validity.name

    def variations(self, cfg, **kwargs):
        radius_list = list(cfg.metric.validity.radius)
        radius_list = [float(x) for x in radius_list]
        return {'radius': radius_list}

    def sample(self, smiles, num_samples, radius):
        generated_smiles = self._find_similars_smiles(smiles,
                                                      num_samples,
                                                      scaled_radius=radius,
                                                      force_unique=False,
                                                      sanitize=False)
        
        num_mol = len(generated_smiles[1:])
        valid_ctr = len(self.get_valid_molecules(generated_smiles[1:]))
        return valid_ctr, num_mol


class Unique(BaseSampleMetric):
    name = 'unique'

    def __init__(self, inferrer, sample_cache, smiles_dataset, remove_invalid=True):
        super().__init__(inferrer, sample_cache, smiles_dataset, remove_invalid)
        self.name = Unique.name

    def variations(self, cfg, **kwargs):
        radius_list = list(cfg.metric.unique.radius)
        radius_list = [float(x) for x in radius_list]
        return {'radius': radius_list}

    def sample(self, smiles, num_samples, radius):
        generated_smiles = self._find_similars_smiles(smiles,
                                                      num_samples,
                                                      scaled_radius=radius,
                                                      force_unique=False,
                                                      sanitize=False)
        generated_smiles = generated_smiles[1:]
        if self.remove_invalid:
            generated_smiles = self.get_valid_molecules(generated_smiles, canonicalize=True)

        # Get the unique ones
        num_mol = len(generated_smiles)
        generated_ctr = len(set(generated_smiles))
        return generated_ctr, num_mol


class Novelty(BaseSampleMetric):
    name = 'novelty'

    def __init__(self, inferrer, sample_cache, smiles_dataset, training_data, remove_invalid=True):
        super().__init__(inferrer, sample_cache, smiles_dataset, remove_invalid)
        self.name = Novelty.name
        self.training_data = training_data

    def variations(self, cfg, **kwargs):
        radius_list = list(cfg.metric.novelty.radius)
        radius_list = [float(x) for x in radius_list]
        return {'radius': radius_list}

    def smiles_not_in_train(self, smiles):
        in_train = self.training_data.is_known_smiles(smiles)
        return not(in_train)

    def sample(self, smiles, num_samples, radius):
        generated_smiles = self._find_similars_smiles(smiles,
                                                      num_samples,
                                                      scaled_radius=radius,
                                                      force_unique=False,
                                                      sanitize=False)

        generated_smiles = generated_smiles[1:]
        if self.remove_invalid:
            generated_smiles = self.get_valid_molecules(generated_smiles, canonicalize=True)

        num_mol = len(generated_smiles)
        novelty_ctr = sum([self.smiles_not_in_train(x) for x in generated_smiles])
        return novelty_ctr, num_mol
