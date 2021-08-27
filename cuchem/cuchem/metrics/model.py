#!/usr/bin/env python3

import logging
import pickle

import cupy
import numpy as np
import pandas as pd

from cuml.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid, KFold
from cuml.metrics.regression import mean_squared_error
from cuchem.utils.metrics import spearmanr
from cuchem.utils.distance import tanimoto_calculate
from cuchem.benchmark.data import BenchmarkData, TrainingData


logger = logging.getLogger(__name__)


class BaseSampleMetric():
    name = None

    """Base class for metrics based on sampling for a single SMILES string"""
    def __init__(self, inferrer):
        self.inferrer = inferrer
        self.benchmark_data = BenchmarkData()
        self.training_data = TrainingData()

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


class BaseEmbeddingMetric():
    name = None

    """Base class for metrics based on embedding datasets"""
    def __init__(self, inferrer):
        self.inferrer = inferrer
        self.benchmark_data = BenchmarkData()

    def variations(self, cfg):
        return NotImplemented

    def _find_embedding(self,
                        smiles,
                        scaled_radius,
                        force_unique,
                        sanitize,
                        max_len):
        num_samples = 1

        # Check db for results from a previous run
        generated_smiles = self.benchmark_data.fetch_n_sampling_data(self.inferrer.__class__.__name__,
                                                                     smiles,
                                                                     num_samples,
                                                                     scaled_radius,
                                                                     force_unique,
                                                                     sanitize)
        if not generated_smiles:
            # Generate new samples and update the database
            generated_smiles = self.inferrer.smiles_to_embedding(smiles,
                                                                 max_len,
                                                                 scaled_radius=scaled_radius,
                                                                 num_samples=num_samples)
        else:
            temp = generated_smiles[0]
            embedding = pickle.loads(temp[1])

            generated_smiles = []
            generated_smiles.append(temp[0])
            generated_smiles.append(embedding)
            generated_smiles.append(pickle.loads(temp[2]))

        return generated_smiles[0], generated_smiles[1], generated_smiles[2]

    def sample(self, smiles, max_len, zero_padded_vals, average_tokens):

        _, embedding, dim = self._find_embedding(smiles, 1, False, True, max_len)

        embedding = cupy.array(embedding)
        embedding = embedding.reshape(dim)

        if zero_padded_vals and len(dim) > 2:
            embedding[len(smiles):, :] = 0.0

        if average_tokens and len(dim) > 2:
            embedding = embedding[:len(smiles)].mean(axis=0).squeeze()
            assert embedding.shape[0] == dim[-1]
        else:
            embedding = embedding.flatten()

        return embedding

    def _calculate_metric(self):
        raise NotImplementedError

    def sample_many(self, smiles_dataset, zero_padded_vals=True, average_tokens=False):
        # Calculate pairwise distances for embeddings
        embeddings = []
        max_len = 0
        for smiles in smiles_dataset.data.to_pandas():
            embedding = self.sample(smiles, smiles_dataset.max_len, zero_padded_vals, average_tokens)
            max_len = max(max_len, embedding.shape[0])
            embeddings.append(cupy.array(embedding))

        if max_len > 0:
            embeddings_resized = []
            for embedding in embeddings:
                n_pad = max_len - embedding.shape[0]
                if n_pad <= 0:
                    embeddings_resized.append(embedding)
                    continue
                embedding = cupy.resize(embedding, max_len)
                embeddings_resized.append(embedding)
            embeddings = embeddings_resized

        return cupy.asarray(embeddings)

    def calculate(self, **kwargs):
        raise NotImplementedError


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
                                                      sanitize=True)
        return len(generated_smiles)


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
                                                      sanitize=True)
        # Get the unquie ones
        generated_smiles = set(generated_smiles)
        return len(generated_smiles)


class Novelty(BaseSampleMetric):
    name = 'novelty'

    def __init__(self, inferrer):
        super().__init__(inferrer)

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
                                                      sanitize=True)

        result = sum([self.smiles_in_train(x) for x in generated_smiles])
        return result


class NearestNeighborCorrelation(BaseEmbeddingMetric):
    """Sperman's Rho for correlation of pairwise Tanimoto distances vs Euclidean distance from embeddings"""

    name = 'nearest neighbor correlation'

    def __init__(self, inferrer):
        super().__init__(inferrer)

    def variations(self, cfg, model_dict=None):
        return cfg.metric.nearestNeighborCorrelation.top_k

    def _calculate_metric(self, embeddings, fingerprints, top_k=None):
        embeddings_dist = pairwise_distances(embeddings)
        del embeddings

        fingerprints_dist = tanimoto_calculate(fingerprints, calc_distance=True)
        del fingerprints

        corr = spearmanr(fingerprints_dist, embeddings_dist, top_k)
        return corr

    def calculate(self, **kwargs):
        smiles_dataset = kwargs['smiles_dataset']
        fingerprint_dataset = kwargs['fingerprint_dataset']
        top_k = kwargs['top_k']

        embeddings = self.sample_many(smiles_dataset,
                                      zero_padded_vals=True,
                                      average_tokens=False)

        # Calculate pairwise distances for fingerprints
        fingerprints = cupy.fromDlpack(fingerprint_dataset.data.to_dlpack())
        fingerprints = cupy.asarray(fingerprints, order='C')

        metric = self._calculate_metric(embeddings, fingerprints, top_k)
        metric = cupy.nanmean(metric)
        top_k = embeddings.shape[0] - 1 if not top_k else top_k
        return pd.Series({'name': self.name, 'value': metric, 'top_k':top_k})


class Modelability(BaseEmbeddingMetric):
    """Ability to model molecular properties from embeddings vs Morgan Fingerprints"""
    name = 'modelability'

    def __init__(self, inferrer):
        super().__init__(inferrer)
        self.embeddings = None

    def variations(self, cfg, model_dict=None):
        return model_dict.keys()

    def gpu_gridsearch_cv(self, estimator, param_dict, xdata, ydata, n_splits=5):
        """Perform grid search with cross validation and return score"""

        best_score = np.inf
        for param in ParameterGrid(param_dict):
            estimator.set_params(**param)
            metric_list = []

            # Generate CV folds
            kfold_gen = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            for train_idx, test_idx in kfold_gen.split(xdata, ydata):
                xtrain, xtest, ytrain, ytest = xdata[train_idx], xdata[test_idx], ydata[train_idx], ydata[test_idx]
                estimator.fit(xtrain, ytrain)
                ypred = estimator.predict(xtest)
                score = mean_squared_error(ypred, ytest).item() # NB: convert to negative MSE and maximize metric for SKLearn GridSearch
                metric_list.append(score)

            metric = np.array(metric_list).mean()
            best_score = min(metric, best_score)
        return best_score

    def _calculate_metric(self, embeddings, fingerprints, properties, estimator, param_dict):
        """Perform grid search for each metric and calculate ratio"""

        metric_array = []
        embedding_errors = []
        fingerprint_errors = []
        for col in properties.columns:
            props = properties[col].astype(cupy.float32).to_array()
            embedding_error = self.gpu_gridsearch_cv(estimator, param_dict, embeddings, props)
            fingerprint_error = self.gpu_gridsearch_cv(estimator, param_dict, fingerprints, props)
            ratio = fingerprint_error / embedding_error # If ratio > 1.0 --> embedding error is smaller --> embedding model is better
            metric_array.append(ratio)
            embedding_errors.append(embedding_error)
            fingerprint_errors.append(fingerprint_error)

        return cupy.array(metric_array), cupy.array(fingerprint_errors), cupy.array(embedding_errors)


    def calculate(self, **kwargs):
        smiles_dataset = kwargs['smiles_dataset']
        fingerprint_dataset = kwargs['fingerprint_dataset']
        properties = kwargs['properties']
        estimator = kwargs['estimator']
        param_dict = kwargs['param_dict']

        embeddings = self.sample_many(smiles_dataset, zero_padded_vals=False, average_tokens=True)
        embeddings = cupy.asarray(embeddings, dtype=cupy.float32)

        fingerprints = cupy.fromDlpack(fingerprint_dataset.data.to_dlpack())
        fingerprints = cupy.asarray(fingerprints, order='C', dtype=cupy.float32)

        metric, fingerprint_errors, embedding_errors  = self._calculate_metric(embeddings,
                                                                               fingerprints,
                                                                               properties,
                                                                               estimator,
                                                                               param_dict)
        logger.info(f'{type(metric)}  {type(fingerprint_errors)} {type(embedding_errors)}')
        metric = cupy.nanmean(metric)
        fingerprint_errors = cupy.nanmean(fingerprint_errors)
        embedding_errors = cupy.nanmean(embedding_errors)

        return pd.Series({'name': self.name,
                          'value': metric,
                          'fingerprint_error': fingerprint_errors,
                          'embedding_error': embedding_errors})
