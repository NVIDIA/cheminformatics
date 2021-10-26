#!/usr/bin/env python3

import logging

import cupy
import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid, KFold
from cuml.metrics import pairwise_distances
from cuml.metrics.regression import mean_squared_error
from cuchem.utils.metrics import spearmanr
from cuchem.utils.distance import tanimoto_calculate


logger = logging.getLogger(__name__)

__all__ = ['NearestNeighborCorrelation', 'Modelability']


class BaseEmbeddingMetric():
    name = None

    """Base class for metrics based on embedding datasets"""
    def __init__(self,
                 inferrer,
                 sample_cache,
                 smiles_dataset,
                 fingerprint_dataset):
        self.inferrer = inferrer
        self.sample_cache = sample_cache
        self.smiles_dataset = smiles_dataset
        self.fingerprint_dataset = fingerprint_dataset

    def variations(self):
        return NotImplemented

    def _find_embedding(self,
                        smiles,
                        max_len):

        # Check db for results from a previous run
        embedding_results = self.sample_cache.fetch_embedding_data(smiles)
        if not embedding_results:
            # Generate new samples and update the database
            embedding_results = self.inferrer.smiles_to_embedding(smiles,
                                                                  max_len + 2)
            embedding = embedding_results.embedding
            embedding_dim = embedding_results.dim

            self.sample_cache.insert_embedding_data(smiles,
                                                    embedding_results.embedding,
                                                    embedding_results.dim)
        else:
            # Convert result to correct format
            embedding, embedding_dim = embedding_results

        return embedding, embedding_dim

    def encode(self, smiles, max_len, zero_padded_vals, average_tokens):
        """Encode a single SMILES to embedding from model"""
        embedding, dim = self._find_embedding(smiles, max_len)

        embedding = cupy.array(embedding).reshape(dim).squeeze()
        assert embedding.ndim == 2, "Metric calculation code currently only works with 2D data (embeddings, not batched)"

        if zero_padded_vals:
            embedding[len(smiles):, :] = 0.0

        if average_tokens:
            embedding = embedding[:len(smiles)].mean(axis=0).squeeze()
            assert (embedding.ndim == 1) & (embedding.shape[0] == dim[-1])
        else:
            embedding = embedding.flatten() # TODO research alternatives to handle embedding sizes in second dim

        return embedding

    def _calculate_metric(self):
        raise NotImplementedError

    def encode_many(self, max_len=None, zero_padded_vals=True, average_tokens=False):
        # Calculate pairwise distances for embeddings

        if not max_len:
            max_len = self.smiles_dataset.max_len

        embeddings = []
        for smiles in self.smiles_dataset.data['canonical_smiles'].to_arrow().to_pylist():
            # smiles = self.smiles_dataset.data.loc[smiles_index]
            embedding = self.encode(smiles, max_len, zero_padded_vals, average_tokens)
            embeddings.append(cupy.array(embedding))

        return cupy.asarray(embeddings)

    def calculate(self):
        raise NotImplementedError


class NearestNeighborCorrelation(BaseEmbeddingMetric):
    """Sperman's Rho for correlation of pairwise Tanimoto distances vs Euclidean distance from embeddings"""

    name = 'nearest neighbor correlation'

    def __init__(self, inferrer, sample_cache, smiles_dataset, fingerprint_dataset):
        super().__init__(inferrer, sample_cache, smiles_dataset, fingerprint_dataset)

    def variations(self, cfg, model_dict=None):
        return cfg.metric.nearestNeighborCorrelation.top_k

    def _calculate_metric(self, embeddings, fingerprints, top_k=None):
        embeddings_dist = pairwise_distances(embeddings)
        del embeddings

        fingerprints_dist = tanimoto_calculate(fingerprints, calc_distance=True)
        del fingerprints

        corr = spearmanr(fingerprints_dist, embeddings_dist, top_k=top_k)
        return corr

    def calculate(self, top_k=None, **kwargs):

        embeddings = self.encode_many(zero_padded_vals=True,
                                      average_tokens=False)

        # Calculate pairwise distances for fingerprints
        fingerprints = cupy.fromDlpack(self.fingerprint_dataset.data.to_dlpack())
        fingerprints = cupy.asarray(fingerprints, order='C')

        metric = self._calculate_metric(embeddings, fingerprints, top_k)
        metric = cupy.nanmean(metric)
        top_k = embeddings.shape[0] - 1 if not top_k else top_k
        return pd.Series({'name': self.name, 'value': metric, 'top_k':top_k})


class Modelability(BaseEmbeddingMetric):
    """Ability to model molecular properties from embeddings vs Morgan Fingerprints"""
    name = 'modelability'

    def __init__(self, inferrer, sample_cache, smiles_dataset, fingerprint_dataset):
        super().__init__(inferrer, sample_cache, smiles_dataset, fingerprint_dataset)

    def variations(self, cfg, model_dict=None):
        return model_dict.keys()

    def gpu_gridsearch_cv(self, estimator, param_dict, xdata, ydata, n_splits=5):
        """Perform grid search with cross validation and return score"""

        assert xdata.shape[0] == ydata.shape[0]

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
                # NB: convert to negative MSE and maximize metric for SKLearn GridSearch
                score = mean_squared_error(ypred, ytest).item()
                metric_list.append(score)

            metric = np.array(metric_list).mean()
            best_score = min(metric, best_score)
        return best_score

    def _calculate_metric(self, embeddings, fingerprints, estimator, param_dict):
        """Perform grid search for each metric and calculate ratio"""

        metric_array = []
        embedding_errors = []
        fingerprint_errors = []
        properties = self.smiles_dataset.properties

        for col in properties.columns:
            props = properties[col].astype(cupy.float32).to_array()

            embedding_error = self.gpu_gridsearch_cv(estimator, param_dict, embeddings, props)
            fingerprint_error = self.gpu_gridsearch_cv(estimator, param_dict, fingerprints, props)

            ratio = fingerprint_error / embedding_error # If ratio > 1.0 --> embedding error is smaller --> embedding model is better
            metric_array.append(ratio)
            embedding_errors.append(embedding_error)
            fingerprint_errors.append(fingerprint_error)

        return cupy.array(metric_array), cupy.array(fingerprint_errors), cupy.array(embedding_errors)

    def calculate(self, estimator, param_dict, **kwfargs): # TODO FIX PROPERTIES CALL
        embeddings = self.encode_many(zero_padded_vals=False, average_tokens=True)
        embeddings = cupy.asarray(embeddings, dtype=cupy.float32)

        fingerprints = cupy.fromDlpack(self.fingerprint_dataset.data.to_dlpack())
        fingerprints = cupy.asarray(fingerprints, order='C', dtype=cupy.float32)

        metric, fingerprint_errors, embedding_errors = self._calculate_metric(embeddings,
                                                                              fingerprints,
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
