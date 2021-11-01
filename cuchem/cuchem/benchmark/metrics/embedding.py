#!/usr/bin/env python3

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, KFold

try:
    import cuml # TODO is there a better way to check for RAPIDS?
except:
    import numpy as xpy
    from sklearn.metrics import pairwise_distances, mean_squared_error
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from cuchem.utils.metrics import spearmanr # Replace this with CPU version: https://github.com/NVIDIA/cheminformatics/blob/daf2989fdcdc9ef349605484d3b96586846396dc/cuchem/tests/test_metrics.py#L235
    from cuchem.utils.distance import tanimoto_calculate # Replace this with similar to CPU version: rdkit.DataManip.Metric.rdMetricMatrixCalc.GetTanimotoDistMat
else:
    import cupy as xpy
    from cuml.metrics import pairwise_distances, mean_squared_error
    from cuml.linear_model import LinearRegression, ElasticNet
    from cuml.svm import SVR
    from cuml.ensemble import RandomForestRegressor
    from cuchem.utils.metrics import spearmanr
    from cuchem.utils.distance import tanimoto_calculate


logger = logging.getLogger(__name__)

__all__ = ['NearestNeighborCorrelation', 'Modelability']


def get_model_dict():
        lr_estimator = LinearRegression(normalize=True)
        lr_param_dict = {'normalize': [True]}

        en_estimator = ElasticNet(normalize=True)
        en_param_dict = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
                         'l1_ratio': [0.1, 0.5, 1.0, 10.0]}

        sv_estimator = SVR(kernel='rbf')
        sv_param_dict = {'C': [0.01, 0.1, 1.0, 10], 'degree': [3,5,7,9]}

        rf_estimator = RandomForestRegressor(accuracy_metric='mse', random_state=0)
        rf_param_dict = {'n_estimators': [10, 50]}

        return {'linear_regression': [lr_estimator, lr_param_dict],
                'elastic_net': [en_estimator, en_param_dict],
                'support_vector_machine': [sv_estimator, sv_param_dict],
                'random_forest': [rf_estimator, rf_param_dict]}

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
        self.name = self.__class__.__name__

    def variations(self):
        return NotImplemented

    def _find_embedding(self,
                        smiles,
                        max_seq_len):

        # Check db for results from a previous run
        embedding_results = self.sample_cache.fetch_embedding_data(smiles)
        if not embedding_results:
            # Generate new samples and update the database
            embedding_results = self.inferrer.smiles_to_embedding(smiles,
                                                                  max_seq_len)
            embedding = embedding_results.embedding
            embedding_dim = embedding_results.dim

            self.sample_cache.insert_embedding_data(smiles,
                                                    embedding_results.embedding,
                                                    embedding_results.dim)
        else:
            # Convert result to correct format
            embedding, embedding_dim = embedding_results

        return embedding, embedding_dim

    def encode(self, smiles, zero_padded_vals, average_tokens, max_seq_len=None):
        """Encode a single SMILES to embedding from model"""
        embedding, dim = self._find_embedding(smiles, max_seq_len)

        embedding = xpy.array(embedding).reshape(dim).squeeze()
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

    def encode_many(self, max_seq_len=None, zero_padded_vals=True, average_tokens=False):
        # Calculate pairwise distances for embeddings

        if not max_seq_len:
            max_seq_len = self.smiles_dataset.max_seq_len

        embeddings = []
        for smiles in self.smiles_dataset.data['canonical_smiles']:
            # smiles = self.smiles_dataset.data.loc[smiles_index]
            embedding = self.encode(smiles, zero_padded_vals, average_tokens, max_seq_len=max_seq_len)
            embeddings.append(xpy.array(embedding))

        return xpy.asarray(embeddings)

    def calculate(self):
        raise NotImplementedError


class NearestNeighborCorrelation(BaseEmbeddingMetric):
    """Sperman's Rho for correlation of pairwise Tanimoto distances vs Euclidean distance from embeddings"""
    name = 'nearest neighbor correlation'

    def __init__(self, inferrer, sample_cache, smiles_dataset, fingerprint_dataset):
        super().__init__(inferrer, sample_cache, smiles_dataset, fingerprint_dataset)
        self.name = NearestNeighborCorrelation.name

    def variations(self, cfg, **kwargs):
        top_k_list = list(cfg.metric.nearest_neighbor_correlation.top_k)
        top_k_list = [int(x) for x in top_k_list]
        return {'top_k': top_k_list}

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
        fingerprints = xpy.asarray(self.fingerprint_dataset.data)

        metric = self._calculate_metric(embeddings, fingerprints, top_k)
        metric = xpy.nanmean(metric)
        top_k = embeddings.shape[0] - 1 if not top_k else top_k
        return pd.Series({'name': self.name, 'value': metric, 'top_k':top_k})


class Modelability(BaseEmbeddingMetric):
    """Ability to model molecular properties from embeddings vs Morgan Fingerprints"""
    name = 'modelability'

    def __init__(self, name, inferrer, sample_cache, smiles_dataset, fingerprint_dataset):
        super().__init__(inferrer, sample_cache, smiles_dataset, fingerprint_dataset)
        self.name = name
        self.model_dict = get_model_dict()

    def variations(self, model_dict=None, **kwargs):
        if model_dict:
            self.model_dict = model_dict
        return {'model': list(self.model_dict.keys())}

    def gpu_gridsearch_cv(self, estimator, param_dict, xdata, ydata, n_splits=5):
        """Perform grid search with cross validation and return score"""
        logger.info(f"Validating input shape {xdata.shape[0]} == {ydata.shape[0]}")
        assert xdata.shape[0] == ydata.shape[0]

        best_score, best_param = np.inf, None
        for param in ParameterGrid(param_dict):
            estimator.set_params(**param)
            
            # Generate CV folds
            kfold_gen = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            kfold_mse = []
            for train_idx, test_idx in kfold_gen.split(xdata, ydata):
                xtrain, xtest, ytrain, ytest = xdata[train_idx], xdata[test_idx], ydata[train_idx], ydata[test_idx]
                estimator.fit(xtrain, ytrain)
                ypred = estimator.predict(xtest)
                mse = mean_squared_error(ypred, ytest).item() # NOTE: convert to negative MSE and maximize metric if SKLearn GridSearch is ever used
                kfold_mse.append(mse)

            avg_mse = np.nanmean(np.array(kfold_mse))
            if avg_mse < best_score:
                best_score, best_param = avg_mse, param
        return best_score, best_param

    def _calculate_metric(self, embeddings, fingerprints, estimator, param_dict):
        """Perform grid search for each metric and calculate ratio"""
        properties = self.smiles_dataset.properties
        assert len(properties.columns) == 1
        prop_name = properties.columns[0]
        properties = xpy.asarray(properties[prop_name], dtype=xpy.float32)

        embedding_error, embedding_param = self.gpu_gridsearch_cv(estimator, param_dict, embeddings, properties)
        fingerprint_error, fingerprint_param = self.gpu_gridsearch_cv(estimator, param_dict, fingerprints, properties)
        ratio = fingerprint_error / embedding_error # If ratio > 1.0 --> embedding error is smaller --> embedding model is better
        return ratio, fingerprint_error, embedding_error, fingerprint_param, embedding_param

    def calculate(self, estimator, param_dict, **kwargs):
        embeddings = self.encode_many(zero_padded_vals=False, average_tokens=True)
        embeddings = xpy.asarray(embeddings, dtype=xpy.float32)
        fingerprints = xpy.asarray(self.fingerprint_dataset.data.values, dtype=xpy.float32)

        results = self._calculate_metric(embeddings,
                                         fingerprints,
                                         estimator,
                                         param_dict)
        ratio, fingerprint_error, embedding_error, fingerprint_param, embedding_param = results
        property_name = self.smiles_dataset.properties.columns[0]
        return pd.Series({'name': self.name, 'value': ratio, 'property': property_name,
                          'fingerprint_error': fingerprint_error, 'embedding_error': embedding_error,
                          'fingerprint_param': fingerprint_param, 'embedding_param': embedding_param})