import logging
import os

import cupy
import generativesampler_pb2
import numpy as np
import pandas as pd

from rdkit import Chem
from cuml.metrics import pairwise_distances
from sklearn.model_selection import ParameterGrid, KFold
from cuml.metrics.regression import mean_squared_error
from cuchem.utils.metrics import spearmanr
from cuchem.utils.distance import tanimoto_calculate
from cuchem.utils.dataset import ZINC_TRIE_DIR, generate_trie_filename
from functools import lru_cache


logger = logging.getLogger(__name__)


def sanitized_smiles(smiles):
    """Ensure SMILES are valid and sanitized, otherwise fill with NaN."""
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol:
        sanitized_smiles = Chem.MolToSmiles(mol)
        return sanitized_smiles
    else:
        return np.NaN


def get_model_iteration(stub):
    """Get Model iteration"""
    spec = generativesampler_pb2.GenerativeSpec(
        model=generativesampler_pb2.GenerativeModel.MegaMolBART,
        smiles="CCC",  # all are dummy vars for this calculation
        radius=0.0001,
        numRequested=1,
        padding=0)

    result = stub.GetIteration(spec)
    return result.iteration


class BaseSampleMetric():
    """Base class for metrics based on sampling for a single SMILES string"""
    def __init__(self):
        self.name = None

    def sample(self):
        return NotImplemented

    def calculate_metric(self, metric_array, num_samples):
        total_samples = len(metric_array) * num_samples
        return np.nansum(metric_array) / float(total_samples)

    def sample_many(self, smiles_dataset, num_samples, func, radius):
        metric_result = list()
        for index in range(len(smiles_dataset.data)):
            smiles = smiles_dataset.data.iloc[index]
            logger.info(f'SMILES: {smiles}')
            result = self.sample(smiles, num_samples, func, radius)
            metric_result.append(result)
        return np.array(metric_result)

    def calculate(self, smiles, num_samples, func, radius):
        metric_array = self.sample_many(smiles, num_samples, func, radius)
        metric = self.calculate_metric(metric_array, num_samples)
        return pd.Series({'name': self.name, 'value': metric, 'radius': radius, 'num_samples': num_samples})


class BaseEmbeddingMetric():
    """Base class for metrics based on embedding datasets"""
    def __init__(self):
        self.name = None

    def sample(self, smiles, max_len, stub, zero_padded_vals, average_tokens):

        spec = generativesampler_pb2.GenerativeSpec(
                model=generativesampler_pb2.GenerativeModel.MegaMolBART,
                smiles=smiles,
                radius=0.0001,  # dummy var for this calculation
                numRequested=1,  # dummy var for this calculation
                padding=max_len)

        result = stub.SmilesToEmbedding(spec)
        shape = [int(x) for x in result.embedding[:2]]
        assert shape[0] == max_len
        embedding = cupy.array(result.embedding[2:])

        embedding = embedding.reshape(shape)
        if zero_padded_vals:
            embedding[len(smiles):, :] = 0.0

        if average_tokens:
            embedding = embedding[:len(smiles)].mean(axis=0).squeeze()
            assert embedding.shape[0] == shape[-1]
        else:
            embedding = embedding.flatten()
        return embedding

    def calculate_metric(self):
        raise NotImplementedError

    @lru_cache(maxsize=None)
    def sample_many(self, smiles_dataset, stub, zero_padded_vals=True, average_tokens=False):
        # Calculate pairwise distances for embeddings
        embeddings = []
        for smiles in smiles_dataset.data.to_pandas():
            embedding = self.sample(smiles, smiles_dataset.max_len, stub, zero_padded_vals, average_tokens)
            embeddings.append(embedding)

        return cupy.asarray(embeddings)

    def calculate(self):
        raise NotImplementedError


class Validity(BaseSampleMetric):
    def __init__(self):
        self.name = 'validity'

    def sample(self, smiles, num_samples, func, radius):
        spec = generativesampler_pb2.GenerativeSpec(
            model=generativesampler_pb2.GenerativeModel.MegaMolBART,
            smiles=smiles,
            radius=radius,
            numRequested=num_samples)

        result = func(spec)
        result = result.generatedSmiles[1:]

        if isinstance(smiles, list):
            result = result[:-1]
        assert len(result) == num_samples
        result = len(pd.Series([sanitized_smiles(x) for x in result]).dropna())
        return result


class Unique(BaseSampleMetric):
    def __init__(self):
        self.name = 'unique'

    def sample(self, smiles, num_samples, func, radius):
        spec = generativesampler_pb2.GenerativeSpec(
            model=generativesampler_pb2.GenerativeModel.MegaMolBART,
            smiles=smiles,
            radius=radius,
            numRequested=num_samples)

        result = func(spec)
        result = result.generatedSmiles[1:]

        if isinstance(smiles, list):
            result = result[:-1]
        assert len(result) == num_samples
        result = len(pd.Series([sanitized_smiles(x) for x in result]).dropna().unique())
        return result


class Novelty(BaseSampleMetric):
    def __init__(self):
        self.name = 'novelty'

    def smiles_in_train(self, smiles):
        """Determine if smiles was in training dataset"""
        in_train = False

        filename = generate_trie_filename(smiles)
        trie_path = os.path.join(ZINC_TRIE_DIR, 'train', filename)
        if os.path.exists(trie_path):
            with open(trie_path, 'r') as fh:
                smiles_list = fh.readlines()
            smiles_list = [x.strip() for x in smiles_list]
            in_train = smiles in smiles_list
        else:
            logger.warn(f'Trie file {filename} not found.')
            in_train = False

        return in_train

    def sample(self, smiles, num_samples, func, radius):
        spec = generativesampler_pb2.GenerativeSpec(
            model=generativesampler_pb2.GenerativeModel.MegaMolBART,
            smiles=smiles,
            radius=radius,
            numRequested=num_samples)

        result = func(spec)
        result = result.generatedSmiles[1:]

        if isinstance(smiles, list):
            result = result[:-1]
        assert len(result) == num_samples

        result = pd.Series([sanitized_smiles(x) for x in result]).dropna()
        result = sum([self.smiles_in_train(x) for x in result])
        return result


class NearestNeighborCorrelation(BaseEmbeddingMetric):
    """Sperman's Rho for correlation of pairwise Tanimoto distances vs Euclidean distance from embeddings"""

    def __init__(self):
        self.name = 'nearest neighbor correlation'

    def calculate_metric(self, embeddings, fingerprints, top_k=None):
        embeddings_dist = pairwise_distances(embeddings)
        del embeddings

        fingerprints_dist = tanimoto_calculate(fingerprints, calc_distance=True)
        del fingerprints

        corr = spearmanr(fingerprints_dist, embeddings_dist, top_k)
        return corr

    def calculate(self, smiles_dataset, fingerprint_dataset, stub, top_k=None):
        embeddings = self.sample_many(smiles_dataset, stub, zero_padded_vals=True, average_tokens=False)

        # Calculate pairwise distances for fingerprints
        fingerprints = cupy.fromDlpack(fingerprint_dataset.data.to_dlpack())
        fingerprints = cupy.asarray(fingerprints, order='C')

        metric = self.calculate_metric(embeddings, fingerprints, top_k)
        metric = cupy.nanmean(metric)
        top_k = embeddings.shape[0] - 1 if not top_k else top_k
        return pd.Series({'name': self.name, 'value': metric, 'top_k':top_k})


class Modelability(BaseEmbeddingMetric):
    """Ability to model molecular properties from embeddings vs Morgan Fingerprints"""

    def __init__(self):
        self.name = 'modelability'
        self.embeddings = None

    def gpu_gridsearch_cv(self, estimator, param_dict, xdata, ydata, n_splits=5):
        """Perform grid search with cross validation and return score"""
        negative_mean_squared_error = lambda x, y: -1 * mean_squared_error(x, y).item()

        best_score = -1 * np.inf  # want to maximize objective
        for param in ParameterGrid(param_dict):
            estimator.set_params(**param)
            metric_list = []

            # Generate CV folds
            kfold_gen = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            for train_idx, test_idx in kfold_gen.split(xdata, ydata):
                xtrain, xtest, ytrain, ytest = xdata[train_idx], xdata[test_idx], ydata[train_idx], ydata[test_idx]
                estimator.fit(xtrain, ytrain)
                ypred = estimator.predict(xtest)
                score = negative_mean_squared_error(ypred, ytest)
                metric_list.append(score)

            metric = np.array(metric_list).mean()
            best_score = max(metric, best_score)
        return best_score

    def calculate_metric(self, embeddings, fingerprints, properties, estimator, param_dict):
        """Perform grid search for each metric and calculate ratio"""

        metric_array = []
        for col in properties.columns:
            props = properties[col].astype(cupy.float32).to_array()
            embedding_error = self.gpu_gridsearch_cv(estimator, param_dict, embeddings, props)
            fingerprint_error = self.gpu_gridsearch_cv(estimator, param_dict, fingerprints, props)
            metric_array.append(embedding_error / fingerprint_error)
        return cupy.array(metric_array)

    def calculate(self, smiles_dataset, fingerprint_dataset, properties, stub, estimator, param_dict):
        embeddings = self.sample_many(smiles_dataset, stub, zero_padded_vals=False, average_tokens=True)
        embeddings = cupy.asarray(embeddings, dtype=cupy.float32)

        fingerprints = cupy.fromDlpack(fingerprint_dataset.data.to_dlpack())
        fingerprints = cupy.asarray(fingerprints, order='C', dtype=cupy.float32)

        metric = self.calculate_metric(embeddings, fingerprints, properties, estimator, param_dict)
        metric = cupy.nanmean(metric)
        return pd.Series({'name': self.name, 'value': metric})
