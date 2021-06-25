import os
import generativesampler_pb2
from rdkit import Chem
import numpy as np
import pandas as pd
import cupy
from cuml.metrics import pairwise_distances
from cuml import LinearRegression, ElasticNet
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from cuml.metrics.regression import mean_squared_error
from nvidia.cheminformatics.utils.metrics import spearmanr
from nvidia.cheminformatics.utils.distance import tanimoto_calculate
from nvidia.cheminformatics.utils.dataset import ZINC_TRIE_DIR, generate_trie_filename
import logging

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
        smiles="CCC", # all are dummy vars for this calculation
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
        raise NotImplemented

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

    def sample(self, smiles, max_len, stub):

        spec = generativesampler_pb2.GenerativeSpec(
                model=generativesampler_pb2.GenerativeModel.MegaMolBART,
                smiles=smiles,
                radius=0.0001, # dummy var for this calculation
                numRequested=1, # dummy var for this calculation
                padding=max_len)

        result = stub.SmilesToEmbedding(spec)
        # Shape is the first two values -- not needed           
        embedding = cupy.array(result.embedding[2:])

        # Zero out padded values
        embedding = embedding.reshape(max_len, -1)
        embedding[len(smiles):, :] = 0.0
        
        embedding = embedding.flatten()
        return embedding

    def calculate_metric(self):
        raise NotImplemented

    def sample_many(self, smiles_dataset, stub):
        # Calculate pairwise distances for embeddings
        embeddings = []
        for smiles in smiles_dataset.data.to_pandas():
            embedding = self.sample(smiles, smiles_dataset.max_len, stub)
            embeddings.append(embedding)
        
        return cupy.asarray(embeddings)

    def calculate(self):
        raise NotImplemented


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

    def calculate_metric(self, embeddings, fingerprint_dataset, top_k=None):
        embeddings_dist = pairwise_distances(embeddings)
        del embeddings

        # Calculate pairwise distances for fingerprints
        fingerprints = cupy.fromDlpack(fingerprint_dataset.data.to_dlpack())
        fingerprints = cupy.asarray(fingerprints, order='C')
        fingerprints_dist = tanimoto_calculate(fingerprints, calc_distance=True)
        del fingerprints, fingerprint_dataset

        corr = spearmanr(fingerprints_dist, embeddings_dist, top_k)
        return corr

    def calculate(self, smiles_dataset, fingerprint_dataset, stub, top_k=None):
        embeddings = self.sample_many(smiles_dataset, stub)
        metric = self.calculate_metric(embeddings, fingerprint_dataset, top_k)
        metric = cupy.nanmean(metric)
        return pd.Series({'name': self.name, 'value': metric, 'top_k':top_k})


class Modelability(BaseEmbeddingMetric):
    """Ability to model molecular properties from embeddings vs Morgan Fingerprints"""
    
    def __init__(self):
        self.name = 'modelability'

    def calculate_metric(self, embeddings, params, fingerprint_dataset, top_k=None):
        # ML models

        from IPython import embed
        embed()

        for param in params.columns:
            rf_model = RandomForestRegressor(n_estimators=10, 
                                             random_state=0)
            rf_model.fit(embeddings[:, :100], params[param]) # DEBUG -- memory issue
            

        # Calculate pairwise distances for fingerprints
        fingerprints = cupy.fromDlpack(fingerprint_dataset.data.to_dlpack())
        fingerprints = cupy.asarray(fingerprints, order='C')
        # CUML fitting
        del fingerprints, fingerprint_dataset

    def calculate(self, smiles_dataset, fingerprint_dataset, stub):
        embeddings = self.sample_many(smiles_dataset, stub)
        metric = self.calculate_metric(embeddings, smiles_dataset.properties, fingerprint_dataset)

        # metric = cupy.nanmean(metric)
        # return pd.Series({'name': self.name, 'value': metric})


from cuml import LinearRegression, ElasticNet
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from cuml.metrics.regression import mean_squared_error
import cupy
import cudf

embeddings = cupy.fromfile('embeddings.pkl').reshape(10,-1).astype(cupy.float32)
params = cudf.read_hdf('params.hdf', 'params')

n_estimator_list = [10, 50, 10000]
rf_model = RandomForestRegressor(n_estimators=n_estimator_list[0], 
                                    random_state=0)

# for col in params.columns:
col = 'mw'
best_mse = np.inf
params[col] = params[col].astype(cupy.float32)
for n_estimator in n_estimator_list:
    rf_model.n_estimators = n_estimator # warm start
    rf_model.fit(embeddings[:, :100], params[col]) # DEBUG -- memory issue
    predict = rf_model.predict(embeddings[:, :100])
    error = mean_squared_error(params[col], predict)
    best_mse = min(error.item(), best_mse)