import generativesampler_pb2
from rdkit import Chem
import numpy as np
import pandas as pd
import cupy
from cuml.metrics import pairwise_distances
from nvidia.cheminformatics.utils.metrics import spearmanr
from nvidia.cheminformatics.utils.distance import tanimoto_calculate


def sanitized_smiles(smiles):
    """Ensure SMILES are valid and sanitized, otherwise fill with NaN."""
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol:
        sanitized_smiles = Chem.MolToSmiles(mol)
        return sanitized_smiles
    else:
        return np.NaN


class Metric():
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
            print('           SMILES', index, smiles)
            result = self.sample(smiles, num_samples, func, radius)
            metric_result.append(result)
        return np.array(metric_result)

    def calculate(self, smiles, num_samples, func, radius):
        metric_array = self.sample_many(smiles, num_samples, func, radius)
        metric = self.calculate_metric(metric_array, num_samples)
        return pd.Series({self.name:metric, 'radius':radius, 'num_samples':num_samples})


class Validity(Metric):
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


class Unique(Metric):
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

    def calculate_metric(self, metric_array, num_samples):
        total_samples = len(metric_array) * num_samples
        return np.nansum(metric_array) / float(total_samples)


class NearestNeighborCorrelation(Metric):
    """Sperman's Rho for correlation of pairwise Tanimoto distances vs Euclidean distance from embeddings"""
    # TODO return padding array and zero out padded values
    
    def __init__(self):
        self.name = 'nearest neighbor correlation'

    def sample(self, smiles, max_len, stub):
        func = stub.SmilesToEmbedding

        spec = generativesampler_pb2.GenerativeSpec(
                model=generativesampler_pb2.GenerativeModel.MegaMolBART,
                smiles=smiles,
                radius=0.0001, # dummy var for this calculation
                numRequested=1, # dummy var for this calculation
                padding=max_len)

        result = func(spec)
        # Shape is the first two values -- not needed           
        embedding = cupy.array(result.embedding[2:])

        # Zero out padded values
        embedding = embedding.reshape(max_len, -1)
        embedding[len(smiles):, :] = 0.0
        
        embedding = embedding.flatten()
        return embedding

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

    def sample_many(self, smiles_dataset, stub):
        # Calculate pairwise distances for embeddings
        embeddings = []
        for smiles in smiles_dataset.data.to_pandas():
            embedding = self.sample(smiles, smiles_dataset.max_len, stub)
            embeddings.append(embedding)
        
        return cupy.asarray(embeddings)

    def calculate(self, smiles_dataset, fingerprint_dataset, stub, top_k=None):
        embeddings = self.sample_many(smiles_dataset, stub)
        metric = self.calculate_metric(embeddings, fingerprint_dataset, top_k)
        return pd.Series({self.name:cupy.nanmean(metric), 'top_k':top_k})
