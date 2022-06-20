#!/usr/bin/env python3

import logging
import time
import numpy as np
import pandas as pd
import pickle
import sqlite3
from contextlib import closing
from sklearn.model_selection import ParameterGrid, KFold

from chembench.data.memcache import Cache

logger = logging.getLogger(__name__)

try:
    import cupy as xpy
    import cudf as xdf
    from cuml.metrics import pairwise_distances, mean_squared_error, r2_score
    from cuml.linear_model import LinearRegression, ElasticNet
    from cuml.svm import SVR
    from cuml.ensemble import RandomForestRegressor
    from chembench.utils.metrics import spearmanr
    from chembench.utils.distance import tanimoto_calculate
    from cuml.experimental.preprocessing import StandardScaler
    RAPIDS_AVAILABLE = True
    logger.info('RAPIDS installation found. Using cupy and cudf where possible.')
except ModuleNotFoundError as e:
    logger.info('RAPIDS installation not found. Numpy and pandas will be used instead.')
    import numpy as xpy
    import pandas as xdf
    from sklearn.metrics import pairwise_distances, mean_squared_error
    from sklearn.linear_model import LinearRegression, ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import spearmanr
    from chembench.utils.distance import tanimoto_calculate
    from sklearn.preprocessing import StandardScaler
    RAPIDS_AVAILABLE = False
__all__ = ['NearestNeighborCorrelation', 'Modelability']


def get_model_dict():
    lr_estimator = LinearRegression(normalize=False) # Normalization done by StandardScaler
    lr_param_dict = {'normalize': [False]}

    en_estimator = ElasticNet(normalize=False)
    en_param_dict = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
                     'l1_ratio': [0.0, 0.2, 0.5, 0.7, 1.0]}

    sv_estimator = SVR(kernel='rbf') # cache_size=4096.0 -- did not seem to improve runtime
    sv_param_dict = {'C': [1.75, 5.0, 7.5, 10.0, 20.0],
                     'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0],
                     'epsilon': [0.001, 0.01, 0.1, 0.3],
                     'degree': [3, 5, 7, 9]}
    if RAPIDS_AVAILABLE:
        rf_estimator = RandomForestRegressor(accuracy_metric='mse', random_state=0) # n_streams=12 -- did not seem to improve runtime
    else:
        rf_estimator = RandomForestRegressor(criterion='mse', random_state=0)
    rf_param_dict = {'n_estimators': [50, 100, 150, 200, 500, 750, 1000]}

    return {'linear_regression': [lr_estimator, lr_param_dict],
            # 'elastic_net': [en_estimator, en_param_dict], # Removing Elastic Net for timing
            'support_vector_machine': [sv_estimator, sv_param_dict],
            'random_forest': [rf_estimator, rf_param_dict],
            }

class BaseEmbeddingMetric():
    name = None

    """Base class for metrics based on embedding datasets"""
    def __init__(self, inferrer, cfg, dataset):
        self.name = self.__class__.__name__
        self.inferrer = inferrer
        self.cfg = cfg

        self.dataset = dataset
        self.smiles_dataset = dataset.smiles
        self.fingerprint_dataset = dataset.fingerprints
        self.smiles_properties = dataset.properties

        self.conn = sqlite3.connect(self.cfg.sampling.db,
                                    uri=True,
                                    check_same_thread=False)

    def __len__(self):
        return len(self.dataset.smiles)

    def variations(self):
        return NotImplemented

    def _find_embedding(self, smiles):
        # Check db for results from a previous run
        result = self.conn.execute('''
                SELECT ss.embedding, ss.embedding_dim
                FROM smiles s, smiles_samples ss
                WHERE s.id = ss.input_id
                    AND s.model_name = ?
                    AND s.force_unique = ?
                    AND s.sanitize = ?
                    AND ss.is_generated = 0
                    AND s.processed = 1
                    AND s.smiles = ?
                LIMIT 1
                ''',
                [self.inferrer.__class__.__name__,  0, 1, smiles]).fetchone()

        embedding = pickle.loads(result[0])
        embedding_dim = pickle.loads(result[1])
        return (embedding, embedding_dim)

    def encode(self, smiles, zero_padded_vals, average_tokens, max_seq_len=None):
        """Encode a single SMILES to embedding from model"""
        embedding, dim = self._find_embedding(smiles)
        embedding = xpy.array(embedding).reshape(dim).squeeze()
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

    def encode_many(self, max_seq_len=None, zero_padded_vals=True, average_tokens=False):
        # Calculate pairwise distances for embeddings

        if not max_seq_len:
            max_seq_len = self.dataset.max_seq_len

        embeddings = []
        for smiles in self.smiles_dataset['canonical_smiles']:
            embedding = self.encode(smiles, zero_padded_vals, average_tokens, max_seq_len=max_seq_len)
            embeddings.append(embedding)

        return embeddings

    def calculate(self):
        raise NotImplementedError

    def cleanup(self):
        pass


class NearestNeighborCorrelation(BaseEmbeddingMetric):
    """Sperman's Rho for correlation of pairwise Tanimoto distances vs Euclidean distance from embeddings"""
    name = 'nearest neighbor correlation'

    def __init__(self, inferrer, cfg, smiles_dataset):
        super().__init__(inferrer, cfg, smiles_dataset)
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

    def calculate(self, top_k=None, average_tokens = False, **kwargs):

        start_time = time.time()
        cache = Cache()
        embeddings = cache.get_data('NN_embeddings')
        # TODO: Possible revisit for performance reasons
        if embeddings is None:
            embeddings = self.encode_many(zero_padded_vals=False, average_tokens=average_tokens) #zero_padded_vals=True
            logger.info(f'Embedding len and type {len(embeddings)}  {type(embeddings[0])}')
            embeddings = xpy.vstack(embeddings)
            fingerprints = xpy.asarray(self.fingerprint_dataset)

            cache.set_data('NN_embeddings', embeddings)
            cache.set_data('NN_fingerprints', fingerprints)
        else:
            fingerprints = cache.get_data('NN_fingerprints')

        logging.info('Encoding time: {}'.format(time.time() - start_time))

        metric = self._calculate_metric(embeddings, fingerprints, top_k)
        metric = xpy.nanmean(metric)
        if RAPIDS_AVAILABLE:
            metric = xpy.asnumpy(metric)

        top_k = embeddings.shape[0] - 1 if not top_k else top_k

        return {'name': self.name, 'value': metric, 'top_k': top_k}

    def cleanup(self):
        cache = Cache()
        cache.delete('NN_embeddings')
        cache.delete('NN_fingerprints')


class Modelability(BaseEmbeddingMetric):
    """Ability to model molecular properties from embeddings vs Morgan Fingerprints"""
    name = 'modelability'

    def __init__(self, name, inferrer, cfg, dataset, label,
                 n_splits=4,
                 return_predictions=False,
                 normalize_inputs=False,
                 data_file=None):
        super().__init__(inferrer, cfg, dataset)
        self.name = name
        self.model_dict = get_model_dict()
        self.n_splits = n_splits
        self.return_predictions = return_predictions
        self.label = label
        self.data_file = data_file

        if normalize_inputs:
            self.norm_data, self.norm_prop = StandardScaler(), StandardScaler()
        else:
            self.norm_data, self.norm_prop = None, None

    def variations(self, model_dict=None, **kwargs):
        if model_dict:
            self.model_dict = model_dict
        return {'model': list(self.model_dict.keys())}

    def gpu_gridsearch_cv(self, estimator, param_dict, xdata, ydata):
        """Perform grid search with cross validation and return score"""
        logger.info(f"Validating input shape {xdata.shape[0]} == {ydata.shape[0]}")
        assert xdata.shape[0] == ydata.shape[0]

        best_score, best_param, best_pred = np.inf, None, None
        # TODO -- if RF method throws errors with large number of estimators, can prune params based on dataset size.
        for param in ParameterGrid(param_dict):
            estimator.set_params(**param)
            logger.info(f"Grid search param {param}")

            # Generate CV folds
            kfold_gen = KFold(n_splits=self.n_splits, shuffle=True, random_state=0)
            kfold_mse = []

            for train_idx, test_idx in kfold_gen.split(xdata, ydata):
                xtrain, xtest, ytrain, ytest = xdata[train_idx], xdata[test_idx], ydata[train_idx], ydata[test_idx]

                if self.norm_data is not None:
                    xtrain, xtest = xtrain.copy(), xtest.copy() # Prevent repeated transforms of same data in memory
                    xtrain = self.norm_data.fit_transform(xtrain) # Must fit transform here to avoid test set leakage
                    xtest = self.norm_data.transform(xtest)

                if self.norm_prop is not None:
                    ytrain, ytest = ytrain.copy(), ytest.copy()
                    ytrain = self.norm_prop.fit_transform(ytrain[:, xpy.newaxis]).squeeze()
                    ytest = self.norm_prop.transform(ytest[:, xpy.newaxis]).squeeze()

                estimator.fit(xtrain, ytrain)
                ypred = estimator.predict(xtest)

                # Ensure error is calculated on untransformed data for external comparison
                if self.norm_prop is not None:
                    ytest_unxform = self.norm_prop.inverse_transform(ytest[:, xpy.newaxis]).squeeze()
                    ypred_unxform = self.norm_prop.inverse_transform(ypred[:, xpy.newaxis]).squeeze()
                else:
                    ytest_unxform, ypred_unxform = ytest, ypred

                # NOTE: convert to negative MSE and maximize metric if SKLearn GridSearch is ever used
                mse = mean_squared_error(ypred_unxform, ytest_unxform).item()
                # r2 = r2_score(ypred_unxform, ytest_unxform).item()
                kfold_mse.append(mse)

            avg_mse = np.nanmean(np.array(kfold_mse))
            if avg_mse < best_score:
                best_score, best_param = avg_mse, param

        if self.return_predictions:
            xdata_pred = self.norm_data.fit_transform(xdata) if self.norm_data is not None else xdata
            ydata_pred = self.norm_prop.fit_transform(ydata[:, xpy.newaxis]).squeeze() if self.norm_prop is not None else ydata

            estimator.set_params(**best_param)
            estimator.fit(xdata_pred, ydata_pred)

            best_pred = estimator.predict(xdata_pred)
            best_pred = self.norm_prop.inverse_transform(best_pred[:, xpy.newaxis]).squeeze() if self.norm_prop is not None else best_pred

        return best_score, best_param, best_pred

    def _calculate_metric(self, embeddings, fingerprints, estimator, param_dict):
        """Perform grid search for each metric and calculate ratio"""
        properties = self.smiles_properties
        assert len(properties.columns) == 1
        prop_name = properties.columns[0]
        properties = xpy.asarray(properties[prop_name], dtype=xpy.float32)

        embedding_error, embedding_param, embedding_pred = self.gpu_gridsearch_cv(estimator, param_dict, embeddings, properties)
        fingerprint_error, fingerprint_param, fingerprint_pred = self.gpu_gridsearch_cv(estimator, param_dict, fingerprints, properties)
        ratio = fingerprint_error / embedding_error # If ratio > 1.0 --> embedding error is smaller --> embedding model is better

        if self.return_predictions & RAPIDS_AVAILABLE:
            embedding_pred, fingerprint_pred = xpy.asnumpy(embedding_pred), xpy.asnumpy(fingerprint_pred)

        results = {'value': ratio,
                   'fingerprint_error': fingerprint_error,
                   'embedding_error': embedding_error,
                   'fingerprint_param': fingerprint_param,
                   'embedding_param': embedding_param,
                   'predictions': {'fingerprint_pred': fingerprint_pred,
                                   'embedding_pred': embedding_pred} }
        return results

    def encode_bulk(self, zero_padded_vals=True, average_tokens=False):
        tmp_table = self.name.replace('-', '_')
        tmp_table += '_' + self.label

        # Create temp table if not already created
        result = self.conn.execute('''
                SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?
                ''',
                [tmp_table]).fetchone()
        if result[0] == 0:
            logger.info(f'Creating temporary table {tmp_table} for bulk encoding...')
            self.conn.execute(f'''
                    CREATE TABLE {tmp_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        smiles TEXT NULL,
                        UNIQUE(smiles)
                    )
                    ''')
            self.conn.execute(f'''
                CREATE INDEX IF NOT EXISTS {tmp_table}_smiles_index
                    ON {tmp_table} (smiles);
                ''')

        # Insert data into temp table if not already inserted
        result = self.conn.execute(f'SELECT count(*) FROM {tmp_table}').fetchone()
        if result[0] == 0:
            dataset_df = pd.DataFrame()
            dataset_df['smiles'] = self.smiles_dataset['canonical_smiles']
            dataset_df = dataset_df.drop_duplicates()
            dataset_df.to_sql(tmp_table, self.conn, index=False, if_exists='append')

            result = self.conn.execute(f'SELECT count(*) FROM {tmp_table}').fetchone()
        logger.info(f'{result[0]} SMILES to encode')

        with closing(self.conn.cursor()) as cursor:
            cursor.execute(f'''
                    SELECT ss.embedding, ss.embedding_dim, s.smiles
                    FROM smiles s, smiles_samples ss, {tmp_table} tmp
                    WHERE s.id = ss.input_id
                        AND s.model_name = ?
                        AND s.force_unique = ?
                        AND s.sanitize = ?
                        AND ss.is_generated = 0
                        AND s.processed = 1
                        AND s.smiles = tmp.smiles
                    ''',
                    [self.inferrer.__class__.__name__,  0, 1])
            embeddings = []
            while True:
                recs = cursor.fetchmany(1000)

                if not recs:
                    break
                for rec in recs:
                    dim = pickle.loads(rec[1])
                    embedding = pickle.loads(rec[0])
                    embedding = xpy.array(embedding).reshape(dim).squeeze()

                    smiles = rec[2]

                    if zero_padded_vals:
                        if dim == 2:
                            embedding[len(smiles):, :] = 0.0
                        else:
                            embedding[len(smiles):] = 0.0

                    if average_tokens:
                        embedding = embedding[:len(smiles)].mean(axis=0).squeeze()
                    else:
                        embedding = embedding.flatten()

                    embeddings.append(embedding)
        return embeddings

    def calculate(self, estimator, param_dict, average_tokens, **kwargs):
        logger.info(f'Processing {self.label}...')
        cache = Cache()
        embeddings = None #cache.get_data(f'Modelability_{self.label}_embeddings') Caching with this label is unaccurate for benchmarking multiple models
        assert(embeddings is None)
        if embeddings is None:
            logger.info(f'Grabbing Fresh Embeddings with average_tokens = {average_tokens}')
            embeddings = self.encode_many(zero_padded_vals=False, average_tokens=average_tokens)
            embeddings = xpy.asarray(embeddings, dtype=xpy.float32)
            fingerprints = xpy.asarray(self.fingerprint_dataset.values, dtype=xpy.float32)

            cache.set_data(f'Modelability_{self.label}_embeddings', embeddings)
            cache.set_data(f'Modelability_{self.label}_fingerprints', fingerprints)
        else:
            fingerprints = cache.get_data('Modelability_' + self.label + '_fingerprints')

        assert embeddings.ndim == 2, AssertionError('Embeddings are not of dimension 2')
        assert fingerprints.ndim == 2, AssertionError('Fingerprints are not of dimension 2')
        assert embeddings.shape[0] == fingerprints.shape[0], AssertionError('Number of samples in embeddings and fingerprints do not match')

        logger.info("Computing metric...")
        results = self._calculate_metric(embeddings,
                                         fingerprints,
                                         estimator,
                                         param_dict)
        results['property'] = self.smiles_properties.columns[0]
        results['name'] = self.name
        return results

    def cleanup(self):
        cache = Cache()
        cache.delete('Modelability_' + self.label + '_embeddings')
        cache.delete('Modelability_' + self.label + '_fingerprints')
