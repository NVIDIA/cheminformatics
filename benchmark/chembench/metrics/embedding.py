import time
import logging
import numpy as np

from sklearn.model_selection import ParameterGrid, KFold

import cupy as cp
from cuml.metrics import pairwise_distances, mean_squared_error
from cuml.linear_model import LinearRegression, ElasticNet
from cuml.metrics.regression import r2_score
from cuml.svm import SVR
from cuml.ensemble import RandomForestRegressor
from cuml.experimental.preprocessing import StandardScaler

from . import BaseEmbeddingMetric

from chembench.data.memcache import Cache
from chembench.utils.metrics import spearmanr
from chembench.utils.distance import tanimoto_calculate


log = logging.getLogger(__name__)

__all__ = ['NearestNeighborCorrelation', 'Modelability']


def get_model_dict():
    # Normalization done by StandardScaler
    lr_estimator = LinearRegression(normalize=True)
    lr_param_dict = {'normalize': [True]}

    en_estimator = ElasticNet(normalize=True)
    en_param_dict = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
                     'l1_ratio': [0.0, 0.2, 0.5, 0.7, 1.0]}

    sv_estimator = SVR(kernel='rbf')
    sv_param_dict = {'C': [1.75, 5.0, 7.5, 10.0, 20.0],
                     'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0],
                     'epsilon': [0.001, 0.01, 0.1, 0.3],
                     'degree': [3, 5, 7, 9]}

    rf_estimator = RandomForestRegressor(accuracy_metric='mse', random_state=0)
    rf_param_dict = {'n_estimators': [50, 100, 150, 200, 500, 750, 1000]}

    return {'linear_regression': [lr_estimator, lr_param_dict],
            'support_vector_machine': [sv_estimator, sv_param_dict],
            'random_forest': [rf_estimator, rf_param_dict],
            }


class NearestNeighborCorrelation(BaseEmbeddingMetric):
    """Sperman's Rho for correlation of pairwise Tanimoto distances vs Euclidean distance from embeddings"""
    name = 'nearest neighbor correlation'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)

    def variations(self, **kwargs):
        top_k_list = list(self.metric_spec['top_k'])

        kwargs = [{'top_k': top_k} for top_k in top_k_list]
        return kwargs

    def _calculate_metric(self, embeddings, fingerprints, top_k=None):
        if embeddings.dtype != np.float32 and embeddings.dtype != np.float64:
            embeddings = embeddings.astype(np.float32)

        embeddings_dist = pairwise_distances(embeddings)
        del embeddings

        fingerprints_dist = tanimoto_calculate(fingerprints, calc_distance=True)
        del fingerprints

        corr = spearmanr(fingerprints_dist, embeddings_dist, top_k=top_k)
        return corr

    def calculate(self, **kwargs):

        top_k = kwargs['top_k']
        average_tokens = self.cfg.model.average_tokens

        start_time = time.time()
        cache = Cache()
        embeddings = None # TODO: cache.get_data('NN_embeddings')
        if embeddings is None:
            smis = self.smiles_dataset[self.dataset.smiles_col]
            embeddings = self.encode_many(smis, zero_padded_vals=False,
                                          average_tokens=average_tokens)

            log.info(f'Embedding len and type {len(embeddings)}  {type(embeddings[0])}')
            embeddings = cp.vstack(embeddings)
            fingerprints = cp.asarray(self.fingerprint_dataset)

            cache.set_data('NN_embeddings', embeddings)
            cache.set_data('NN_fingerprints', fingerprints)
        else:
            fingerprints = cache.get_data('NN_fingerprints')

        logging.info('Encoding time: {}'.format(time.time() - start_time))

        metric = self._calculate_metric(embeddings, fingerprints, top_k)
        metric = cp.nanmean(metric)
        metric = cp.asnumpy(metric)

        top_k = embeddings.shape[0] - 1 if not top_k else top_k

        return {'name': self.name, 'value': metric, 'top_k': top_k}

    def cleanup(self):
        cache = Cache()
        cache.delete('NN_embeddings')
        cache.delete('NN_fingerprints')


class Modelability(BaseEmbeddingMetric):
    """Ability to model molecular properties from embeddings vs Morgan Fingerprints"""
    name = 'modelability'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)
        # self.name = name
        self.model_dict = get_model_dict()
        self.metric_spec = metric_spec
        self.n_splits = metric_spec['n_splits']
        self.return_predictions = metric_spec['return_predictions']

        if metric_spec['normalize_inputs']:
            self.norm_data, self.norm_prop = StandardScaler(), StandardScaler()
        else:
            self.norm_data, self.norm_prop = None, None

    def variations(self, model_dict=None, **kwargs):
        # TODO: Revisit get_model_dict to make this function simpler.
        if 'variations' in self.dataset_spec:
            group_by = self.dataset_spec['variations']['group_by']
            groups = self.dataset.smiles.groupby(level=group_by)

            kwargs = []
            for gene, smis in groups:
                gene_kwargs = [{'model': k,
                                'average_tokens': self.cfg.model.average_tokens,
                                'estimator': self.model_dict[k][0],
                                'param_dict': self.model_dict[k][1]} for k in self.model_dict.keys()]

                properties = self.dataset.properties.loc[
                    smis.index.get_level_values(self.dataset_spec['index_col'])]
                fingerprints = self.dataset.fingerprints.loc[
                    smis.index.get_level_values(self.dataset_spec['index_col'])]
                smis = smis['canonical_smiles'].tolist()
                for ar in gene_kwargs:
                    ar['gene'] = gene
                    ar['smiles'] = smis
                    ar['properties'] = properties
                    ar['fingerprints'] = fingerprints
                    kwargs.append(ar)
        else:
            kwargs = [{'model': k,
                       'average_tokens': self.cfg.model.average_tokens,
                       'estimator': self.model_dict[k][0],
                       'param_dict': self.model_dict[k][1]} for k in self.model_dict.keys()]
        return kwargs

    def gpu_gridsearch_cv(self, estimator, param_dict, xdata, ydata):
        """Perform grid search with cross validation and return score"""
        log.info(f"Validating input shape {xdata.shape[0]} == {ydata.shape[0]}")
        assert xdata.shape[0] == ydata.shape[0]

        best_score, best_param, best_pred = np.inf, None, None
        # TODO -- if RF method throws errors with large number of estimators, can prune params based on dataset size.
        for param in ParameterGrid(param_dict):
            estimator.set_params(**param)
            log.info(f"Grid search param {param}")

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
                    ytrain = self.norm_prop.fit_transform(ytrain[:, cp.newaxis]).squeeze()
                    ytest = self.norm_prop.transform(ytest[:, cp.newaxis]).squeeze()

                estimator.fit(xtrain, ytrain)
                ypred = estimator.predict(xtest)

                # Ensure error is calculated on untransformed data for external comparison
                if self.norm_prop is not None:
                    ytest_unxform = self.norm_prop.inverse_transform(ytest[:, cp.newaxis]).squeeze()
                    ypred_unxform = self.norm_prop.inverse_transform(ypred[:, cp.newaxis]).squeeze()
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
            ydata_pred = self.norm_prop.fit_transform(ydata[:, cp.newaxis]).squeeze() if self.norm_prop is not None else ydata

            estimator.set_params(**best_param)
            estimator.fit(xdata_pred, ydata_pred)

            best_pred = estimator.predict(xdata_pred)
            best_pred = self.norm_prop.inverse_transform(best_pred[:, cp.newaxis]).squeeze() if self.norm_prop is not None else best_pred

        return best_score, best_param, best_pred

    def _calculate_metric(self, embeddings, fingerprints, properties, estimator, param_dict):
        """Perform grid search for each metric and calculate ratio"""
        assert len(properties.columns) == 1
        prop_name = properties.columns[0]
        properties = cp.asarray(properties[prop_name], dtype=cp.float32)

        embedding_error, embedding_param, embedding_pred = \
            self.gpu_gridsearch_cv(estimator, param_dict, embeddings, properties)
        fingerprint_error, fingerprint_param, fingerprint_pred = \
            self.gpu_gridsearch_cv(estimator, param_dict, fingerprints, properties)
        ratio = fingerprint_error / embedding_error # If ratio > 1.0 --> embedding error is smaller --> embedding model is better

        r2_emb = r2_score(properties, embedding_pred)
        rmse_emb = mean_squared_error(properties, embedding_pred, squared=False).item()

        r2_fp = r2_score(properties, fingerprint_pred)
        rmse_fp = mean_squared_error(properties, fingerprint_pred, squared=False).item()

        results = {'value': ratio,
                   'embedding_error': embedding_error,
                   'embedding_r2': r2_emb,
                   'embedding_rmse': rmse_emb,
                   'embedding_param': embedding_param,
                   'fingerprint_error': fingerprint_error,
                   'fingerprint_r2': r2_fp,
                   'fingerprint_rmse': rmse_fp,
                   'fingerprint_param': fingerprint_param,
                   'predictions': {'fingerprint_pred': cp.asnumpy(fingerprint_pred),
                                   'embedding_pred': cp.asnumpy(embedding_pred)} }
        return results

    def calculate(self, **kwargs):
        estimator = kwargs['estimator']
        param_dict = kwargs['param_dict']
        average_tokens = kwargs['average_tokens']

        # TODO: Revisit to eliminate the need for label to be an object level variable
        if 'gene' in kwargs:
            self.label = kwargs['gene']

        log.info(f'Processing {self.label}...')
        cache = Cache()
        embeddings = None #cache.get_data(f'Modelability_{self.label}_embeddings') Caching with this label is unaccurate for benchmarking multiple models
        if embeddings is None:
            log.info(f'Grabbing Fresh Embeddings with average_tokens = {average_tokens}')
            if 'smiles' in kwargs:
                smis = kwargs['smiles']
            else:
                smis = self.smiles_dataset['canonical_smiles']
            embeddings = self.encode_many(smis, zero_padded_vals=False,
                                          average_tokens=average_tokens)
            embeddings = cp.asarray(embeddings, dtype=cp.float32)

            if 'fingerprints' in kwargs:
                fingerprints = kwargs['fingerprints']
                fingerprints = cp.asarray(fingerprints.values,
                                          dtype=cp.float32)
            else:
                fingerprints = cp.asarray(self.fingerprint_dataset.values,
                                          dtype=cp.float32)

            cache.set_data(f'Modelability_{self.label}_embeddings', embeddings)
            cache.set_data(f'Modelability_{self.label}_fingerprints', fingerprints)
        else:
            fingerprints = cache.get_data('Modelability_' + self.label + '_fingerprints')

        assert embeddings.ndim == 2, AssertionError('Embeddings are not of dimension 2')
        assert fingerprints.ndim == 2, AssertionError('Fingerprints are not of dimension 2')
        assert embeddings.shape[0] == fingerprints.shape[0], AssertionError('Number of samples in embeddings and fingerprints do not match')

        log.info("Computing metric...")
        if 'smiles' in kwargs:
            properties = kwargs['properties']
        else:
            properties = self.smiles_properties

        results = []
        result = self._calculate_metric(embeddings,
                                        fingerprints,
                                        properties,
                                        estimator,
                                        param_dict)
        result['property'] = properties.columns[0]
        result['name'] = self.name
        results.append(result)

        if self.metric_spec.get('y_randomize'):
            randomzed_embedding = embeddings.copy()
            # randomzed_fingerprints = fingerprints.copy()
            for i in range(self.metric_spec['random_iterations']):
                cp.random.shuffle(randomzed_embedding)
                # cp.random.shuffle(randomzed_fingerprints)
                result = self._calculate_metric(randomzed_embedding,
                                                fingerprints,
                                                properties,
                                                estimator,
                                                param_dict)
                result['property'] = properties.columns[0]
                result['name'] = self.name
                result['model'] = f'y_randomized_{kwargs["model"]}_{i}'
                results.append(result)

        return results

    def cleanup(self):
        cache = Cache()
        cache.delete('Modelability_' + self.label + '_embeddings')
        cache.delete('Modelability_' + self.label + '_fingerprints')
