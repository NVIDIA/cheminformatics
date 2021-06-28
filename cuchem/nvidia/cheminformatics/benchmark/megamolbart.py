#!/usr/bin/env python3

import sys
import pandas as pd
import torch
import grpc
import generativesampler_pb2_grpc
from datetime import datetime
import os
import logging
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from cuml import LinearRegression, ElasticNet
from cuml.svm import SVR

logger = logging.getLogger(__name__)

# TODO add this path to the PYTHONPATH variable in the Dockerfile
sys.path.insert(0, '/workspace/cuchem')
from nvidia.cheminformatics.datasets.loaders import ZINC15_TestSplit_20K_Samples, ZINC15_TestSplit_20K_Fingerprints
from nvidia.cheminformatics.metrics.model import Validity, Unique, Novelty, NearestNeighborCorrelation, Modelability, get_model_iteration

OUTPUT_DIR = '/workspace/megamolbart/benchmark'
DEFAULT_MAX_SEQ_LEN = 512 # TODO: Import from MegaMolBART codebase?

num_samples = 10
radius_list = [0.01, 0.1, 0.5] # TODO calculate radius and automate this
top_k_list = [None, 50, 100, 500] # TODO decide on top k value

# Metrics
validity = Validity()
unique = Unique()
novelty = Novelty()
nn = NearestNeighborCorrelation()
modelability = Modelability()
metric_list = [nn, modelability, validity, unique, novelty]

# ML models
rf_estimator = RandomForestRegressor(accuracy_metric='mse', random_state=0)
rf_param_dict = {'n_estimators': [10, 50]}

sv_estimator = SVR(kernel='rbf')
sv_param_dict = {'C': [0.01, 0.1, 1.0, 10], 'degree': [3,5,7,9]}

lr_estimator = LinearRegression(normalize=True)
lr_param_dict = {'normalize': [True]}

en_estimator = ElasticNet(normalize=True)
en_param_dict = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100], 'l1_ratio': [0.1, 0.5, 1.0, 10.0]}

model_dict = {'randomforest': [rf_estimator, rf_param_dict],
                'supportvector': [sv_estimator, sv_param_dict],
                'linearregression': [lr_estimator, lr_param_dict],
                'elasticnet': [en_estimator, en_param_dict]}

# Datasets
smiles_dataset = ZINC15_TestSplit_20K_Samples(max_len=DEFAULT_MAX_SEQ_LEN)
fingerprint_dataset = ZINC15_TestSplit_20K_Fingerprints() 
smiles_dataset.load()
fingerprint_dataset.load(smiles_dataset.data.index)
n_data = len(smiles_dataset.data)
assert (fingerprint_dataset.data.index == smiles_dataset.data.index)

# DEBUG
n_data = 10000
smiles_dataset.data = smiles_dataset.data.iloc[:n_data]
smiles_dataset.properties = smiles_dataset.properties.iloc[:n_data]
fingerprint_dataset.data = fingerprint_dataset.data.iloc[:n_data]

def save_metric_results(metric_list):
    metric_df = pd.concat(metric_list, axis=1).T
    metric = metric_df['name'].iloc[0].replace(' ', '_')
    iteration = metric_df['iteration'].iloc[0]
    metric_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', f'{metric}_{iteration}.csv'), index=False)

convert_runtime = lambda x: x.seconds + (x.microseconds / 1.0e6)

if __name__ == '__main__':

    with torch.no_grad():
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = generativesampler_pb2_grpc.GenerativeSamplerStub(channel)
            func = stub.FindSimilars
            iteration = get_model_iteration(stub)

            for metric in metric_list:
                logger.info(f'METRIC: {metric}')
                result_list = []

                if metric.name == nn.name:
                    iter_list = top_k_list
                elif metric.name == modelability.name:
                    iter_list = list(model_dict.keys())
                else:
                    iter_list = radius_list
                    
                for iter_val in iter_list:
                    start_time = datetime.now()

                    if metric.name == nn.name:
                        result = metric.calculate(smiles_dataset, fingerprint_dataset, stub, top_k=iter_val)
                    elif metric.name == modelability.name:
                        estimator, param_dict = model_dict[iter_val]
                        result = metric.calculate(smiles_dataset, fingerprint_dataset, smiles_dataset.properties, stub, estimator, param_dict)
                        result['model'] = iter_val
                    else:
                        result = metric.calculate(smiles_dataset, num_samples, func, radius=iter_val)

                    run_time = convert_runtime(datetime.now() - start_time)
                    result['iteration'] = iteration
                    result['run_time'] = run_time
                    result['data_size'] = n_data
                    result_list.append(result)
                    save_metric_results(result_list)
