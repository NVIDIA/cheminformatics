#!/usr/bin/env python3

import sys
import pandas as pd
import torch
import grpc
import generativesampler_pb2_grpc
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

# TODO add this path to the PYTHONPATH variable in the Dockerfile
sys.path.insert(0, '/workspace/cuchem')
from nvidia.cheminformatics.datasets.loaders import ChEMBL_20K_Samples, ChEMBL_20K_Fingerprints
from nvidia.cheminformatics.metrics.model import Validity, Unique, Novelty, NearestNeighborCorrelation, get_model_iteration

OUTPUT_DIR = '/workspace/megamolbart/benchmark'
DEFAULT_MAX_SEQ_LEN = 512 # TODO: How to import from MegaMolBART codebase?

num_samples = 10
radius_list = [0.01, 0.1] # TODO calculate radius and automate this
top_k_list = [None, 100, 500, 1000] # TODO decide on top k value

# Metrics
validity = Validity()
unique = Unique()
novelty = Novelty()
nn = NearestNeighborCorrelation()
metric_list = [nn, validity, unique, novelty]

# Datasets
smiles_dataset = ChEMBL_20K_Samples(max_len=DEFAULT_MAX_SEQ_LEN)
fingerprint_dataset = ChEMBL_20K_Fingerprints() 
n_data = len(smiles_dataset.data)

smiles_dataset.load()
fingerprint_dataset.load(smiles_dataset.data.index)
assert (fingerprint_dataset.data.index == smiles_dataset.data.index)

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
                else:
                    iter_list = radius_list
                    
                for iter_val in iter_list:
                    start_time = datetime.now()

                    if metric.name == nn.name:
                        result = metric.calculate(smiles_dataset, fingerprint_dataset, stub, top_k=iter_val)
                    else:
                        result = metric.calculate(smiles_dataset, num_samples, func, radius=iter_val)

                    run_time = convert_runtime(datetime.now() - start_time)
                    result['iteration'] = iteration
                    result['run_time'] = run_time
                    result['data_size'] = n_data
                    result_list.append(result)
                    save_metric_results(result_list)
