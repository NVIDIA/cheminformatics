#!/usr/bin/env python3

import sys
import pandas as pd
import torch
import grpc
import generativesampler_pb2_grpc
from datetime import datetime
import cudf
import os

# TODO add this path to the PYTHONPATH variable in the Dockerfile
sys.path.insert(0, '/workspace/cuchem')
from nvidia.cheminformatics.datasets.loaders import ChEMBL_20K_Samples, ChEMBL_20K_Fingerprints
from nvidia.cheminformatics.metrics.model import Validity, Unique, NearestNeighborCorrelation

OUTPUT_DIR = '/workspace/megamolbart/benchmark'
DEFAULT_MAX_SEQ_LEN = 250 # 512 TODO: Very long lengths appear to cause memory issues


iteration = 610000 # set manually according to model TODO: set from model class
num_samples = 10
radius_list = [0.0001, 0.001, 0.01, 0.1] # DEBUG
top_k_list = [None, 100, 500]#, 1000] # DEBUG
n_data = 1000 # DEBUG

# Metrics
validity = Validity()
unique = Unique()
nn = NearestNeighborCorrelation()
metric_list = [unique.name, validity.name]  # DEBUG [nn.name validity.name, unique.name]

# Datasets
smiles_dataset = ChEMBL_20K_Samples(max_len=DEFAULT_MAX_SEQ_LEN)
fingerprint_dataset = ChEMBL_20K_Fingerprints() 

smiles_dataset.load()
fingerprint_dataset.load(smiles_dataset.data.index)
assert (fingerprint_dataset.data.index == smiles_dataset.data.index)

smiles_dataset.data = smiles_dataset.data.iloc[:n_data] # DEBUG
fingerprint_dataset.data = fingerprint_dataset.data.iloc[:n_data] # DEBUG


def save_metric_results(metric_list, metric, iteration):
    metric_df = pd.concat(metric_list, axis=1).T
    metric = metric.replace(' ', '_')
    metric_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', f'{metric}_{iteration}.csv'), index=False)


convert_runtime = lambda x: x.seconds + (x.microseconds / 1.0e6)


if __name__ == '__main__':

    val_list, uni_list, near_list = [], [], []
    with torch.no_grad():
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = generativesampler_pb2_grpc.GenerativeSamplerStub(channel)
            func = stub.FindSimilars

            for metric in metric_list:
                print('      METRIC ', metric)
                if metric == nn.name:
                    for top_k in top_k_list:
                        start_time = datetime.now()
                        near = nn.calculate(smiles_dataset, fingerprint_dataset, stub, top_k=top_k)
                        run_time = convert_runtime(datetime.now() - start_time)
                        near['iteration'] = iteration
                        near['run_time'] = run_time
                        near['data_size'] = n_data
                        near_list.append(near)
                        save_metric_results(near_list, metric, iteration)
                    del near_list

                elif metric == validity.name:
                    for radius in radius_list:
                        start_time = datetime.now()
                        val = validity.calculate(smiles_dataset, num_samples, func, radius)
                        run_time = convert_runtime(datetime.now() - start_time)
                        val['iteration'] = iteration
                        val['run_time'] = run_time
                        val['data_size'] = n_data
                        val_list.append(val)
                        save_metric_results(val_list, metric, iteration)
                    del val_list

                elif metric == unique.name:
                    for radius in radius_list:
                        start_time = datetime.now()
                        uni = unique.calculate(smiles_dataset, num_samples, func, radius)
                        run_time = convert_runtime(datetime.now() - start_time)
                        uni['iteration'] = iteration
                        uni['run_time'] = run_time
                        uni['data_size'] = n_data
                        uni_list.append(uni)
                        save_metric_results(uni_list, metric, iteration)
                    del uni_list
