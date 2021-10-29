import os
import sys
import time
import logging
import hydra
import pandas as pd
import numpy as np
from datetime import datetime

# DL Models supported
from cuchem.wf.generative import MegatronMolBART
from cuchem.wf.generative import Cddd

# Dataset classess
from cuchem.benchmark.datasets.fingerprints import ZINC15TestSplitFingerprints
from cuchem.benchmark.datasets.molecules import ZINC15TestSplit
from cuchem.benchmark.datasets.molecules import (ChEMBLApprovedDrugsPhyschem,
                                                 MoleculeNetESOLPhyschem,
                                                 MoleculeNetFreeSolvPhyschem,
                                                 MoleculeNetLipophilicityPhyschem )
from cuchem.benchmark.datasets.fingerprints import (ChEMBLApprovedDrugsFingerprints,
                                                    MoleculeNetESOLFingerprints,
                                                    MoleculeNetFreeSolvFingerprints,
                                                    MoleculeNetLipophilicityFingerprints )
from cuchem.benchmark.datasets.bioactivity import (ExCAPEBioactivity, ExCAPEFingerprints)

# Data caches
from cuchem.benchmark.data import PhysChemEmbeddingData, BioActivityEmbeddingData, SampleCacheData, ZINC15TrainDataset

# Metrics
from cuchem.benchmark.metrics.sampling import Validity, Unique, Novelty
from cuchem.benchmark.metrics.embeddings import NearestNeighborCorrelation, Modelability

# ML models supported
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from cuml import LinearRegression, ElasticNet
from cuml.svm import SVR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_runtime(time_): 
    return time_.seconds + (time_.microseconds / 1.0e6)

def wait_for_megamolbart_service(inferrer):
    retry_count = 0
    while retry_count < 30:
        try:
            # Wait for upto 5 min for the server to be up
            iteration = inferrer.get_iteration()
            break
        except Exception as e:
            logging.warning(f'Service not available. Retrying {retry_count}...')
            time.sleep(10)
            retry_count += 1
            continue
    logging.info(f'Service found after {retry_count} retries.')
    return iteration


def get_model():
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


def save_metric_results(metric_list, output_dir):
    metric_df = pd.concat(metric_list, axis=1).T
    logger.info(metric_df)
    metric = metric_df['name'].iloc[0].replace(' ', '_')
    iteration = metric_df['iteration'].iloc[0]
    metric_df.to_csv(os.path.join(output_dir, f'{metric}_iteration{iteration}.csv'), index=False)


@hydra.main(config_path=".", config_name="benchmark")
def main(cfg):
    logger.info(cfg)

    output_dir = cfg.output.path
    os.makedirs(output_dir, exist_ok=True)

    input_size = None
    if cfg.sampling.input_size:
        i_size_ = int(cfg.sampling.input_size)
        input_size = i_size_ if i_size_ > 0 else input_size

    max_seq_len = int(cfg.sampling.max_seq_len) # TODO: pull from inferrer

    # Inferrer (DL model)
    if cfg.model.name == 'MegaMolBART':
        inferrer = MegatronMolBART()
    elif cfg.model.name == 'CDDD':
        inferrer = Cddd()
    else:
        logger.error(f'Model {cfg.model.name} not supported')
        sys.exit(1)

    # Metrics
    metric_list = []

    for sampling_metric in [Validity, Unique, Novelty]:
        name = sampling_metric.name

        if eval(f'cfg.metric.{name}.enabled'):
            smiles_dataset = ZINC15TestSplit(max_seq_len=max_seq_len)
            smiles_dataset.load(data_len=input_size)
            sample_cache = SampleCacheData()

            param_list = [inferrer, sample_cache, smiles_dataset]
            if name == 'novelty':
                training_data = ZINC15TrainDataset()
                param_list += [training_data]
            metric_list.append({name: sampling_metric(*param_list)})

    if cfg.metric.nearest_neighbor_correlation.enabled:
        name = 'nearest neighbor correlation'

        smiles_dataset = ZINC15TestSplit(max_seq_len=max_seq_len) # TODO FIX THIS ChEMBLApprovedDrugsPhyschem(max_seq_len=max_seq_len)
        smiles_dataset.load(data_len=input_size)
        fingerprint_dataset = ZINC15TestSplitFingerprints() # TODO FIX THIS ChEMBLApprovedDrugsFingerprints()
        fingerprint_dataset.load(smiles_dataset.data.index)
        assert len(smiles_dataset.data) == len(fingerprint_dataset.data)

        embedding_cache = PhysChemEmbeddingData() # TODO FIX THIS

        metric_list.append({name: NearestNeighborCorrelation(inferrer,
                                                      embedding_cache,
                                                      smiles_dataset,
                                                      fingerprint_dataset)})

    if cfg.metric.modelability.bioactivity.enabled:
        smiles_dataset = ExCAPEBioactivity(max_seq_len=max_seq_len)
        fingerprint_dataset = ExCAPEFingerprints()
        embedding_cache = PhysChemEmbeddingData() # TODO FIX THIS BioActivityEmbeddingData()

        smiles_dataset.load(data_len=input_size)
        fingerprint_dataset.load(smiles_dataset.data.index)

        groups = list(zip(smiles_dataset.data.groupby(level=0),
                     smiles_dataset.properties.groupby(level=0),
                     fingerprint_dataset.data.groupby(level=0)))

        for (label, sm_), (_, prop_), (_, fp_) in groups:
            smiles_dataset.data = sm_ # TODO ensure this isn't overwriting the original dataset
            smiles_dataset.properties = prop_
            fingerprint_dataset.data = fp_

            metric_list.append({label: Modelability('modelability-bioactivity',
                                            inferrer,
                                            embedding_cache,
                                            smiles_dataset,
                                            fingerprint_dataset)})

    if cfg.metric.modelability.physchem.enabled:
        smiles_dataset_list = [MoleculeNetESOLPhyschem(max_seq_len=max_seq_len),
                                 MoleculeNetFreeSolvPhyschem(max_seq_len=max_seq_len),
                                 MoleculeNetLipophilicityPhyschem(max_seq_len=max_seq_len)]
        fingerprint_dataset_list = [MoleculeNetESOLFingerprints(),
                                     MoleculeNetFreeSolvFingerprints(),
                                     MoleculeNetLipophilicityFingerprints()]
        
        embedding_cache = PhysChemEmbeddingData()
        
        for smdata, fpdata in zip(smiles_dataset_list, fingerprint_dataset_list):
            smdata.load(data_len=input_size)
            fpdata.load(smdata.data.index)

        groups = zip([x.table_name for x in smiles_dataset_list],
                    smiles_dataset_list,
                    fingerprint_dataset_list)

        for (label, smiles_, fp_) in groups:
            metric_list.append({label: Modelability(f'modelability-physchem',
                                            inferrer,
                                            embedding_cache,
                                            smiles_,
                                            fp_)})

    # ML models
    model_dict = get_model() # TODO move this to metrics

    iteration = None
    iteration = wait_for_megamolbart_service(inferrer)

    for metric_dict in metric_list:
        metric_key, metric = list(metric_dict.items())[0]
        logger.info(f'Metric name: {metric.name}')

        iter_dict = metric.variations(cfg=cfg, model_dict=model_dict)
        iter_label, iter_vals = list(iter_dict.items())[0]
        
        result_list = []
        for iter_val in iter_vals:
            start_time = datetime.now()
            
            kwargs = {iter_label: iter_val}
            if metric.name.startswith('modelability'):
                estimator, param_dict = model_dict[iter_val]
                kwargs.update({'estimator': estimator, 'param_dict': param_dict})
                if metric.name.endswith('bioactivity'):
                    kwargs['gene'] = iter_label
            
            if metric.name in ['validity', 'unique', 'novelty']:
                kwargs['num_samples'] = int(cfg.sampling.sample_size)

            result = metric.calculate(**kwargs)
            run_time = convert_runtime(datetime.now() - start_time)

            result['iteration'] = iteration
            result['run_time'] = run_time
            result['data_size'] = min(input_size, metric.smiles_dataset.data.shape[0])
            
            if 'model' in kwargs:
                result['model'] = kwargs['model']
            if 'gene' in kwargs:
                result['gene'] = kwargs['gene']

            result_list.append(result)
            save_metric_results(result_list, output_dir)

if __name__ == '__main__':
    main()
