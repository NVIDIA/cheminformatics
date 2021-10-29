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
from cuchem.benchmark.data import PhysChemEmbeddingData, SampleCacheData, ZINC15TrainDataset

# Metrics
from cuchem.benchmark.metrics.sampling import Validity, Unique, Novelty
from cuchem.benchmark.metrics.embeddings import NearestNeighborCorrelation, Modelability

# ML models supported
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from cuml import LinearRegression, ElasticNet
from cuml.svm import SVR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    os.makedirs(cfg.output.path, exist_ok=True)

    output_dir = cfg.output.path
    seq_len = int(cfg.sampling.seq_len) # TODO: pull from MegaMolBART
    input_size = int(cfg.sampling.input_size)

    if cfg.model.name == 'MegaMolBART':
        inferrer = MegatronMolBART()
    elif cfg.model.name == 'CDDD':
        inferrer = Cddd()
    else:
        logger.error(f'Model {cfg.model.name} not supported')
        sys.exit(1)

    sample_cache = SampleCacheData()
    embedding_cache = PhysChemEmbeddingData()

    training_data = ZINC15TrainDataset()

    smiles_dataset = ZINC15TestSplit(max_len=seq_len)
    smiles_dataset.load()

    # Metrics
    # for sampling_metric in [Validity, Unique, Novelty]:
    # name = sampling_metric.name
    # if eval(f'cfg.metric.{name}.enabled'):
    #     param_list = [inferrer, sample_cache, smiles_dataset]
    #     if name == 'novelty':
    #         param_list += training_data
    #     metric_list.append({name: sampling_metric(*param_list)})
    # # TODO move datasets down here

    metric_list = []
    if cfg.metric.validity.enabled:
        name = 'validity'
        metric_list.append({name: Validity(inferrer, sample_cache, smiles_dataset)})

    if cfg.metric.unique.enabled:
        name = 'unique'
        metric_list.append({name: Unique(inferrer, sample_cache, smiles_dataset)})

    if cfg.metric.novelty.enabled:
        name = 'novelty'
        metric_list.append({name: Novelty(inferrer, sample_cache, smiles_dataset, training_data)})

    if cfg.metric.nearest_neighbor_correlation.enabled:
        fingerprint_dataset = ZINC15TestSplitFingerprints()
        fingerprint_dataset.load(smiles_dataset.data.index)
        fingerprint_dataset.data = fingerprint_dataset.data.iloc[:input_size]

        metric_list.append({name: NearestNeighborCorrelation(inferrer,
                                                      embedding_cache,
                                                      smiles_dataset,
                                                      fingerprint_dataset)})

    if cfg.metric.modelability.bioactivity.enabled:
        excape_bioactivity_dataset = ExCAPEBioactivity()
        excape_fingerprint_dataset = ExCAPEFingerprints()

        excape_bioactivity_dataset.load()
        excape_fingerprint_dataset.load()

        groups = zip(excape_bioactivity_dataset.data.groupby(level=0),
                     excape_bioactivity_dataset.properties.groupby(level=0),
                     excape_fingerprint_dataset.data.groupby(level=0))

        for (label, sm_), (_, prop_), (_, fp_) in groups:
            excape_bioactivity_dataset.data = sm_
            excape_bioactivity_dataset.properties = prop_
            excape_fingerprint_dataset.data = fp_

            metric_list.append({label: Modelability('modelability-bioactivity',
                                            inferrer,
                                            embedding_cache,
                                            excape_bioactivity_dataset,
                                            excape_fingerprint_dataset)})

    if cfg.metric.modelability.physchem.enabled:
        physchem_dataset_list = [MoleculeNetESOLPhyschem(),
                                 MoleculeNetFreeSolvPhyschem(),
                                 MoleculeNetLipophilicityPhyschem()]
        physchem_fingerprint_list = [MoleculeNetESOLFingerprints(),
                                     MoleculeNetFreeSolvFingerprints(),
                                     MoleculeNetLipophilicityFingerprints()]
        for x in physchem_dataset_list:
            x.load()
        for x in physchem_fingerprint_list:
            x.load()

        groups = zip([x.table_name for x in physchem_dataset_list],
                    physchem_dataset_list,
                    physchem_fingerprint_list)

        # TODO: Ideally groups should be of sample_size size.
        for (label, smiles_, fp_) in groups:
            smiles_.data = smiles_.data.iloc[:input_size] # TODO for testing
            smiles_.properties = smiles_.properties.iloc[:input_size]  # TODO for testing
            fp_.data = fp_.data.iloc[:input_size]  # TODO for testing
            metric_list.append({label: Modelability(f'modelability-physchem',
                                            inferrer,
                                            embedding_cache,
                                            smiles_,
                                            fp_)})

    # ML models
    model_dict = get_model()

    if input_size <= 0:
        input_size = len(smiles_dataset.data)

    # Filter and rearrage data as expected by downstream components.
    smiles_dataset.data = smiles_dataset.data.iloc[:input_size]['canonical_smiles'] # TODO REMOVE THIS

    convert_runtime = lambda x: x.seconds + (x.microseconds / 1.0e6)
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
