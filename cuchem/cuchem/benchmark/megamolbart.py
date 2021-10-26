import os
import sys
import time
import logging
import hydra
import pandas as pd

import numpy as np

from datetime import datetime
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from cuml import LinearRegression, ElasticNet
from cuml.svm import SVR

from cuchem.wf.generative import MegatronMolBART
from cuchem.wf.generative import Cddd
from cuchem.benchmark.data import PhysChemEmbeddingData, SampleCacheData, ZINC15TrainDataset
from cuchem.benchmark.datasets.fingerprints import ZINC15TestSplitFingerprints
from cuchem.benchmark.datasets.molecules import ZINC15TestSplit
from cuchem.benchmark.metrics.sampling import Validity, Unique, Novelty
from cuchem.benchmark.metrics.embeddings import NearestNeighborCorrelation, Modelability
from cuchem.benchmark.datasets.molecules import (ChEMBLApprovedDrugsPhyschem,
                                                 MoleculeNetESOLPhyschem,
                                                 MoleculeNetFreeSolvPhyschem,
                                                 MoleculeNetLipophilicityPhyschem )
from cuchem.benchmark.datasets.fingerprints import (ChEMBLApprovedDrugsFingerprints,
                                                    MoleculeNetESOLFingerprints,
                                                    MoleculeNetFreeSolvFingerprints,
                                                    MoleculeNetLipophilicityFingerprints )
from cuchem.benchmark.datasets.bioactivity import (ExCAPEBioactivity, ExCAPEFingerprints)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_fp(smiles, inferrer, max_len):
    emb_result = inferrer.smiles_to_embedding(smiles, max_len)
    emb = np.array(emb_result.embedding)
    emb = np.reshape(emb, emb_result.dim)
    return emb


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
        rf_estimator = RandomForestRegressor(accuracy_metric='mse', random_state=0)
        rf_param_dict = {'n_estimators': [10, 50]}

        sv_estimator = SVR(kernel='rbf')
        sv_param_dict = {'C': [0.01, 0.1, 1.0, 10], 'degree': [3,5,7,9]}

        lr_estimator = LinearRegression(normalize=True)
        lr_param_dict = {'normalize': [True]}

        en_estimator = ElasticNet(normalize=True)
        en_param_dict = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
                         'l1_ratio': [0.1, 0.5, 1.0, 10.0]}

        return {'random forest': [rf_estimator, rf_param_dict],
                'support vector machine': [sv_estimator, sv_param_dict],
                'linear regression': [lr_estimator, lr_param_dict],
                'elastic net': [en_estimator, en_param_dict]}


def save_metric_results(metric_list, output_dir):
    metric_df = pd.concat(metric_list, axis=1).T
    logger.info(metric_df)
    metric = metric_df['name'].iloc[0].replace(' ', '_')
    iteration = metric_df['iteration'].iloc[0]
    metric_df.to_csv(os.path.join(output_dir, f'{metric}_{iteration}.csv'), index=False)


@hydra.main(config_path=".", config_name="benchmark")
def main(cfg):
    logger.info(cfg)
    os.makedirs(cfg.output.path, exist_ok=True)

    output_dir = cfg.output.path
    seq_len = int(cfg.samplingSpec.seq_len) # Import from MegaMolBART codebase?
    input_size = int(cfg.samplingSpec.input_size)

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
    metric_list = []
    if cfg.metric.validity.enabled == True:
        metric_list.append(Validity(inferrer, sample_cache, smiles_dataset))

    if cfg.metric.unique.enabled == True:
        metric_list.append(Unique(inferrer, sample_cache, smiles_dataset))

    if cfg.metric.novelty.enabled == True:
        metric_list.append(Novelty(inferrer, sample_cache, smiles_dataset, training_data))

    if cfg.metric.nearest_neighbor_correlation.enabled == True:
        fingerprint_dataset = ZINC15TestSplitFingerprints()
        fingerprint_dataset.load(smiles_dataset.data.index)
        fingerprint_dataset.data = fingerprint_dataset.data.iloc[:input_size]

        metric_list.append(NearestNeighborCorrelation(inferrer,
                                                      embedding_cache,
                                                      smiles_dataset,
                                                      fingerprint_dataset))

    if cfg.metric.modelability.bio_activity.enabled == True:
        excape_bioactivity_dataset = ExCAPEBioactivity()
        excape_fingerprint_dataset = ExCAPEFingerprints()

        excape_bioactivity_dataset.load()
        excape_fingerprint_dataset.load(max_data_size=input_size)

        groups = zip(excape_bioactivity_dataset.data.groupby(level=0),
                     excape_bioactivity_dataset.properties.groupby(level=0),
                     excape_fingerprint_dataset.data.groupby(level=0))

        import cupy
        for (label, sm_), (_, prop_), (_, fp_) in groups:
            excape_bioactivity_dataset.data = sm_
            excape_bioactivity_dataset.properties = prop_
            excape_fingerprint_dataset.data = fp_

            metric_list.append(Modelability(inferrer,
                                            embedding_cache,
                                            excape_bioactivity_dataset,
                                            excape_fingerprint_dataset))

    if cfg.metric.modelability.phys_chem.enabled == True:
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

        groups = zip([x.name for x in physchem_dataset_list],
                    physchem_dataset_list,
                    physchem_fingerprint_list)

        # TODO: Ideally groups should be of sample_size size.
        for (label, smiles_, fp_) in groups:
            smiles_.data = smiles_.data.iloc[:input_size] # TODO for testing
            smiles_.properties = smiles_.properties.iloc[:input_size]  # TODO for testing
            fp_.data = fp_.data.iloc[:input_size]  # TODO for testing

            print(label, smiles_.data.head(n=1), smiles_.properties.head(n=1), fp_.data.head(n=1))
            metric_list.append(Modelability(inferrer,
                                            embedding_cache,
                                            smiles_,
                                            fp_))

    # ML models
    model_dict = get_model()

    if input_size <= 0:
        input_size = len(smiles_dataset.data)

    # Filter and rearrage data as expected by downstream components.
    smiles_dataset.data = smiles_dataset.data.iloc[:input_size]['canonical_smiles']

    convert_runtime = lambda x: x.seconds + (x.microseconds / 1.0e6)
    iteration = None
    iteration = wait_for_megamolbart_service(inferrer)

    for metric in metric_list:
        logger.info(f'METRIC: {metric.name}')
        result_list = []

        iter_list = metric.variations(cfg, model_dict=model_dict)

        for iter_val in iter_list:
            start_time = datetime.now()

            try:
                iter_val = int(iter_val)
            except ValueError:
                pass

            estimator, param_dict = None, None
            if iter_val in model_dict:
                estimator, param_dict = model_dict[iter_val]

            result = metric.calculate(top_k=iter_val, # TODO IS THIS A BUG
                                      estimator=estimator,
                                      param_dict=param_dict,
                                      num_samples=cfg.samplingSpec.sample_size,
                                      radius=iter_val)

            run_time = convert_runtime(datetime.now() - start_time)
            result['iteration'] = iteration
            result['run_time'] = run_time
            result['data_size'] = input_size
            result_list.append(result)
            save_metric_results(result_list, output_dir)

if __name__ == '__main__':
    main()
