import os
import sys
import logging
import hydra
import pandas as pd
from datetime import datetime

# Dataset classess
from cuchembm.datasets.physchem import (ChEMBLApprovedDrugs,
                                        MoleculeNetESOL,
                                        MoleculeNetFreeSolv,
                                        MoleculeNetLipophilicity,
                                        ZINC15TestSplit)
from cuchembm.datasets.bioactivity import ExCAPEDataset
from cuchembm.inference.megamolbart import MegaMolBARTWrapper

# Data caches
from cuchembm.data import (PhysChemEmbeddingData,
                           BioActivityEmbeddingData,
                           SampleCacheData,
                           ZINC15TrainDataset,
                           ChEMBLApprovedDrugsEmbeddingData)

# Metrics
from cuchembm.metrics import (Validity,
                              Unique,
                              Novelty,
                              NearestNeighborCorrelation,
                              Modelability)

# LOGGING_LEVEL = logging.DEBUG
# logging.basicConfig(level=LOGGING_LEVEL, filename='/logs/benchmark.log')
# console = logging.StreamHandler()
# console.setLevel(LOGGING_LEVEL)
# console.setFormatter(logging.Formatter('%(asctime)s %(name)s [%(levelname)s]: %(message)s'))
# logging.getLogger("").addHandler(console)


log = logging.getLogger(__name__)

def convert_runtime(time_):
    return time_.seconds + (time_.microseconds / 1.0e6)


def wait_for_megamolbart_service(inferrer):
    retry_count = 0
    while retry_count < 30:
        if inferrer.is_ready(timeout=10):
            logging.info(f'Service found after {retry_count} retries.')
            return True
        else:
            logging.warning(f'Service not available. Retrying {retry_count}...')
            retry_count += 1
            continue
    return False


def save_metric_results(metric_list, output_dir):
    metric_df = pd.concat(metric_list, axis=1).T
    log.info(metric_df)
    metric = metric_df['name'].iloc[0].replace(' ', '_')
    iteration = metric_df['iteration'].iloc[0]
    csv_file_path = os.path.join(output_dir, f'{metric}_iteration{iteration}.csv')
    write_header = False if os.path.exists(csv_file_path) else True
    metric_df.to_csv(csv_file_path, index=False, mode='a', header=write_header)


@hydra.main(config_path=".", config_name="benchmark")
def main(cfg):
    log.info(cfg)

    output_dir = cfg.output.path
    os.makedirs(output_dir, exist_ok=True)

    input_size = None
    if cfg.sampling.input_size:
        i_size_ = int(cfg.sampling.input_size)
        input_size = i_size_ if i_size_ > 0 else input_size

    max_seq_len = int(cfg.sampling.max_seq_len)  # TODO: pull from inferrer

    # Inferrer (DL model)
    if cfg.model.name == 'MegaMolBART':
        inferrer = MegaMolBARTWrapper()
    elif cfg.model.name == 'CDDD':
        from cuchem.wf.generative import Cddd
        inferrer = Cddd()
    else:
        log.error(f'Model {cfg.model.name} not supported')
        sys.exit(1)

    # Metrics
    metric_list = []

    for sampling_metric in [Validity, Unique, Novelty]:
        name = sampling_metric.name

        if eval(f'cfg.metric.{name}.enabled'):
            smiles_dataset = ZINC15TestSplit(max_seq_len=max_seq_len)
            sample_cache = SampleCacheData()

            smiles_dataset.load(data_len=input_size)

            param_list = [inferrer, sample_cache, smiles_dataset]
            if name == 'novelty':
                training_data = ZINC15TrainDataset()
                param_list += [training_data]
            metric_list.append({name: sampling_metric(*param_list)})

    if cfg.metric.nearest_neighbor_correlation.enabled:
        name = NearestNeighborCorrelation.name

        smiles_dataset = ChEMBLApprovedDrugs(max_seq_len=max_seq_len)
        # fingerprint_dataset = ChEMBLApprovedDrugs()
        embedding_cache = ChEMBLApprovedDrugsEmbeddingData()

        smiles_dataset.load(data_len=input_size)
        # fingerprint_dataset.load(smiles_dataset.data.index)
        # assert len(smiles_dataset.data) == len(fingerprint_dataset.data)
        # assert smiles_dataset.data.index.equals(fingerprint_dataset.data.index)

        metric_list.append({name: NearestNeighborCorrelation(inferrer,
                                                             embedding_cache,
                                                             smiles_dataset)})

    if cfg.metric.modelability.physchem.enabled:
        # Could concat datasets to make prep similar to bioactivity
        smiles_dataset_list = [MoleculeNetESOL(max_seq_len=max_seq_len),
                               MoleculeNetFreeSolv(max_seq_len=max_seq_len),
                               MoleculeNetLipophilicity(max_seq_len=max_seq_len)]

        embedding_cache = PhysChemEmbeddingData()

        for dataset in smiles_dataset_list:
            log.info(f'Loading {dataset.table_name}...')
            dataset.load(data_len=input_size, columns=['smiles'])

            metric_list.append(
                {dataset.table_name: Modelability('modelability-physchem',
                                                  inferrer,
                                                  embedding_cache,
                                                  dataset)})

    if cfg.metric.modelability.bioactivity.enabled:

        excape_dataset = ExCAPEDataset(max_seq_len=max_seq_len)
        embedding_cache = BioActivityEmbeddingData()

        excape_dataset.load(data_len=input_size)
        log.info('Creating groups...')
        groups = list(zip(excape_dataset.smiles.groupby(level='gene'),
                          excape_dataset.properties.groupby(level='gene'),
                          excape_dataset.fingerprints.groupby(level='gene')))

        for (label, sm_), (_, prop_), (_, fp_) in groups:
            excape_dataset.smiles = sm_
            excape_dataset.properties = prop_
            excape_dataset.fingerprints = fp_
            log.info(f'Creating bioactivity Modelability for {label}...')

            metric_list.append({label: Modelability('modelability-bioactivity',
                                                    inferrer,
                                                    embedding_cache,
                                                    excape_dataset)})

    iteration = None
    iteration = wait_for_megamolbart_service(inferrer)

    for metric_dict in metric_list:
        metric_key, metric = list(metric_dict.items())[0]
        log.info(f'Metric name: {metric.name}')

        iter_dict = metric.variations(cfg=cfg)
        iter_label, iter_vals = list(iter_dict.items())[0]

        result_list = []
        for iter_val in iter_vals:
            start_time = datetime.now()

            kwargs = {iter_label: iter_val}
            if metric.name.startswith('modelability'):
                estimator, param_dict = metric.model_dict[iter_val]
                kwargs.update({'estimator': estimator, 'param_dict': param_dict})
                if metric.name.endswith('bioactivity'):
                    kwargs['gene'] = metric_key

            if metric.name in ['validity', 'unique', 'novelty']:
                kwargs['num_samples'] = int(cfg.sampling.sample_size)

            result = metric.calculate(**kwargs)
            run_time = convert_runtime(datetime.now() - start_time)

            result['iteration'] = iteration
            result['run_time'] = run_time
            result['data_size'] = len(metric.dataset.smiles)

            if 'model' in kwargs:
                result['model'] = kwargs['model']
            if 'gene' in kwargs:
                result['gene'] = kwargs['gene']

            result_list.append(result)
        save_metric_results(result_list, output_dir)


if __name__ == '__main__':
    main()
