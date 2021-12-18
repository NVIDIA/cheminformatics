import os
from pydoc import locate
import logging
import hydra
import pandas as pd
from datetime import datetime
from copy import deepcopy
from cuchembm.plot import (create_aggregated_plots, make_model_plots)

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
                           CDDDTrainDataset,
                           ChEMBLApprovedDrugsEmbeddingData)

# Metrics
from cuchembm.metrics import (Validity,
                              Unique,
                              Novelty,
                              NearestNeighborCorrelation,
                              Modelability)


log = logging.getLogger('model benchmarking')

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


def save_metric_results(mode_name, metric_list, output_dir, return_predictions):
    """Save CSV for metrics"""

    metric_df = pd.concat(metric_list, axis=1).T
    metric = metric_df.iloc[0]['name'].replace(' ', '_')
    file_path = os.path.join(output_dir, f'{mode_name}_{metric}')

    if return_predictions:
        pickle_file = file_path + '.pkl'
        logging.info(f'Writing predictions to {pickle_file}...')

        if os.path.exists(pickle_file):
            pickle_df = pd.read_pickle(pickle_file)
            pickle_df = pd.concat([pickle_df, metric_df], axis=0)
        else:
            pickle_df = metric_df

        pickle_df.to_pickle(pickle_file)

    if 'predictions' in metric_df.columns:
        metric_df.drop('predictions', inplace=True, axis=1)

    log.info(metric_df)
    csv_file_path = file_path + '.csv'
    write_header = False if os.path.exists(csv_file_path) else True
    metric_df.to_csv(csv_file_path, index=False, mode='a', header=write_header)


def get_input_size(metric_cfg):
    input_size = None
    if metric_cfg.input_size:
        i_size_ = int(metric_cfg.input_size)
        input_size = i_size_ if i_size_ > 0 else input_size
    return input_size


@hydra.main(config_path=".", config_name="benchmark_metrics")
def main(cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(cfg)
    log.info(f'Timestamp: {timestamp}')

    output_dir = cfg.output.path
    os.makedirs(output_dir, exist_ok=True)

    max_seq_len = int(cfg.sampling.max_seq_len)  # TODO: pull from inferrer

    # Inferrer (DL model)
    if cfg.model.name == 'MegaMolBART':
        inferrer = MegaMolBARTWrapper()
        training_data_class = ZINC15TrainDataset
    elif cfg.model.name == 'CDDD':
        from cuchembm.inference.cddd import CdddWrapper
        inferrer = CdddWrapper()
        training_data_class = CDDDTrainDataset
    else:
        log.warning(f'Creating model {cfg.model.name} & training dataclass {cfg.model.dataclass}')
        inf_class = locate(cfg.model.name)
        inferrer = inf_class()
        training_data_class = locate(cfg.model.dataclass)

    # Metrics
    metric_list = []

    for sampling_metric in [Validity, Unique, Novelty]:
        name = sampling_metric.name
        metric_cfg = eval(f'cfg.metric.{name}')
        input_size = get_input_size(metric_cfg)

        if metric_cfg.enabled:
            smiles_dataset = ZINC15TestSplit(max_seq_len=max_seq_len)
            sample_cache = SampleCacheData()

            smiles_dataset.load(data_len=input_size)

            param_list = [inferrer, sample_cache, smiles_dataset]
            if name == 'novelty':
                training_data = training_data_class()
                param_list += [training_data]
            metric_list.append({name: sampling_metric(*param_list)})

    if cfg.metric.nearest_neighbor_correlation.enabled:
        name = NearestNeighborCorrelation.name
        metric_cfg = cfg.metric.nearest_neighbor_correlation
        input_size = get_input_size(metric_cfg)

        smiles_dataset = ChEMBLApprovedDrugs(max_seq_len=max_seq_len)
        embedding_cache = ChEMBLApprovedDrugsEmbeddingData()

        smiles_dataset.load(data_len=input_size)

        metric_list.append({name: NearestNeighborCorrelation(inferrer,
                                                             embedding_cache,
                                                             smiles_dataset)})

    if cfg.metric.modelability.physchem.enabled:
        metric_cfg = cfg.metric.modelability.physchem
        input_size = get_input_size(metric_cfg)
        # Could concat datasets to make prep similar to bioactivity
        smiles_dataset_list = [MoleculeNetESOL(max_seq_len=max_seq_len),
                               MoleculeNetFreeSolv(max_seq_len=max_seq_len),
                               MoleculeNetLipophilicity(max_seq_len=max_seq_len)]

        embedding_cache = PhysChemEmbeddingData()
        n_splits = metric_cfg.n_splits

        for smiles_dataset in smiles_dataset_list:
            log.info(f'Loading {smiles_dataset.table_name}...')
            smiles_dataset.load(data_len=input_size, columns=['SMILES'])

            metric_list.append(
                {smiles_dataset.table_name: Modelability('modelability-physchem',
                                                         inferrer,
                                                         embedding_cache,
                                                         smiles_dataset,
                                                         n_splits,
                                                         metric_cfg.return_predictions,
                                                         metric_cfg.normalize_inputs)})

    if cfg.metric.modelability.bioactivity.enabled:
        metric_cfg = cfg.metric.modelability.bioactivity
        input_size = get_input_size(metric_cfg)

        excape_dataset = ExCAPEDataset(max_seq_len=max_seq_len)
        embedding_cache = BioActivityEmbeddingData()

        excape_dataset.load(columns=['SMILES', 'Gene_Symbol'])

        log.info('Creating groups...')

        n_splits = metric_cfg.n_splits
        gene_dataset = deepcopy(excape_dataset)
        for label, sm_ in excape_dataset.smiles.groupby(level='gene'):
            gene_dataset.smiles = sm_

            index = sm_.index.get_level_values(gene_dataset.index_col)
            gene_dataset.properties = excape_dataset.properties.loc[index]
            gene_dataset.fingerprints = excape_dataset.fingerprints.loc[index]

            log.info(f'Creating bioactivity Modelability metric for {label}...')

            metric_list.append({label: Modelability('modelability-bioactivity',
                                                    inferrer,
                                                    embedding_cache,
                                                    gene_dataset,
                                                    n_splits,
                                                    metric_cfg.return_predictions,
                                                    metric_cfg.normalize_inputs)})

    wait_for_megamolbart_service(inferrer)

    for metric_dict in metric_list:
        metric_key, metric = list(metric_dict.items())[0]
        iter_dict = metric.variations(cfg=cfg)
        iter_label, iter_vals = list(iter_dict.items())[0]

        result_list = []
        for iter_val in iter_vals:
            start_time = datetime.now()
            log.debug(f'Metric name: {metric.name}::{iter_val}')

            kwargs = {iter_label: iter_val}
            if metric.name.startswith('modelability'):
                estimator, param_dict = metric.model_dict[iter_val]
                kwargs.update({'estimator': estimator, 'param_dict': param_dict})
                if metric.name.endswith('bioactivity'):
                    kwargs['n_splits'] = cfg.metric.modelability.bioactivity.n_splits
                    kwargs['gene'] = metric_key
                else:
                    kwargs['n_splits'] = cfg.metric.modelability.physchem.n_splits

            if metric.name in ['validity', 'unique', 'novelty']:
                kwargs['num_samples'] = int(cfg.sampling.sample_size)
                metric_cfg = eval('cfg.metric.' + metric.name)
                kwargs['remove_invalid'] = metric_cfg.get('remove_invalid', None)

            log.info(f'Metric name: {metric.name}::{metric_key} with args {kwargs}')
            result = metric.calculate(**kwargs)
            run_time = convert_runtime(datetime.now() - start_time)

            result['inferrer'] = cfg.model.name
            result['iteration'] = 0 # TODO: update with version from model inferrer when implemented
            result['run_time'] = run_time
            result['timestamp'] = timestamp
            result['data_size'] = len(metric.dataset.smiles)

            # Updates to irregularly used arguments
            key_list = ['model', 'gene', 'remove_invalid', 'n_splits']
            for key in key_list:
                if key in kwargs:
                    result[key] = kwargs[key]

            result_list.append(pd.Series(result))
        if result['name'].startswith('modelability'):
            metric_name = result['name'].split('-')[1]
            return_predictions = cfg.metric.modelability[metric_name]['return_predictions']
        else:
            return_predictions = False
        save_metric_results(cfg.model.name, result_list, output_dir, return_predictions=return_predictions)

    # Plotting
    create_aggregated_plots(output_dir)
    make_model_plots(max_seq_len, 'physchem', output_dir)
    make_model_plots(max_seq_len, 'bioactivity', output_dir)



if __name__ == '__main__':
    main()
