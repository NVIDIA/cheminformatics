import logging
import pandas as pd
import glob
import os
import re
import numpy as np
from datetime import datetime
from cuchembench.datasets.physchem import (MoleculeNetESOL,
                                        MoleculeNetFreeSolv,
                                        MoleculeNetLipophilicity)
from cuchembench.datasets.bioactivity import ExCAPEDataset

logger = logging.getLogger(__name__)

MODEL_RENAME_REGEX = re.compile(r"""(?:(?:cuchem(?:bm|bench).inference.Grpc)|(?:nemo_chem.models.megamolbart.NeMo))?(?P<model>.+?)(?:Wrapper)?$""")
MODEL_SIZE_REGEX = re.compile(r"""(?P<model_size>x?small)""")

MEGAMOLBART_SAMPLE_RADIUS = {'10M': (0.1, 0.1),
                             'xsmall': (1.0, 1.0),
                             'small':  (0.70, 0.75)} 

PHYSCHEM_UNIT_RENAMER = {'logD': 'Lipophilicity (log[D])',
                         'log_solubility_(mol_per_L)': 'ESOL (log[solubility])',
                         'hydration_free_energy': 'FreeSolv (hydration free energy)'}


def load_aggregated_metric_results(metric_paths: list, metric_labels: list, homogenize_timestamp=False):
    """Load aggregated metric results from CSV files"""
    custom_date_parser = lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S")

    combined_metric_df = []
    for metric_path, metric_label in zip(metric_paths, metric_labels):
        file_list = glob.glob(os.path.join(metric_path, '**', '*.csv'), recursive=True)
        metric_df = []

        for file in file_list:
            df = pd.read_csv(file, parse_dates=['timestamp'], date_parser=custom_date_parser)
            try:
                model_size = re.search(MODEL_SIZE_REGEX, file).group('model_size')
            except:
                model_size = ""
            df['model_size'] = model_size
            metric_df.append(df)

        metric_df = pd.concat(metric_df, axis=0).reset_index(drop=True)
        metric_df['timestamp'] = metric_df['timestamp'].dt.to_period('M') # Floor by month
        if homogenize_timestamp:
            metric_df['timestamp'] = metric_df['timestamp'].min()
        
        # Cleanup names -- need to regularize with team
        metric_df['name'] = metric_df['name'].str.replace('modelability-', '')
        physchem_mask = metric_df['name'].str.contains('physchem')
        metric_df.loc[physchem_mask, 'name'] = 'physchem'

        metric_df['inferrer'] = metric_df['inferrer'].str.extract(MODEL_RENAME_REGEX, expand=False)
        if metric_label:
            metric_df['inferrer'] = metric_df['inferrer'] + '-' + metric_label
        combined_metric_df.append(metric_df)
        
    return pd.concat(combined_metric_df, axis=0).reset_index(drop=True)


def make_aggregated_embedding_df(metric_df, models=['linear_regression', 'elastic_net', 'support_vector_machine', 'random_forest']):
    """Select aggregated embedding metrics from metric dataframe"""
    embedding_mask = metric_df['name'].isin(['validity', 'unique', 'novelty']).pipe(np.invert)
    embedding_df = metric_df[embedding_mask]

    cat = pd.CategoricalDtype(['nearest neighbor correlation', 'physchem', 'bioactivity'], ordered=True)
    embedding_df['name'] = embedding_df['name'].astype(cat)

    embedding_df['model'] = embedding_df['model'].fillna('')
    mask = embedding_df['model'].isin(models + [''])
    embedding_df = embedding_df[mask]
    cat = pd.CategoricalDtype(models, ordered=True)
    embedding_df['model'] = embedding_df['model'].astype(cat)
    return embedding_df


def load_physchem_input_data(max_seq_len):
    """Create dataframe wtih input physchem property values that are predicted from models"""
    dataset_list = [MoleculeNetESOL(max_seq_len=max_seq_len),
                           MoleculeNetFreeSolv(max_seq_len=max_seq_len),
                           MoleculeNetLipophilicity(max_seq_len=max_seq_len)]

    input_data = []
    for dataset in dataset_list:
        dataset.load(columns=['SMILES'])
        data = dataset.properties

        property_name = dataset.properties_cols[0]
        data['property'] = property_name
        data.rename(columns={property_name: 'value'}, inplace=True)
        input_data.append(data)

    input_data = pd.concat(input_data, axis=0)
    return input_data


def load_bioactivity_input_data(max_seq_len):
    """Create dataframe wtih input bioactivity values that are predicted from models"""
    dataset = ExCAPEDataset(max_seq_len=max_seq_len)

    dataset.load(columns=['SMILES', 'Gene_Symbol'])
    property_name = dataset.properties_cols[0]

    input_data = dataset.properties.copy()
    input_data.index = dataset.smiles.index
    input_data.rename(columns={property_name: 'value'}, inplace=True)
    input_data = input_data.reset_index(level='gene')
    return input_data


def load_plot_data(pkl_path, input_data, group_col):
    """Create dataframe with predictions and input data"""

    results_data = []
    file_list = glob.glob(pkl_path, recursive=True)
    for results in file_list:
        logger.info(f'Loading file {results}')
        pkl_df = pd.read_pickle(results)
        for _, row in pkl_df.iterrows():
            df = pd.DataFrame(row['predictions']).rename(columns=lambda x: x.replace('_pred', ''))
            property_name = row[group_col]
            df['property'] = property_name
            df['inferrer'] = row['inferrer']
            df['model'] = row['model']

            true_value = input_data[input_data[group_col] == property_name].drop(group_col, axis=1)
            true_value.index = df.index
            assert len(df) == len(true_value)

            df = pd.concat([df, true_value], axis=1)
            assert len(df) == len(true_value)
            results_data.append(df)

    results_data = pd.concat(results_data, axis=0)
    results_data['inferrer'] = results_data['inferrer'].str.extract(MODEL_RENAME_REGEX, expand=False)
    results_data = results_data.set_index(['property', 'inferrer', 'model', 'value']).stack().reset_index().rename(columns={'level_4':'feature', 0:'prediction'})
    return results_data