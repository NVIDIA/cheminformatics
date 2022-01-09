import pandas as pd
import glob
import os
import re
import numpy as np
from datetime import datetime
from cuchembm.datasets.physchem import (MoleculeNetESOL,
                                        MoleculeNetFreeSolv,
                                        MoleculeNetLipophilicity)
from cuchembm.datasets.bioactivity import ExCAPEDataset

MODEL_RENAME_REGEX = re.compile(r"""(?:cuchembm.inference.Grpc)?(?P<model>.+?)(?:Wrapper)?$""")

def load_aggregated_metric_results(output_dir):
    """Load aggregated metric results from CSV files"""
    custom_date_parser = lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S")
    file_list = glob.glob(os.path.join(output_dir, '**', '*.csv'), recursive=True)
    metric_df = list()

    for file in file_list:
        df = pd.read_csv(file, parse_dates=['timestamp'], date_parser=custom_date_parser)
        metric_df.append(df)

    metric_df = pd.concat(metric_df, axis=0).reset_index(drop=True)
    metric_df['timestamp'] = metric_df['timestamp'].dt.to_period('M') # Floor by month
    metric_df['name'] = metric_df['name'].str.replace('modelability-', '')
    metric_df['inferrer'] = metric_df['inferrer'].str.extract(MODEL_RENAME_REGEX, expand=False)
    return metric_df


def make_aggregated_embedding_df(metric_df):
    """Select aggregated embedding metrics from metric dataframe"""
    embedding_mask = metric_df['name'].isin(['validity', 'unique', 'novelty']).pipe(np.invert)
    embedding_df = metric_df[embedding_mask]

    cat = pd.CategoricalDtype(['nearest neighbor correlation', 'physchem', 'bioactivity'], ordered=True)
    embedding_df['name'] = embedding_df['name'].astype(cat)
    cat = pd.CategoricalDtype(['linear_regression', 'elastic_net', 'support_vector_machine', 'random_forest'], ordered=True)
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
    results_data['row'] = results_data.apply(lambda x: ', '.join([x['property'], x['inferrer']]), axis=1)
    return results_data