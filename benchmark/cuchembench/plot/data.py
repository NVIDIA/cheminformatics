
import glob
import os
from datetime import datetime
import pandas as pd
from typing import List, Optional
from cuchembench.datasets.physchem import (MoleculeNetESOL,
                                        MoleculeNetFreeSolv,
                                        MoleculeNetLipophilicity)
from cuchembench.datasets.bioactivity import ExCAPEDataset

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_physchem_input_data(max_seq_len):
    """Create dataframe wtih input physchem property values that are predicted from models"""
    dataset_list = [MoleculeNetESOL(max_seq_len=max_seq_len),
                           MoleculeNetFreeSolv(max_seq_len=max_seq_len),
                           MoleculeNetLipophilicity(max_seq_len=max_seq_len)]

    input_data = {}
    for dataset in dataset_list:
        dataset.load(columns=['SMILES'])
        property_name = dataset.properties_cols[0]
        data = dataset.properties
        data = data.rename(columns={property_name: 'value'})
        input_data[property_name] = data.values.squeeze()

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
    input_data = dict(tuple(input_data.groupby('gene')['value']))
    input_data = {k: v.values.squeeze() for k, v in input_data.items()}
    return input_data


def load_benchmark_files(metric_paths: List[str], 
                         metric_labels: Optional[List[str]] = None, 
                         parse_timestamps: bool = False,
                         file_type: str = 'csv'):
    """Load aggregated metric results from CSV or pkl files"""
    
    metric_labels = metric_labels if metric_labels else [None] * len(metric_paths)
    assert len(metric_labels) == len(metric_paths)
    custom_date_parser = lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S")
    
    file_type = file_type.lower()
    assert file_type in ['csv', 'pkl'], AssertionError(f'File type must be one of "csv" or "pkl", got {file_type}.')

    combined_metric_df = []
    for metric_path, metric_label in zip(metric_paths, metric_labels):
        file_list = glob.glob(os.path.join(metric_path, '**', f'*.{file_type}'), recursive=True)
            
        metric_df = []
        for file in file_list:
            if file_type == 'csv':
                kwargs = {'parse_dates':['timestamp'], 'date_parser':custom_date_parser} if parse_timestamps else {}
                df = pd.read_csv(file, **kwargs)
            else:
                df = pd.read_pickle(file)
            
            exp_name = file.split('-')[-2] if not metric_label else metric_label
            df['exp_name'] = exp_name
            metric_df.append(df)

        metric_df = pd.concat(metric_df, axis=0).reset_index(drop=True)

        if parse_timestamps:
            metric_df['timestamp'] = metric_df['timestamp'].min()
            metric_df['timestamp'] = metric_df['timestamp'].dt.to_period('M') # Floor by month
        
        # Cleanup names -- need to regularize with team
        metric_df['name'] = metric_df['name'].str.replace('modelability-', '')
        combined_metric_df.append(metric_df)
        
    return pd.concat(combined_metric_df, axis=0).reset_index(drop=True)
