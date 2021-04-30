#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
from rdkit import Chem

import sys
sys.path.insert(0, "/workspace/megamolbart")
from megamolbart.inference import MegaMolBART

# TODO ensure ouput smiles != input
def validate_smiles_list(smiles_list):
    """Ensure SMILES are valid and sanitized, otherwise fill with NaN."""
    smiles_clean_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol:
            sanitized_smiles = Chem.MolToSmiles(mol)
        else:
            sanitized_smiles = np.NaN
        smiles_clean_list.append(sanitized_smiles)

    return smiles_clean_list


def do_sampling(data, func, num_samples, radius_list, radius_scale):
    """Sampling for single molecule and interpolation between two molecules."""
    smiles_results = list()
    for radius in radius_list:
        simulated_radius = radius / radius_scale
        
        for smiles in data:
            smiles_df = func(smiles, num_samples, radius=simulated_radius)
            smiles_df = smiles_df[smiles_df['Generated']]
            smiles_df = pd.DataFrame({'OUTPUT': validate_smiles_list(smiles_df['SMILES'].tolist())})
            smiles_df['RADIUS'] = radius
            if isinstance(smiles, list):
                smiles = ','.join(smiles)
            smiles_df['INPUT'] = smiles
            smiles_results.append(smiles_df)

    smiles_results = pd.concat(smiles_results, axis=0).reset_index(drop=True)[['RADIUS', 'INPUT', 'OUTPUT']]
    return smiles_results


if __name__ == '__main__':

    num_molecules = 3
    num_samples = 4
    radius_list = [0.0001, 0.001]    
    data = pd.read_csv('/workspace/cuchem/tests/data/benchmark_approved_drugs.csv')

    with torch.no_grad():
        

        wf = MegaMolBART()
        master_df = list()
        for sample_type in ['single', 'interp']:

            # func is what controls which sampling is used
            if sample_type == 'single':
                sampled_data = data['canonical_smiles'].sample(n=num_molecules, replace=False, random_state=0).tolist()
                func = wf.find_similars_smiles
            else:
                sampled_data = [data['canonical_smiles'].sample(n=2, replace=False, random_state=i).tolist() for i in range(num_molecules)]
                func = wf.interpolate_from_smiles
                
            smiles_results = do_sampling(sampled_data, func, num_samples, radius_list, wf.radius_scale)
            smiles_results['SAMPLE'] = sample_type
            master_df.append(smiles_results)
        master_df = pd.concat(master_df, axis=0).set_index(['SAMPLE', 'RADIUS', 'INPUT'])

    
    def _percent_valid(df):
        return len(df['OUTPUT'].dropna()) / float(len(df['OUTPUT']))
    
    def _percent_unique(df):
        return len(set(df['OUTPUT'].dropna().tolist())) / float(len(df['OUTPUT']))

    # Aggregate statistics for each molecule
    percent_valid_per_query = master_df.groupby(level=[0,1,2]).apply(lambda x: _percent_valid(x))
    percent_valid_per_query.name = 'percent_valid'
    percent_unique_per_query = master_df.groupby(level=[0,1,2]).apply(lambda x: _percent_unique(x))
    percent_unique_per_query.name = 'percent_unique'
    results = pd.merge(percent_valid_per_query, percent_unique_per_query, left_index=True, right_index=True)

    # Aggregate statistics for entire dataset
    percent_valid = master_df.groupby(level=[0,1]).apply(lambda x: _percent_valid(x))
    percent_valid.name = 'percent_valid'
    percent_valid = percent_valid.to_frame().reset_index()
    percent_valid['INPUT'] = 'TOTAL'
    percent_valid = percent_valid.set_index(['SAMPLE', 'RADIUS', 'INPUT'])

    percent_unique = master_df.groupby(level=[0,1]).apply(lambda x: _percent_unique(x))
    percent_unique.name = 'percent_unique'
    percent_unique = percent_unique.to_frame().reset_index()
    percent_unique['INPUT'] = 'TOTAL'
    percent_unique = percent_unique.set_index(['SAMPLE', 'RADIUS', 'INPUT'])
    overall = pd.merge(percent_valid, percent_unique, left_index=True, right_index=True)
    results = pd.concat([results, overall], axis=0)


        
