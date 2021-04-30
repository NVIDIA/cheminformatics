#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
from rdkit import Chem

import sys
sys.path.insert(0, "/workspace/megamolbart")
from megamolbart.inference import MegaMolBART, clean_smiles_list


if __name__ == '__main__':

    num_molecules = 2
    num_samples = 3
    radius_list = [0.001]

    data = pd.read_csv('/workspace/cuchem/tests/data/benchmark_approved_drugs.csv')
    if num_molecules > 0:
        data = data.sample(n=num_molecules, replace=False)

    single_smiles_results = list()
    with torch.no_grad():
        for radius in radius_list:
            wf = MegaMolBART()
            simulated_radius = wf.radius_scale / radius

            for smiles in data['canonical_smiles'].tolist():
                smiles_df = wf.find_similars_smiles(smiles, num_samples, radius=simulated_radius)
                smiles_df = smiles_df[smiles_df['Generated']]
                smiles_df = pd.DataFrame({'OUTPUT': clean_smiles_list(smiles_df['SMILES'].tolist())})
                smiles_df['INPUT'] = smiles
                smiles_df['RADIUS'] = simulated_radius
                single_smiles_results.append(smiles_df)
    
    single_smiles_results = pd.concat(single_smiles_results, axis=0).reset_index(drop=True).dropna()[['RADIUS', 'INPUT', 'OUTPUT']]
    
    percent_valid_per_query = single_smiles_results.groupby(['RADIUS', 'INPUT']).size().astype(float) / num_samples
    percent_valid_per_query.name = 'percent_valid'
    percent_unique_per_query = single_smiles_results.groupby(['RADIUS', 'INPUT']).apply(lambda x: len(set(x['OUTPUT']))).astype(float) / num_samples
    percent_unique_per_query.name = 'percent_unique'

    percent_unique = single_smiles_results.groupby('RADIUS')['OUTPUT'].nunique() / float(num_molecules * num_samples)
    percent_valid = single_smiles_results.groupby('RADIUS')['OUTPUT'].size()
    overall = pd.Series({'percent_valid': percent_valid, 'percent_unique': percent_unique})
    overall.name = 'TOTAL'

    results = pd.merge(percent_valid_per_query, percent_unique_per_query, left_index=True, right_index=True)

    # mols_df_2 = wf.interpolate_from_smiles([smiles1, smiles2], num_interp)

        
