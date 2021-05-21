#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from rdkit import Chem

import grpc
import generativesampler_pb2
import generativesampler_pb2_grpc


BENCHMARK_DRUGS_PATH = '/workspace/cuchem/tests/data/benchmark_approved_drugs.csv'
BENCHMARK_OUTPUT_PATH = '/workspace/megamolbart/benchmark'


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


def do_sampling(data, func, num_samples, radius_list, radius_scale, generation_type):
    """Sampling for single molecule and interpolation between two molecules."""
    smiles_results = list()

    for radius in radius_list:
        simulated_radius = radius / radius_scale

        for pos, smiles in enumerate(data):
            if pos % 50 == 0:
                print(radius, pos)

            spec = generativesampler_pb2.GenerativeSpec(
                model=generativesampler_pb2.GenerativeModel.MegaMolBART,
                smiles=smiles,
                radius=0.0001,
                numRequested=num_samples)

            result = func(spec)
            result = result.generatedSmiles
            smiles_df = pd.DataFrame({'SMILES': result,
                                      'Generated': [True for i in range(len(result))]})
            if generation_type == 'sample':
                smiles_df['Generated'].iat[0] = False
            else:
                smiles_df['Generated'].iat[0] = False
                smiles_df['Generated'].iat[smiles_df.shape[0] - 1] = False

            smiles_df = smiles_df[smiles_df['Generated']]
            # All molecules are valid and not the same as input
            smiles_df = pd.DataFrame({'OUTPUT': validate_smiles_list(smiles_df['SMILES'].tolist())})
            if not isinstance(smiles, list):
                smiles = [smiles]
            mask = smiles_df['OUTPUT'].isin(smiles).pipe(np.invert)
            smiles_df = smiles_df[mask]
            smiles_df['INPUT'] = ','.join(smiles)
            smiles_df['RADIUS'] = radius
            smiles_results.append(smiles_df)

    smiles_results = pd.concat(smiles_results, axis=0).reset_index(drop=True)[['RADIUS', 'INPUT', 'OUTPUT']]
    return smiles_results


def calc_statistics(df, level):
    """Calculate validity and uniqueness statistics per molecule or per radius / sampling type"""
    def _percent_valid(df):
        return len(df.dropna()) / float(len(df))

    def _percent_unique(df):
        return len(set(df.dropna().tolist())) / float(len(df))

    results = df.groupby(level=level).agg([_percent_valid, _percent_unique])
    results.columns = ['percent_valid', 'percent_unique']
    return results


def plot_results(overall):
    """Plot the overall data statistics"""
    fig, axList = plt.subplots(ncols=2, nrows=1)
    fig.set_size_inches(10, 6)
    for ax, sample, kind in zip(axList, ['interp', 'single'], ['bar', 'line']):
        plt_data = overall.loc[sample]
        plt_data.plot(kind=kind, ax=ax)
        ax.set(title=f'Latent Space Sampling Type: {sample.title()}')
        if sample == 'single':
            ax.set(xlabel='Radius', ylabel='Percent')
        else:
            ax.set(xlabel='', ylabel='Percent')
            ax.set_xticklabels([])
    fig.savefig(os.path.join(BENCHMARK_OUTPUT_PATH, 'overall.png'))


if __name__ == '__main__':
    DEFAULT_MAX_SEQ_LEN = 512
    num_molecules = 500
    num_samples = 10
    radius_list = [0.00001, 0.00005, 0.0001, 0.0005]
    data = pd.read_csv(BENCHMARK_DRUGS_PATH)
    mask = data['canonical_smiles'].map(len) <= DEFAULT_MAX_SEQ_LEN
    print(data.shape)
    data = data[mask]
    print(data.shape)

    with torch.no_grad():
        master_df = list()

        with grpc.insecure_channel('localhost:50051') as channel:
            stub = generativesampler_pb2_grpc.GenerativeSamplerStub(channel)

            for sample_type in ['single', 'interp']:

                # func is what controls which sampling is used
                if sample_type == 'single':
                    sampled_data = data['canonical_smiles'].sample(
                        n=num_molecules, replace=False, random_state=0).tolist()
                    smiles_results = do_sampling(sampled_data, stub.FindSimilars, num_samples, radius_list, 1.0, 'sample')
                else:
                    # Sample two at a time -- must ensure seed is different each time
                    sampled_data = [data['canonical_smiles'].sample(
                        n=2, replace=False, random_state=i).tolist() for i in range(num_molecules)]
                    # radius not used for sampling two at a time -- enter dummy here
                    smiles_results = do_sampling(sampled_data, stub.Interpolate, num_samples, [1.0], 1.0, 'interpolate')
                    #smiles_results['RADIUS'] = np.NaN

                smiles_results['SAMPLE'] = sample_type
                master_df.append(smiles_results)

    indexes = ['SAMPLE', 'RADIUS', 'INPUT']
    master_df = pd.concat(master_df, axis=0).set_index(indexes)
    results = calc_statistics(master_df, indexes)
    overall = calc_statistics(master_df, indexes[:-1])

    with open(os.path.join(BENCHMARK_OUTPUT_PATH, 'results.md'), 'w') as fh:
        results.reset_index().to_markdown(fh)

    with open(os.path.join(BENCHMARK_OUTPUT_PATH, 'overall.md'), 'w') as fh:
        overall.reset_index().to_markdown(fh)

    plot_results(overall)
