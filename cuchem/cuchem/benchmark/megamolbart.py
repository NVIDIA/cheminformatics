import os
import sys
import time
import logging
import hydra
import pandas as pd

import cupy
import dask.dataframe as dd
import numpy as np

from datetime import datetime
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from cuml import LinearRegression, ElasticNet
from cuml.svm import SVR

from cuchem.wf.generative import MegatronMolBART
from cuchem.wf.generative import Cddd
from cuchem.datasets.loaders import ZINC15_TestSplit_20K_Samples, ZINC15_TestSplit_20K_Fingerprints
from cuchem.metrics.model import Validity, Unique, Novelty, NearestNeighborCorrelation, Modelability

gene_list = ["ABL1", "ACHE", "ADAM17", "ADORA2A", "ADORA2B", "ADORA3", "ADRA1A", "ADRA1D", 
             "ADRB1", "ADRB2", "ADRB3", "AKT1", "AKT2", "ALK", "ALOX5", "AR", "AURKA", 
             "AURKB", "BACE1", "CA1", "CA12", "CA2", "CA9", "CASP1", "CCKBR", "CCR2", 
             "CCR5", "CDK1", "CDK2", "CHEK1", "CHRM1", "CHRM2", "CHRM3", "CHRNA7", "CLK4", 
             "CNR1", "CNR2", "CRHR1", "CSF1R", "CTSK", "CTSS", "CYP19A1", "DHFR", "DPP4", 
             "DRD1", "DRD3", "DRD4", "DYRK1A", "EDNRA", "EGFR", "EPHX2", "ERBB2", "ESR1", 
             "ESR2", "F10", "F2", "FAAH", "FGFR1", "FLT1", "FLT3", "GHSR", "GNRHR", "GRM5", 
             "GSK3A", "GSK3B", "HDAC1", "HPGD", "HRH3", "HSD11B1", "HSP90AA1", "HTR2A", 
             "HTR2C", "HTR6", "HTR7", "IGF1R", "INSR", "ITK", "JAK2", "JAK3", "KCNH2", 
             "KDR", "KIT", "LCK", "MAOB", "MAPK14", "MAPK8", "MAPK9", "MAPKAPK2", "MC4R", 
             "MCHR1", "MET", "MMP1", "MMP13", "MMP2", "MMP3", "MMP9", "MTOR", "NPY5R", 
             "NR3C1", "NTRK1", "OPRD1", "OPRK1", "OPRL1", "OPRM1", "P2RX7", "PARP1", "PDE5A", 
             "PDGFRB", "PGR", "PIK3CA", "PIM1", "PIM2", "PLK1", "PPARA", "PPARD", "PPARG", 
             "PRKACA", "PRKCD", "PTGDR2", "PTGS2", "PTPN1", "REN", "ROCK1", "ROCK2", "S1PR1", 
             "SCN9A", "SIGMAR1", "SLC6A2", "SLC6A3", "SRC", "TACR1", "TRPV1", "VDR"]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_fp(smiles, inferrer, max_len):
    emb_result = inferrer.smiles_to_embedding(smiles, max_len)
    emb = np.array(emb_result.embedding)
    emb = np.reshape(emb, emb_result.dim)
    return emb


def fetch_filtered_excape_data(inferrer, 
                               max_len,
                               data_dir='/data/ExCAPE'):
    """
    Loads the data and filter records with reference to genes in interest
    """
    filter_data_file = os.path.join(data_dir, 'filter_data.csv')
    embedding_file = os.path.join(data_dir, 'embedding.csv')
    if os.path.exists(filter_data_file):
        filtered_df = pd.read_csv(filter_data_file)
        smiles_df = pd.read_csv(embedding_file)
    else:
        data = dd.read_csv(os.path.join(data_dir, 'pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv'),
                          delimiter='\t',
                          usecols=['SMILES', 'Gene_Symbol', 'pXC50'])

        filtered_df = data[data.Gene_Symbol.isin(gene_list)]
        filtered_df = filtered_df[filtered_df['SMILES'].map(len) <= max_len]

        filtered_df = filtered_df.compute()
        filtered_df = filtered_df.sort_values(['Gene_Symbol', 'pXC50']).reset_index(drop=True)
        filtered_df.index.name = 'index'
        
        smiles = filtered_df['SMILES'].unique().to_pandas()
        smiles = smiles[smiles.str.len() <= max_len]

        emb_df = smiles.apply(compute_fp, args=(inferrer, max_len))
        emb_df = pd.DataFrame(emb_df)

        smiles_df = emb_df.merge(smiles, left_index=True, right_index=True)

        # Save for later use.
        logger.info('Saving filtered Excape DB records...')
        logger.info('Saving embedding of SMILES filtered from Excape DB records')
        filtered_df.to_csv(filter_data_file)
        smiles_df.to_csv(embedding_file)

    return filtered_df, smiles_df
    
    
def summarize_excape_data(df_data):
    """
    Summarize the data to produce count of unique Gene_Symbol, Entrez_ID 
    combinations with min and max pXC50 values
    """
    df_data['Cnt'] = cupy.zeros(df_data.shape[0])
    
    # Group by Entrez_ID and compute data for each Entrez_ID
    df_data = df_data.set_index(['Gene_Symbol', 'Entrez_ID'])
    grouped_df = df_data.groupby(level=['Entrez_ID', 'Gene_Symbol'])

    count_df = grouped_df.count()
    count_df.drop('pXC50', axis=1, inplace=True)
    count_df = count_df.sort_values('Cnt', ascending=False)
    count_df['pXC50_min'] = grouped_df.pXC50.min()
    count_df['pXC50_max'] = grouped_df.pXC50.max()
    return count_df()


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
    sample_size = int(cfg.samplingSpec.sample_size)

    if cfg.model.name == 'MegaMolBART':
        inferrer = MegatronMolBART()
    elif cfg.model.name == 'CDDD':
        inferrer = Cddd()
    else:
        logger.error(f'Model {cfg.model.name} not supported')
        sys.exit(1)

    # Metrics
    metric_list = []
    if cfg.metric.validity.enabled == True:
        metric_list.append(Validity(inferrer))

    if cfg.metric.unique.enabled == True:
        metric_list.append(Unique(inferrer))

    if cfg.metric.novelty.enabled == True:
        metric_list.append(Novelty(inferrer))

    if cfg.metric.nearestNeighborCorrelation.enabled == True:
        metric_list.append(NearestNeighborCorrelation(inferrer))

    if cfg.metric.modelability.enabled == True:
        metric_list.append(Modelability(inferrer))

    # ML models
    model_dict = get_model()

    # Create Datasets of size input_size. Initialy load 20% more then reduce to
    # input_size after cleaning and preprocessing.

    smiles_dataset = ZINC15_TestSplit_20K_Samples(max_len=seq_len)
    fingerprint_dataset = ZINC15_TestSplit_20K_Fingerprints()
    smiles_dataset.load()

    # excape_df, excape_emb_df = fetch_filtered_excape_data(inferrer, seq_len)
    # print(excape_df.head())
    # print(excape_emb_df.head())
    # exit(1)

    fingerprint_dataset.load(smiles_dataset.data.index)
    n_data = cfg.samplingSpec.input_size
    if n_data <= 0:
        n_data = len(smiles_dataset.data)
    # assert fingerprint_dataset.data.index == smiles_dataset.data.index

    # DEBUG
    smiles_dataset.data = smiles_dataset.data.iloc[:n_data]
    smiles_dataset.properties = smiles_dataset.properties.iloc[:n_data]
    fingerprint_dataset.data = fingerprint_dataset.data.iloc[:n_data]

    # DEBUG
    n_data = cfg.samplingSpec.input_size

    convert_runtime = lambda x: x.seconds + (x.microseconds / 1.0e6)

    iteration = None
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

            result = metric.calculate(smiles_dataset=smiles_dataset,
                                      fingerprint_dataset=fingerprint_dataset,
                                      top_k=iter_val, # TODO IS THIS A BUG
                                      properties=smiles_dataset.properties,
                                      estimator=estimator,
                                      param_dict=param_dict,
                                      num_samples=sample_size,
                                      radius=iter_val)

            run_time = convert_runtime(datetime.now() - start_time)
            result['iteration'] = iteration
            result['run_time'] = run_time
            result['data_size'] = n_data
            result_list.append(result)
            save_metric_results(result_list, output_dir)


if __name__ == '__main__':
    main()
