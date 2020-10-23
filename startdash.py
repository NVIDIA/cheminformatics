#!/opt/conda/envs/rapids/bin/python3
#
# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
import os, wget, gzip
import hashlib
import logging
from datetime import datetime

from dask.distributed import Client, LocalCluster
import dask_cudf
import dask.bag as db

import cudf, cuml

import pandas as pd
import numpy as np

import sklearn.cluster
import sklearn.decomposition
import umap

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils
from PIL import Image

import chemvisualize

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('nvChemViz')
formatter = logging.Formatter(
        '%(asctime)s %(name)s [%(levelname)s]: %(message)s')

###############################################################################
#
# function defs: np2cudf
#
###############################################################################

def np2dataframe(df, enable_gpu):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
    if not enable_gpu:
        return df

    return cudf.DataFrame(df)


def MorganFromSmiles(smiles, radius=2, nBits=512):
    m = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    ar = np.array(fp)
    return ar


def ToNpArray(fingerprints):
    fingerprints = np.asarray(fingerprints, dtype=np.float32)
    return fingerprints


###############################################################################
#
# Download SMILES from FTP
#
###############################################################################

def dl_chemreps(chemreps_local_path='/data/chembl_26_chemreps.txt.gz'):

    chemreps_url = 'ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_26/chembl_26_chemreps.txt.gz'
    chemreps_sha256 = '0585b113959592453c2e1bb6f63f2fc9d5dd34be8f96a3a3b3f80e78d5dbe1bd'
    chemreps_exists_and_is_good = False

    while not chemreps_exists_and_is_good:
        if os.path.exists(chemreps_local_path):
            with open(chemreps_local_path, 'rb') as file:
                local_sha256 = hashlib.sha256(file.read()).hexdigest()
            if chemreps_sha256==local_sha256:
                chemreps_exists_and_is_good = True
                logger.info('chembl chemreps file found locally, SHA256 matches')
        if not chemreps_exists_and_is_good:
            logger.info('downloading chembl chemreps file...')
            wget.download(chemreps_url, chemreps_local_path)


###############################################################################
#
# MAIN
#
###############################################################################

if __name__=='__main__':

    # start dask cluster
    logger.info('Starting dash cluster...')
    cluster = LocalCluster(dashboard_address=':9001', n_workers=12)
    client = Client(cluster)

    enable_gpu = True
    max_molecule = 10000
    pca_components = 64 # Number of PCA components or False to not use PCA

    # ensure we have data
    dl_chemreps()

    smiles_list = []
    chemblID_list = []
    count=1

    chemreps_local_path = '/data/chembl_26_chemreps.txt.gz'
    with gzip.open(chemreps_local_path, 'rb') as fp:
        fp.__next__()
        for i,line in enumerate(fp):
            fields = line.split()
            chemblID_list.append(fields[0].decode("utf-8"))
            smiles_list.append(fields[1].decode("utf-8"))
            count+=1
            if count>max_molecule:
                break

    logger.info('Initializing Morgan fingerprints...')
    results = db.from_sequence(smiles_list).map(MorganFromSmiles).compute()

    np_fingerprints = np.stack(results).astype(np.float32)

    # take np.array shape (n_mols, nBits) for GPU DataFrame
    df_fingerprints = np2dataframe(np_fingerprints, enable_gpu)

    # prepare one set of clusters
    if pca_components:
        task_start_time = datetime.now()
        if enable_gpu:
            pca = cuml.PCA(n_components=pca_components)
        else:
            pca = sklearn.decomposition.PCA(n_components=pca_components)
        
        df_fingerprints = pca.fit_transform(df_fingerprints)
        print('Runtime PCA time (hh:mm:ss.ms) {}'.format(
            datetime.now() - task_start_time))
    else:
        pca = False
        print('PCA has been skipped')
    
    task_start_time = datetime.now()
    n_clusters = 7
    if enable_gpu:
        kmeans_float = cuml.KMeans(n_clusters=n_clusters)
    else:
        kmeans_float = sklearn.cluster.KMeans(n_clusters=n_clusters)
    kmeans_float.fit(df_fingerprints)
    print('Runtime Kmeans time (hh:mm:ss.ms) {}'.format(
        datetime.now() - task_start_time))

    # UMAP
    task_start_time = datetime.now()
    if enable_gpu:
        umap = cuml.UMAP(n_neighbors=100,
                    a=1.0,
                    b=1.0,
                    learning_rate=1.0)
    else:
        umap = umap.UMAP()

    Xt = umap.fit_transform(df_fingerprints)
    print('Runtime UMAP time (hh:mm:ss.ms) {}'.format(
        datetime.now() - task_start_time))

    if enable_gpu:
        df_fingerprints.add_column('x', Xt[0].to_array())
        df_fingerprints.add_column('y', Xt[1].to_array())
        df_fingerprints.add_column('cluster', kmeans_float.labels_)
    else:
        df_fingerprints['x'] = Xt[:,0]
        df_fingerprints['y'] = Xt[:,1]
        df_fingerprints['cluster'] = kmeans_float.labels_

    # start dash
    v = chemvisualize.ChemVisualization(
        df_fingerprints.copy(), n_clusters, chemblID_list,
        enable_gpu=enable_gpu, pca_model=pca)

    logger.info('navigate to https://localhost:5000')
    v.start('0.0.0.0')
