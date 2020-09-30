#!/opt/conda/envs/rapids/bin/python3
#
# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
import os, sys, wget, gzip
import hashlib
import logging

from dask.distributed import Client, LocalCluster
import dask_cudf
import dask.bag as db

import cudf, cuml
from cuml import KMeans, UMAP

import pandas as pd
import numpy as np

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

def np2cudf(df):
    # convert numpy array to cuDF dataframe
    df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
    pdf = cudf.DataFrame()
    for c,column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf


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

    # ensure we have data
    dl_chemreps()

    smiles_list = []
    chemblID_list = []
    count=1
    max=10000

    chemreps_local_path = '/data/chembl_26_chemreps.txt.gz'
    with gzip.open(chemreps_local_path, 'rb') as fp:
        fp.__next__()
        for i,line in enumerate(fp):
            fields = line.split()
            chemblID_list.append(fields[0].decode("utf-8"))
            smiles_list.append(fields[1].decode("utf-8"))
            count+=1
            if count>max:
                break

    logger.info('Initializing Morgan fingerprints...')
    results = db.from_sequence(smiles_list).map(MorganFromSmiles).compute()

    np_array_fingerprints = np.stack(results).astype(np.float32)

    # take np.array shape (n_mols, nBits) for GPU DataFrame
    gdf = np2cudf(np_array_fingerprints)

    # prepare one set of clusters
    n_clusters = 7
    kmeans_float = KMeans(n_clusters=n_clusters)
    kmeans_float.fit(gdf)
    
    # UMAP
    umap = UMAP(n_neighbors=100,
                a=1.0,
                b=1.0,
                learning_rate=1.0)
    Xt = umap.fit_transform(gdf)
    gdf.add_column('x', Xt[0].to_array())
    gdf.add_column('y', Xt[1].to_array())

    gdf.add_column('cluster', kmeans_float.labels_)

    # start dash
    v = chemvisualize.ChemVisualization(
        gdf.copy(), n_clusters, chemblID_list)

    logger.info('navigate to https://localhost:5000')
    v.start('0.0.0.0')

