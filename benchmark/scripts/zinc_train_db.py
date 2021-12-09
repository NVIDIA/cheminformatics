#! /usr/bin/env python3
import argparse
import os
import sys
import csv
import logging
from dask import dataframe as dd
from dask.distributed import LocalCluster, Client
import sqlite3 as lite
from rdkit import Chem
import glob

logger = logging.getLogger('zinc_train_db')
logging.basicConfig(level=logging.INFO)

def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        cannonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    except:
        cannonical_smiles = smiles
    return cannonical_smiles


def upload(path, db_name, n_workers, threads_per_worker, canonicalize=True):
    logger.info(f'Loading data from {path}...')
    filelist = sorted(glob.glob(path))
    assert len(filelist) >= 1, AssertionError(f'No CSV files were found at {path}')
    logger.info(f'Found {len(filelist)} files ...')

    db = f'sqlite:///data/db/{db_name}.sqlite3'
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster, asynchronous=True)

    for pos, file in enumerate(filelist):
        logger.info(f'Processing file {file}...')
        zinc_data = dd.read_csv(file)

        # Canonicalize SMILES
        if canonicalize:
            canonical_zinc_data = zinc_data['smiles'].apply(canonicalize_smiles, meta=('smiles', 'object'))
            zinc_data = zinc_data.drop('smiles', axis=1)
            zinc_data['smiles'] = canonical_zinc_data

        if_exists = 'replace' if pos==0 else 'append'
        zinc_data.to_sql('train_data', db, if_exists=if_exists)


def parse_args():
    parser = argparse.ArgumentParser(description='Load Training Data')
    parser.add_argument('-z', '--zinc_data_path',
                        dest='zinc_data_path',
                        type=str,
                        default='/data/zinc_csv_split/train/*.csv',
                        help='Wildcard path to the CSV files containing ZINC15 (only) training data')
    parser.add_argument('-c', '--cddd_data_path',
                        dest='cddd_data_path',
                        type=str,
                        default='/data/cddd_data/*.csv',
                        help='Wildcard path to the CSV file containing CDDD training data')
    parser.add_argument('-w', '--workers',
                        dest='workers',
                        type=int,
                        default=4
                        )
    parser.add_argument('-t', '--threads_per_worker',
                        dest='threads_per_worker',
                        type=int,
                        default=2
                        )
    parser.add_argument('-n', '--no_canonicalize',
                        dest='canonicalize',
                        action='store_false',
                        )

    args = parser.parse_args(sys.argv[1:])
    return args


if __name__ == '__main__':

    args = parse_args()
    logger.info(f'Using arguments {args}...')

    logger.info('Loading CDDD training data ...')
    # upload(path=args.cddd_data_path, db_name='cddd_train', n_workers=args.workers, threads_per_worker=args.threads_per_worker, canonicalize=args.canonicalize)

    logger.info('Loading ZINC15 training data ...')
    upload(path=args.zinc_data_path, db_name='zinc_train', n_workers=args.workers, threads_per_worker=args.threads_per_worker, canonicalize=args.canonicalize)
