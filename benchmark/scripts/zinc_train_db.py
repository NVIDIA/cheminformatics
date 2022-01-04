#! /usr/bin/env python3
import argparse
import sys
import logging
import sqlite3
from contextlib import closing

from dask import dataframe as dd
from dask.distributed import LocalCluster, Client
from rdkit import Chem

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
    db = f'sqlite:////data/db/{db_name}.sqlite3'
    with closing (LocalCluster(n_workers=n_workers,
                               threads_per_worker=threads_per_worker)) as cluster, cluster,\
        closing(Client(cluster, asynchronous=True)) as client:

        zinc_data = dd.read_csv(path)
        # Canonicalize SMILES
        if canonicalize:
            canonical_zinc_data = zinc_data['smiles'].apply(canonicalize_smiles,
                                                            meta=('smiles', 'object'))
            zinc_data = zinc_data.drop('smiles', axis=1)
            zinc_data['smiles'] = canonical_zinc_data

        zinc_data.to_sql('train_data', db)

    with closing(sqlite3.connect(db, uri=True)) as con, con, \
                closing(con.cursor()) as cur:
        cur.execute('CREATE INDEX smiles_idx ON train_data(smiles)')


def parse_args():
    parser = argparse.ArgumentParser(description='Load Training Data')
    parser.add_argument('-p', '--data_path',
                        dest='data_path',
                        type=str,
                        default='/data/cddd_data/*.csv',
                        help='Wildcard path to the CSV files containing training data')
    parser.add_argument('-d', '--db_name',
                        dest='db_name',
                        type=str,
                        required=True,
                        choices=['cddd_train', 'zinc_train'],
                        help='Database name')
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

    logger.info(f'Loading {args.db_name} training data ...')
    upload(path=args.data_path,
           db_name=args.db_name,
           n_workers=args.workers,
           threads_per_worker=args.threads_per_worker,
           canonicalize=args.canonicalize)
