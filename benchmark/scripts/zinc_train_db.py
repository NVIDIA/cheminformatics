#! /usr/bin/env python3
import argparse
import glob
import sys
import logging
import sqlite3
from contextlib import closing

from cuchembench.utils.smiles import get_murcko_scaffold
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


def upload(path, db_name, n_workers, threads_per_worker, canonicalize=True, construct_scaffold=True):
    logger.info(f'Loading data from {path}...')
    paths = list(sorted(glob.glob(path + 'x0[0-6][0-9]*')) + sorted(glob.glob(path + 'x07[0-2]*')))
    logger.info(f'Loading GLOB data from {paths}...')
    db = f'sqlite:////data/db/{db_name}.sqlite3' 
    with closing (LocalCluster(n_workers=n_workers,
                               threads_per_worker=threads_per_worker)) as cluster, cluster,\
        closing(Client(cluster, asynchronous=True)) as client:

        zinc_data = dd.read_csv(paths)
        # Canonicalize SMILES
        if canonicalize:
            canonical_zinc_data = zinc_data['smiles'].apply(canonicalize_smiles,
                                                            meta=('smiles', 'object'))
            zinc_data = zinc_data.drop('smiles', axis=1)
            zinc_data['smiles'] = canonical_zinc_data

        if construct_scaffold:
            scaffolds = zinc_data['smiles'].apply(get_murcko_scaffold, meta=('smiles', 'object'))
            zinc_data['scaffold'] = scaffolds  

        zinc_data.to_sql('train_data', db)

    with closing(sqlite3.connect(f'/data/db/{db_name}.sqlite3', uri=True)) as con, con, \
                closing(con.cursor()) as cur:
        cur.execute('CREATE INDEX smiles_idx ON train_data(smiles)')
        cur.execute('''
                    CREATE TABLE IF NOT EXISTS scaffolds (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scaffold TEXT NOT NULL,
                        UNIQUE(scaffold)
                    );
                    ''')
        cur.execute('''
                    INSERT INTO scaffolds (scaffold)
                        Select distinct scaffold from train_data;
                    ''')


def parse_args():
    parser = argparse.ArgumentParser(description='Load Training Data')
    parser.add_argument('-p', '--data_path',
                        dest='data_path',
                        type=str,
                        default='/data/zinc_csv/train/', #x0[00-72]*
                        help='Wildcard path to the CSV files containing training data')
    parser.add_argument('-d', '--db_name',
                        dest='db_name',
                        type=str,
                        required=False,
                        default='zinc_train_half',
                        choices=['cddd_train', 'zinc_train', 'zinc_train_half'],
                        help='Database name')
    parser.add_argument('-w', '--workers',
                        dest='workers',
                        type=int,
                        default=24
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
    parser.add_argument('-ns', '--no_scaffold',
                        dest='construct_scaffold',
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
