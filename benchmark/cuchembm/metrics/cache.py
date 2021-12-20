import os
import logging
import pickle
import sqlite3
import pandas as pd
import numpy as np

from contextlib import closing
from cuchembm.inference.megamolbart import MegaMolBARTWrapper
from cuchemcommon.utils.smiles import validate_smiles

format = '%(asctime)s %(name)s [%(levelname)s]: %(message)s'
logging.basicConfig(level=logging.INFO,
                    filename='/logs/molecule_generator.log',
                    format=format)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(format))
logging.getLogger("").addHandler(console)

log = logging.getLogger('cuchembm.molecule_generator')

__all__ = ['MoleculeGenerator']


class MoleculeGenerator():
    def __init__(self, inferrer) -> None:
        self.db = '/data/db/generated_smiles.sqlite3'
        self.inferrer = inferrer

        execute_db_creation = False
        if not os.path.exists(self.db):
            execute_db_creation = True

        self.conn = sqlite3.connect(self.db)
        if execute_db_creation:
            with closing(self.conn.cursor()) as cursor:
                sql_file = open("/workspace/benchmark/scripts/generated_smiles_db.sql")
                sql_as_string = sql_file.read()
                cursor.executescript(sql_as_string)

    def _insert_generated_smiles(self,
                                 smiles_id,
                                 smiles_df):
        log.debug('Inserting benchmark data...')

        generated_smiles = smiles_df['SMILES'].to_list()
        embeddings = smiles_df['embeddings'].to_list()
        embeddings_dim = smiles_df['embeddings_dim'].to_list()

        with closing(self.conn.cursor()) as cursor:
            cursor.execute(
                'UPDATE smiles set processed = 1 WHERE id = ?',
                [smiles_id])

            generated = False
            for i in range(len(generated_smiles)):
                gsmiles, is_valid, fp = validate_smiles(generated_smiles[i],
                                                        return_fingerprint=True)
                embedding = list(embeddings[i])
                embedding_dim = list(embeddings_dim[i])

                embedding = pickle.dumps(embedding)
                embedding_dim = pickle.dumps(embedding_dim)
                fp = pickle.dumps(fp)
                cursor.execute(
                    '''
                    INSERT INTO smiles_samples(input_id, smiles, embedding,
                                               embedding_dim, is_valid,
                                               finger_print, is_generated)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    ''',
                    [smiles_id, gsmiles, sqlite3.Binary(embedding),
                     sqlite3.Binary(embedding_dim), is_valid, fp, generated])
                generated = True

            self.conn.commit()

    def fetch_samples(self,
                      smiles,
                      num_samples,
                      scaled_radius,
                      force_unique,
                      sanitize):
        """
        Fetch the benchmark data for a given set of parameters.
        """
        log.debug('Fetching benchmark data...')
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(
                '''
                SELECT ss.smiles, ss.embedding, ss.embedding_dim, ss.is_valid
                FROM smiles s, smiles_samples ss
                WHERE s.id = ss.input_id
                    AND s.model_name = ?
                    AND s.scaled_radius = ?
                    AND s.force_unique = ?
                    AND s.sanitize = ?
                    AND s.smiles = ?
                LIMIT ?;
                ''',
                [self.inferrer.__class__.__name__, smiles, scaled_radius,
                 force_unique, sanitize, num_samples])
            generated_smiles = cursor.fetchall()

        return generated_smiles

    def fetch_embedding(self, smiles):
        """
        Fetch the embedding of a given SMILES string.
        """
        log.debug('Fetching benchmark data...')
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(
                '''
                SELECT ss.smiles, ss.embedding, ss.embedding_dim
                FROM smiles as s, smiles_samples as ss
                WHERE s.id = ss.input_id
                    AND s.model_name = ?
                    AND s.smiles = s.smiles
                    AND s.smiles = ?
                    LIMIT 1;
                ''',
                [self.inferrer.__class__.__name__, smiles])

            return cursor.fetchone()

    def generate_and_store(self,
                           csv_data_files,
                           num_requested=10,
                           scaled_radius=1,
                           force_unique=False,
                           sanitize=True):

        process_pending = False
        with closing(self.conn.cursor()) as cursor:
            pending_recs = cursor.execute(
                'SELECT count(*) from smiles where processed = 0').fetchone()
            if pending_recs[0] > 0:
                process_pending = True

        if process_pending is False:
            all_dataset_df = []
            for csv_data_file in csv_data_files:
                smiles_col_name = csv_data_files[csv_data_file]['col_name']
                dataset_type = csv_data_files[csv_data_file]['dataset_type']
                dataset_df = pd.DataFrame()
                dataset_df['smiles'] = pd.read_csv(csv_data_file)[smiles_col_name]
                dataset_df['dataset_type'] = np.full(shape=dataset_df.shape[0], fill_value=dataset_type)

                all_dataset_df.append(dataset_df)

            df = pd.DataFrame()
            df = pd.concat(all_dataset_df).drop_duplicates()
            df['model_name'] = np.full(shape=df.shape[0], fill_value=self.inferrer.__class__.__name__)
            df['num_samples'] = np.full(shape=df.shape[0], fill_value=num_requested)
            df['scaled_radius'] = np.full(shape=df.shape[0], fill_value=scaled_radius)
            df['force_unique'] = np.full(shape=df.shape[0], fill_value=force_unique)
            df['sanitize'] = np.full(shape=df.shape[0], fill_value=sanitize)
            df.to_sql('smiles', self.conn, index=False, if_exists='append')
            del df
        else:
            log.warn(f"There are pending records in the database.")
            log.warn(f"Please rerun after this job to process {csv_data_files}.")

        while True:
            df = pd.read_sql_query(
                'SELECT * from smiles where processed = 0 LIMIT 1000',
                self.conn)
            if df.shape[0] == 0:
                break

            for row in df.itertuples ():
                print(row.id, row.smiles)
                log.debug(f'Generating embeddings for {row.smiles}...')
                result = self.inferrer.find_similars_smiles(row.smiles,
                                                            num_requested=row.num_samples,
                                                            scaled_radius=row.scaled_radius,
                                                            force_unique=row.force_unique,
                                                            sanitize=row.sanitize)

                self._insert_generated_smiles(row.id, result)

    def compute_sampling_metrics(self,
                                 scaled_radius=1,
                                 force_unique=False,
                                 sanitize=True,
                                 dataset_type='SAMPLE'):
        """
        Compute the sampling metrics for a given set of parameters.
        """
        # Valid SMILES
        valid_result = self.conn.execute('''
            SELECT ss.is_valid, count(*)
            FROM smiles s, smiles_samples ss
            WHERE s.id = ss.input_id
                AND s.model_name = ?
                AND s.scaled_radius = ?
                AND s.force_unique = ?
                AND s.sanitize = ?
                AND ss.is_generated = 1
                AND s.processed = 1
                AND s.dataset_type = ?
            GROUP BY ss.is_valid;
            ''',
            [self.inferrer.__class__.__name__, scaled_radius,
             force_unique, sanitize, dataset_type])

        total_molecules = 0
        valid_molecules = 0
        for rec in valid_result.fetchall():
            total_molecules += rec[1]
            if rec[0] == 1:
                valid_molecules += rec[1]
        log.info(f'Total molecules: {total_molecules}')
        log.info(f'Valid molecules: {valid_molecules}')

        # Unique SMILES
        unique_result = self.conn.execute(f'''
            SELECT avg(ratio)
            FROM (
                SELECT CAST(count(DISTINCT ss.smiles) as float) / CAST(count(ss.smiles) as float) ratio
            FROM smiles s, smiles_samples ss
            WHERE s.id = ss.input_id
                AND s.model_name = ?
                AND s.scaled_radius = ?
                AND s.force_unique = ?
                AND s.sanitize = ?
                AND ss.is_valid = 1
                AND ss.is_generated = 1
                AND s.processed = 1
                AND s.dataset_type = ?
            GROUP BY s.id
            )''',
            [self.inferrer.__class__.__name__, scaled_radius, force_unique,
             sanitize, dataset_type])
        rec = unique_result.fetchone()
        unique_ratio = rec[0]

        # Novel SMILES
        self.conn.execute('ATTACH ? AS training_db', ['/data/db/zinc_train.sqlite3'])
        res = self.conn.execute('''
            SELECT count(distinct ss.smiles)
            FROM main.smiles s, main.smiles_samples ss, training_db.train_data td
            WHERE ss.smiles = td.smiles
                AND s.id = ss.input_id
                AND s.model_name = ?
                AND s.scaled_radius = ?
                AND s.force_unique = ?
                AND s.sanitize = ?
                AND ss.is_valid = 1
                AND ss.is_generated = 1
                AND s.processed = 1
                AND s.dataset_type = ?
            ''',
            [self.inferrer.__class__.__name__, scaled_radius, force_unique,
             sanitize, dataset_type])
        rec = res.fetchone()
        novel_molecules = valid_molecules - rec[0]
        log.info(f'Novel molecules: {novel_molecules}')

        log.info(f'Validity Ratio: {valid_molecules/total_molecules}')
        log.info(f'Unique Ratio: {unique_ratio}')
        log.info(f'Novelity Ratio: {novel_molecules/valid_molecules}')


data_files = {
    '/workspace/benchmark/cuchembm/csv_data/benchmark_ZINC15_test_split.csv':
        {'col_name': 'canonical_smiles',
         'dataset_type': 'SAMPLE'},
    '/workspace/benchmark/cuchembm/csv_data/benchmark_ChEMBL_approved_drugs_physchem.csv':
        {'col_name': 'canonical_smiles',
         'dataset_type': 'EMBEDDING'},
    '/workspace/benchmark/cuchembm/csv_data/benchmark_MoleculeNet_Lipophilicity.csv':
        {'col_name': 'SMILES',
         'dataset_type': 'EMBEDDING'},
    '/workspace/benchmark/cuchembm/csv_data/benchmark_MoleculeNet_ESOL.csv':
        {'col_name': 'SMILES',
         'dataset_type': 'EMBEDDING'},
    '/workspace/benchmark/cuchembm/csv_data/benchmark_MoleculeNet_FreeSolv.csv':
        {'col_name': 'SMILES',
         'dataset_type': 'EMBEDDING'}
    }

inferrer = MegaMolBARTWrapper()
generator = MoleculeGenerator(inferrer)
generator.generate_and_store(data_files,
                             num_requested=10,
                             scaled_radius=1,
                             force_unique=False,
                             sanitize=True)

import time
start_time = time.time()
generator.compute_sampling_metrics(scaled_radius=1,
                                   force_unique=False,
                                   sanitize=True,
                                   dataset_type='SAMPLE')
log.info("Time(sec): {}".format(time.time() - start_time))