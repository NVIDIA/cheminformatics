import os
import logging
import pickle
import sqlite3
import tempfile
import pandas as pd
import numpy as np
import concurrent.futures

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
    def __init__(self, inferrer, db_file=None) -> None:
        self.db = db_file
        self.inferrer = inferrer

        if self.db is None:
            self.db = tempfile.NamedTemporaryFile(prefix='gsmiles_',
                                                  suffix='.sqlite3',
                                                  dir='/tmp')
        execute_db_creation = False
        if not os.path.exists(self.db):
            execute_db_creation = True

        self.conn = sqlite3.connect(self.db, check_same_thread=False)
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
                           sanitize=True,
                           concurrent_requests=4):

        new_sample_db = False
        with closing(self.conn.cursor()) as cursor:
            recs = cursor.execute('SELECT count(*) from smiles').fetchone()
            if recs[0] == 0:
                new_sample_db = True

        if new_sample_db:
            all_dataset_df = []
            for csv_data_file in csv_data_files:
                smiles_col_name = csv_data_files[csv_data_file]['col_name']
                dataset_type = csv_data_files[csv_data_file]['dataset_type']

                input_size = -1
                if 'input_size' in csv_data_files[csv_data_file]:
                    input_size = csv_data_files[csv_data_file]['input_size']

                if input_size > 0:
                    file_df = pd.read_csv(csv_data_file, nrows=input_size)
                else:
                    file_df = pd.read_csv(csv_data_file)

                dataset_df = pd.DataFrame()
                dataset_df['smiles'] = file_df[smiles_col_name]
                dataset_df['dataset_type'] = np.full(shape=dataset_df.shape[0], fill_value=dataset_type)
                del file_df

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            while True:
                df = pd.read_sql_query(
                    'SELECT * from smiles where processed = 0 LIMIT 1000',
                    self.conn)
                if df.shape[0] == 0:
                    break

                futures = {executor.submit(self._sample, row): \
                    row for row in df.itertuples()}
                for future in concurrent.futures.as_completed(futures):
                    smiles = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        log.warning(f'{smiles} generated an exception: {exc}')

    def _sample(self, row):
        result = self.inferrer.find_similars_smiles(row.smiles,
                                                    num_requested=row.num_samples,
                                                    scaled_radius=row.scaled_radius,
                                                    force_unique=row.force_unique,
                                                    sanitize=row.sanitize)
        self._insert_generated_smileprocessed(row.id, result)
