import logging
import pickle
import sqlite3
import tempfile
import pandas as pd
import numpy as np
import concurrent.futures
import threading

from contextlib import closing
from cuchembench.utils.smiles import validate_smiles, get_murcko_scaffold

format = '%(asctime)s %(name)s [%(levelname)s]: %(message)s'
logging.basicConfig(level=logging.INFO,
                    filename='/logs/molecule_generator.log',
                    format=format)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(format))
logging.getLogger("").addHandler(console)

log = logging.getLogger('cuchembench.molecule_generator')

__all__ = ['MoleculeGenerator']

lock = threading.Lock()


class MoleculeGenerator():
    def __init__(self, inferrer, db_file=None, batch_size=100, nbits=512) -> None:
        self.db = db_file
        self.inferrer = inferrer
        self.batch_size = batch_size
        self.nbits = nbits
        if self.db is None:
            self.db = tempfile.NamedTemporaryFile(prefix='gsmiles_',
                                                  suffix='.sqlite3',
                                                  dir='/tmp')

        self.conn = sqlite3.connect(self.db, check_same_thread=False)
        result = self.conn.execute('''
            SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?
            ''',
            ['smiles']).fetchone()
        if result[0] == 0:
            with closing(self.conn.cursor()) as cursor:
                sql_file = open("/workspace/benchmark/scripts/generated_smiles_db.sql")
                sql_as_string = sql_file.read()
                cursor.executescript(sql_as_string)
    
    def _insert_generated_smiles(self,
                                 smiles_id,
                                 smiles_df):

        log.info(f'Inserting samples for {smiles_id}...')
        generated_smiles = smiles_df['SMILES'].to_list()
        embeddings = smiles_df['embeddings'].to_list()
        embeddings_dim = smiles_df['embeddings_dim'].to_list()

        lock.acquire(blocking=True, timeout=-1)
        with self.conn as conn:
            with closing(conn.cursor()) as cursor:
                generated = False
                # Replace this loop with pandas to SQLite insert
                for i in range(len(generated_smiles)):
                    gsmiles, is_valid, fp = validate_smiles(generated_smiles[i], return_fingerprint=True, nbits = self.nbits)
                    gscaffold = get_murcko_scaffold(gsmiles)
                    # log.info(f'Scaffold {gscaffold}...')
                    embedding = list(embeddings[i])
                    embedding_dim = list(embeddings_dim[i])

                    embedding = pickle.dumps(embedding)
                    embedding_dim = pickle.dumps(embedding_dim)
                    fp = pickle.dumps(fp)
                    cursor.execute(
                        '''
                        INSERT INTO smiles_samples(input_id, smiles, embedding,
                                                embedding_dim, is_valid,
                                                finger_print, is_generated, scaffold)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        [smiles_id, gsmiles, sqlite3.Binary(embedding),
                        sqlite3.Binary(embedding_dim), is_valid, fp, generated, gscaffold])
                    generated = True # First molecule is always the input

                cursor.execute(
                    'UPDATE smiles set processed = 1 WHERE id = ?',
                    [smiles_id])
        lock.release()

    def generate_and_store(self,
                           csv_data_files,
                           num_requested=10,
                           scaled_radius=1,
                           force_unique=False,
                           sanitize=True,
                           concurrent_requests=4):

        new_sample_db = False
        with closing(self.conn.cursor()) as cursor:
            recs = cursor.execute('''
                SELECT count(*) from smiles s
                WHERE s.model_name = ?
                    AND s.scaled_radius = ?
                    AND s.force_unique = ?
                    AND s.sanitize = ?
                ''',
                [self.inferrer.__class__.__name__, scaled_radius, force_unique, sanitize]).fetchone()
            if recs[0] == 0:
                new_sample_db = True

        if new_sample_db:
            all_dataset_df = []
            for spec_name in csv_data_files:
                spec = csv_data_files[spec_name]
                log.info(f'Reading {spec_name}...')
                smiles_col_name = spec['col_name']
                dataset_type = spec['dataset_type']

                input_size = -1
                if 'input_size' in spec:
                    input_size = spec['input_size']

                # Input must be a CSV file or dataframe. Anything else will fail.
                if isinstance(spec['dataset'], str):
                    # If input dataset is a csv file
                    if input_size > 0:
                        file_df = pd.read_csv(spec['dataset'], nrows=input_size)
                    else:
                        file_df = pd.read_csv(spec['dataset'])
                else:
                    # If input dataset is a dataframe
                    file_df = spec['dataset']

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

        while True:
            df = pd.read_sql_query('''
                SELECT id, smiles
                FROM smiles
                WHERE processed = 0 AND scaled_radius = ? AND num_samples = ? AND dataset_type = 'SAMPLE' LIMIT ?
                ''',
                self.conn, params = [scaled_radius, num_requested, self.batch_size])
            if df.shape[0] == 0:
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = {executor.submit(self._sample, df['id'].tolist(), df['smiles'].tolist(), num_requested, scaled_radius, force_unique, sanitize)}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        log.warning(f'generated an exception: {exc}')
                        log.exception(exc)
        while True:
            df = pd.read_sql_query('''
                SELECT id, smiles
                FROM smiles
                WHERE processed = 0 AND dataset_type = 'EMBEDDING' LIMIT ?
                ''',
                self.conn, params = [self.batch_size])
            if df.shape[0] == 0:
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = {executor.submit(self._sample, df['id'].tolist(), df['smiles'].tolist(), num_requested, scaled_radius, force_unique, sanitize, dataset_type = "EMBEDDING")}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        log.exception(exc)

    def _sample(self, ids, smiles, num_samples, scaled_radius, force_unique, sanitize, dataset_type = "SAMPLE"):
        if dataset_type == 'SAMPLE':
            if len(smiles) == 1: # for CDDD and legacy
                smiles = smiles[0]
            results = self.inferrer.find_similars_smiles(smiles,
                                                        num_requested=num_samples,
                                                        scaled_radius=scaled_radius,
                                                        force_unique=(force_unique == 1),
                                                        sanitize=(sanitize == 1))
            if isinstance(smiles, str):
                results = [results]
        else:
            if len(smiles) == 1: # for CDDD and legacy
                smiles = smiles[0]
                emb = self.inferrer.smiles_to_embedding(smiles)
                result = pd.DataFrame()
                result['SMILES'] = [smiles]
                result['embeddings_dim'] = [emb.dim]
                result['embeddings'] = [emb.embedding]
                results = [result]
            else:
                _, _, embedding_list = self.inferrer.smiles_to_embedding(smiles)
                results = []
                for idx in range(len(smiles)):
                    result = pd.DataFrame()
                    result['SMILES'] = [smiles[idx]]
                    result['embeddings'] = [embedding_list[idx].embedding]
                    result['embeddings_dim'] = [embedding_list[idx].dim]
                    results.append(result)
        
        for id, result in zip(ids, results):
            self._insert_generated_smiles(id, result)
