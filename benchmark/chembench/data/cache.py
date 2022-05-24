import os
import logging
import pickle
import sqlite3
import tempfile
import pandas as pd
import numpy as np
import concurrent.futures
import threading

from contextlib import closing
from chembench.utils.smiles import validate_smiles, get_murcko_scaffold

format = '%(asctime)s %(name)s [%(levelname)s]: %(message)s'
logging.basicConfig(level=logging.INFO,
                    filename='/logs/molecule_generator.log',
                    format=format)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(format))
logging.getLogger("").addHandler(console)

log = logging.getLogger('cuchembench.molecule_generator')

__all__ = ['DatasetCacheGenerator']

lock = threading.Lock()


class DatasetCacheGenerator():
    def __init__(self, inferrer, db_file=None, batch_size=100, nbits=512) -> None:
        self.db = db_file
        self.inferrer = inferrer
        self.batch_size = batch_size
        self.nbits = nbits
        if self.db is None:
            self.db = tempfile.NamedTemporaryFile(prefix='gsmiles_',
                                                  suffix='.sqlite3',
                                                  dir='/tmp')

        os.makedirs(os.path.dirname(self.db), exist_ok=True)
        self.conn = sqlite3.connect(self.db, check_same_thread=False)
        result = self.conn.execute('''
            SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?
            ''',
            ['smiles']).fetchone()

        if result[0] == 0:
            with closing(self.conn.cursor()) as cursor:
                # @(dreidenbach) changed to my workspace path
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

    def _sample(self, ids, smis, num_samples, scaled_radius, dataset_type="SAMPLE"):
        if dataset_type == 'SAMPLE':
            results = self.inferrer.find_similars_smiles(smis,
                                                         num_requested=num_samples,
                                                         scaled_radius=scaled_radius)
            if isinstance(smis, str):
                results = [results]
        else:
            #TODO: Scaffolding for insert
            if len(smis) == 1: # for CDDD and legacy
                smis = smis[0]
                emb = self.inferrer.smiles_to_embedding(smis)
                # import pdb; pdb.set_trace()
                result = pd.DataFrame()
                result['SMILES'] = [smis]
                result['embeddings_dim'] = [emb.dim]
                result['embeddings'] = [emb.embedding]
                results = [result]
            else:
                _, _, embedding_list = self.inferrer.smiles_to_embedding(smis)
                results = []
                for idx in range(len(smis)):
                    result = pd.DataFrame()
                    result['SMILES'] = [smis[idx]]
                    result['embeddings'] = [embedding_list[idx].embedding]
                    result['embeddings_dim'] = [embedding_list[idx].dim]
                    results.append(result)

        for id, result in zip(ids, results):
            self._insert_generated_smiles(id, result)

    def initialize_db(self,
                      dataset,
                      num_requested=10):
        log.info(f'Creating recs for dataset {dataset}...')

        dataset_type = dataset.type
        if dataset_type == 'EMBEDDING':
            num_requested = 0;

        if dataset.input_size:
            df_file = pd.read_csv(dataset.file,
                                  nrows=dataset.input_size)
        else:
            df_file = pd.read_csv(dataset.file)
        sr_smiles = df_file[dataset.smiles_column_name].drop_duplicates()

        radius = dataset.radius if hasattr(dataset, 'radius') else [0]
        model_name = self.inferrer.__class__.__name__
        for radii in radius:
            dataset_df = pd.DataFrame()
            dataset_df['smiles'] = sr_smiles
            dataset_df['model_name'] = np.full(shape=sr_smiles.shape[0], fill_value=model_name)
            dataset_df['num_samples'] = np.full(shape=sr_smiles.shape[0], fill_value=num_requested)
            dataset_df['scaled_radius'] = np.full(shape=sr_smiles.shape[0],
                                                  fill_value = 0 if dataset_type == 'EMBEDDING' else radii)
            dataset_df['dataset_type'] = np.full(shape=sr_smiles.shape[0], fill_value=dataset_type)

            dataset_df.to_sql('smiles_tmp', self.conn, index=False, if_exists='append')

            if dataset_type == 'EMBEDDING':
                self.conn.executescript(f'''
                    INSERT INTO smiles
                    (smiles, model_name, num_samples, scaled_radius, dataset_type)
                        SELECT *
                        FROM smiles_tmp
                        WHERE smiles_tmp.smiles not in (
                            SELECT smiles
                            FROM smiles
                            WHERE model_name = '{model_name}');

                    DROP TABLE smiles_tmp;
                    ''')
            else:
                self.conn.executescript(f'''

                    INSERT INTO smiles
                    (smiles, model_name, num_samples, scaled_radius, dataset_type)
                        SELECT *
                        FROM smiles_tmp
                        WHERE smiles_tmp.smiles not in (
                            SELECT smiles
                            FROM smiles
                            WHERE model_name = '{model_name}'
                                AND scaled_radius = {radii});

                    UPDATE smiles
                    SET num_samples = {num_requested},
                        dataset_type = 'SAMPLE',
                        scaled_radius = {radii}
                    WHERE model_name = '{model_name}'
                        AND dataset_type = 'EMBEDDING'
                        AND smiles in (select smiles from smiles_tmp);

                    DROP TABLE smiles_tmp;
                    ''')

            del dataset_df

    def sample(self, concurrent_requests=None):

        recs = self.conn.execute('''
            SELECT dataset_type, num_samples, scaled_radius, count(*)
            FROM smiles
            GROUP BY dataset_type, num_samples, scaled_radius
            ORDER BY dataset_type DESC, scaled_radius
            ''').fetchall()

        for rec in recs:
            dataset_type = rec[0]
            num_requested = rec[1]
            scaled_radius = rec[2]

            log.info(f'Processing {dataset_type} with {num_requested} samples and radius {scaled_radius} for {rec[3]} recs...')

            while True:
                df = pd.read_sql_query('''
                    SELECT id, smiles
                    FROM smiles
                    WHERE processed = 0 and dataset_type = ? LIMIT ?
                    ''',
                    self.conn, params = [dataset_type, self.batch_size])
                if df.shape[0] == 0:
                    break

                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                    futures = {executor.submit(self._sample,
                                               df['id'].tolist(),
                                               df['smiles'].tolist(),
                                               num_requested,
                                               scaled_radius)}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()
                        except Exception as exc:
                            log.exception(exc)
