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


log = logging.getLogger(__name__)

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
                sql_file = open("/workspace/cheminformatics/benchmark/scripts/generated_smiles_db.sql")
                sql_as_string = sql_file.read()
                cursor.executescript(sql_as_string)

    def _insert_sample(self, smi_id, g_smi, emb, cursor, generated):
        smi, is_valid, fp = validate_smiles(g_smi,
                                            return_fingerprint=True,
                                            nbits=self.nbits)
        gscaffold = get_murcko_scaffold(smi)

        dim = pickle.dumps(list(emb.shape))
        emb = pickle.dumps(emb)
        fp = pickle.dumps(fp)
        cursor.execute(
            '''
            INSERT INTO smiles_samples(input_id, smiles, embedding,
                                       embedding_dim, is_valid,
                                       finger_print, is_generated, scaffold)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            [smi_id, smi, sqlite3.Binary(emb), sqlite3.Binary(dim),
            is_valid, fp, generated, gscaffold])

    def _insert_generated_smis(self,
                               smi_id,
                               smi,
                               emb,
                               g_smis=None,
                               g_embs=None):
        log.info(f'Inserting samples for {smi_id}...')
        lock.acquire(blocking=True, timeout=-1)
        with self.conn as conn:
            with closing(conn.cursor()) as cursor:
                self._insert_sample(smi_id, smi, emb, cursor, False)

                if g_smis:
                    for i in range(len(g_smis)):
                        self._insert_sample(smi_id, g_smis[i], g_embs[i], cursor, True)

                cursor.execute('UPDATE smiles set processed = 1 WHERE id = ?', [smi_id])

        lock.release()

    def _sample(self, ids, smis, num_samples, scaled_radius, dataset_type):
        embs = self.inferrer.smis_to_embedding(smis)
        if dataset_type == 'SAMPLE':
            g_smis, g_embs = self.inferrer.sample(smis,
                                                  return_embedding=True,
                                                  num_samples=num_samples,
                                                  sampling_kwarg={"scaled_radius": scaled_radius})
            for i in range(len(ids)):
                start_idx = i * num_samples
                end_idx = start_idx + num_samples
                self._insert_generated_smis(ids[i],
                                            smis[i],
                                            embs[i],
                                            g_smis=g_smis[start_idx: end_idx],
                                            g_embs=g_embs[start_idx: end_idx])
        else:
            for i in range(len(ids)):
                self._insert_generated_smis(ids[i],
                                            smis[i],
                                            embs[i],
                                            g_smis=None,
                                            g_embs=None)

    def initialize_db(self,
                      dataset,
                      radius,
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

        for radii in radius:
            dataset_df = pd.DataFrame()
            dataset_df['smiles'] = sr_smiles
            dataset_df['num_samples'] = np.full(shape=sr_smiles.shape[0], fill_value=num_requested)
            dataset_df['scaled_radius'] = np.full(shape=sr_smiles.shape[0],
                                                  fill_value = 0 if dataset_type == 'EMBEDDING' else radii)
            dataset_df['dataset_type'] = np.full(shape=sr_smiles.shape[0], fill_value=dataset_type)

            dataset_df.to_sql('smiles_tmp', self.conn, index=False, if_exists='append')

            if dataset_type == 'EMBEDDING':
                self.conn.executescript(f'''
                    INSERT INTO smiles
                    (smiles, num_samples, scaled_radius, dataset_type)
                        SELECT *
                        FROM smiles_tmp
                        WHERE smiles_tmp.smiles not in (
                            SELECT smiles
                            FROM smiles);

                    DROP TABLE smiles_tmp;
                    ''')
            else:
                if radii is None:
                    radius_condition = 'scaled_radius is NULL'
                    set_radius = 'scaled_radius = NULL'
                else:
                    radius_condition = f'scaled_radius = {radii}'
                    set_radius = f'scaled_radius = {radii}'

                self.conn.executescript(f'''
                    INSERT INTO smiles
                    (smiles, num_samples, scaled_radius, dataset_type)
                        SELECT *
                        FROM smiles_tmp
                        WHERE smiles_tmp.smiles not in (
                            SELECT smiles
                            FROM smiles
                            WHERE {radius_condition});

                    UPDATE smiles
                    SET num_samples = {num_requested},
                        dataset_type = 'SAMPLE',
                        {set_radius}
                    WHERE dataset_type = 'EMBEDDING'
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
                    WHERE processed = 0 and dataset_type = ?
                    LIMIT ?
                    ''',
                    self.conn, params = [dataset_type, self.batch_size])

                if df.shape[0] == 0:
                    break
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                    futures = {executor.submit(self._sample,
                                               df['id'].tolist(),
                                               df['smiles'].tolist(),
                                               num_requested,
                                               scaled_radius,
                                               dataset_type)}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()
                        except Exception as exc:
                            # log.exception(exc)
                            raise exc
