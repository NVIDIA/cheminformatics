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
        self._process_pending = False

        execute_db_creation = False
        if not os.path.exists(self.db):
            execute_db_creation = True

        self.conn = sqlite3.connect(self.db)
        if execute_db_creation:
            with closing(self.conn.cursor()) as cursor:
                sql_file = open("/workspace/benchmark/scripts/generated_smiles_db.sql")
                sql_as_string = sql_file.read()
                cursor.executescript(sql_as_string)

        with closing(self.conn.cursor()) as cursor:
            pending_recs = cursor.execute(
                'SELECT count(*) from smiles where processed = 0').fetchone()[0]
            if pending_recs > 0:
                self._process_pending = True

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

    def generate_and_store(self,
                           csv_data_file,
                           smiles_col_name,
                           num_requested=10,
                           scaled_radius=1,
                           force_unique=False,
                           sanitize=True):

        if self._process_pending is False:
            input_df = pd.read_csv(csv_data_file)
            df = pd.DataFrame()
            df['smiles'] = input_df[smiles_col_name]
            del input_df

            df['model_name'] = np.full(shape=df.shape[0], fill_value=self.inferrer.__class__.__name__)
            df['num_samples'] = np.full(shape=df.shape[0], fill_value=num_requested)
            df['scaled_radius'] = np.full(shape=df.shape[0], fill_value=scaled_radius)
            df['force_unique'] = np.full(shape=df.shape[0], fill_value=force_unique)
            df['sanitize'] = np.full(shape=df.shape[0], fill_value=sanitize)
            df.to_sql('smiles', self.conn, index=False, if_exists='append')
            del df


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

    def fetch_samples(self,
                      model_name,
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
                SELECT id FROM smiles
                WHERE model_name=?
                    AND smiles=?
                    AND scaled_radius=?
                    AND force_unique=?
                    AND sanitize=?
                ''',
                [model_name, smiles, scaled_radius, force_unique, sanitize])
            id = cursor.fetchone()

            if not id:
                return None

            cursor.execute(
                '''
                SELECT smiles, embedding, embedding_dim, is_valid
                FROM smiles_samples WHERE input_id=? LIMIT ?
                ''',
                [id[0], num_samples])
            generated_smiles = cursor.fetchall()

        return generated_smiles

    def fetch_embedding(self,
                        model_name,
                        smiles):
        """
        Fetch the embedding of a given SMILES string.
        """
        log.debug('Fetching benchmark data...')
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(
                '''
                SELECT id FROM smiles
                WHERE model_name=?
                    AND smiles=?
                ''',
                [model_name, smiles])
            id = cursor.fetchone()

            if not id:
                return None

            cursor.execute(
                '''
                SELECT smiles, embedding, embedding_dim
                FROM smiles_samples WHERE input_id=?
                LIMIT ?
                ''',
                [id[0], 1])
            return cursor.fetchone()


inferrer = MegaMolBARTWrapper()
generator = MoleculeGenerator(inferrer)
generator.generate_and_store('/workspace/benchmark/cuchembm/csv_data/benchmark_ZINC15_test_split.csv',
                             'canonical_smiles',
                             num_requested=10,
                             scaled_radius=1,
                             force_unique=False,
                             sanitize=True)