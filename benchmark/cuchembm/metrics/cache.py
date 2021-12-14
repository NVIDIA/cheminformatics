import os
import logging
import pickle
import sqlite3
import pandas as pd

from contextlib import closing
from cuchembm.inference.megamolbart import MegaMolBARTWrapper
from cuchemcommon.utils.smiles import validate_smiles

logging.basicConfig(level=logging.INFO, filename='/logs/molecule_generator.log')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(name)s [%(levelname)s]: %(message)s'))
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
                                 smiles,
                                 smiles_df,
                                 num_requested,
                                 scaled_radius,
                                 force_unique,
                                 sanitize):
        log.debug('Inserting benchmark data...')
        model_name = self.inferrer.__class__.__name__

        generated_smiles = smiles_df['SMILES'].to_list()
        embeddings = smiles_df['embeddings'].to_list()
        embeddings_dim = smiles_df['embeddings_dim'].to_list()

        with closing(self.conn.cursor()) as cursor:
            id = cursor.execute(
                '''
                INSERT INTO smiles(model_name, smiles, num_samples,
                                   scaled_radius, force_unique, sanitize)
                VALUES(?, ?,?,?,?,?)
                ''',
                [model_name, smiles, num_requested,
                 scaled_radius, force_unique, sanitize]).lastrowid

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
                                               embedding_dim, is_valid, finger_print)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ''',
                    [id, gsmiles, sqlite3.Binary(embedding),
                     sqlite3.Binary(embedding_dim), is_valid, fp])
            self.conn.commit()

    def generate_and_store(self,
                           csv_data_file,
                           smiles_col_name,
                           num_requested=10,
                           scaled_radius=1,
                           force_unique=False,
                           sanitize=True):
        df = pd.read_csv(csv_data_file)

        smiles_series = df[smiles_col_name]
        for index, smiles in smiles_series.iteritems():
            log.debug(f'Generating embeddings for {smiles}...')
            result = self.inferrer.find_similars_smiles(smiles,
                                                        num_requested=num_requested,
                                                        scaled_radius=scaled_radius,
                                                        force_unique=force_unique,
                                                        sanitize=sanitize)

            self._insert_generated_smiles(smiles,
                                          result,
                                          num_requested,
                                          scaled_radius,
                                          force_unique,
                                          sanitize)

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
                FROM smiles_samples WHERE input_id=?
                LIMIT ?
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