import os
import pickle
import sqlite3
import logging

from typing import List

from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.context import Context

logger = logging.getLogger(__name__)


class TrainingData(object, metaclass=Singleton):

    def __init__(self):

        context = Context()
        db_file = context.get_config('data_mount_path', default='/data')
        db_file = os.path.join(db_file, 'db/zinc_train.sqlite3')

        logger.info(f'Benchmark database {db_file}...')
        self.conn = sqlite3.connect(db_file)

    def is_known_smiles(self, smiles: str) -> bool:
        """
        Checks if the given SMILES is known.
        :param data:
        :return:
        """
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT smiles FROM train_data
            WHERE smiles=?
            ''',
            [smiles])
        id = cursor.fetchone()
        cursor.close()
        return True if id else False


class BenchmarkData(object, metaclass=Singleton):

    def __init__(self):

        context = Context()
        db_file = context.get_config('data_mount_path', default='/data')
        db_file = os.path.join(db_file, 'db/benchmark.sqlite3')

        logger.info(f'Benchmark database {db_file}...')
        self.conn = sqlite3.connect(db_file)

        cursor = self.conn.cursor()

        sql_file = open("/workspace/cuchem/benchmark/scripts/benchmark.sql")
        sql_as_string = sql_file.read()
        cursor.executescript(sql_as_string)


    def insert_sampling_data(self,
                             smiles,
                             num_samples,
                             scaled_radius,
                             force_unique,
                             sanitize,
                             generated_smiles: List[str],
                             embeddings: List,
                             embeddings_dim: List):
        """
        Inserts a list of dicts into the benchmark data table.
        :param data:
        :return:
        """
        logger.debug('Inserting benchmark data...')
        cursor = self.conn.cursor()
        id = cursor.execute(
            '''
            INSERT INTO smiles(smiles, num_samples, scaled_radius,
                                force_unique, sanitize)
            VALUES(?,?,?,?,?)
            ''',
            [smiles, num_samples, scaled_radius, force_unique, sanitize]).lastrowid

        for i in range(len(generated_smiles)):
            gsmiles = generated_smiles[i]
            embedding = list(embeddings[i])
            embedding_dim = list(embeddings_dim[i])

            embedding = pickle.dumps(embedding)
            embedding_dim = pickle.dumps(embedding_dim)
            cursor.execute(
                '''
                INSERT INTO smiles_samples(input_id, smiles, embedding, embedding_dim)
                VALUES(?, ?, ?, ?)
                ''', [id, gsmiles, sqlite3.Binary(embedding), sqlite3.Binary(embedding_dim)])
        self.conn.commit()


    def fetch_sampling_data(self,
                            smiles,
                            num_samples,
                            scaled_radius,
                            force_unique,
                            sanitize):
        """
        Fetch the benchmark data for a given set of parameters.
        :param data:
        :return:
        """
        logger.debug('Fetching benchmark data...')
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT id FROM smiles
            WHERE smiles=?
                  AND num_samples=?
                  AND scaled_radius=?
                  AND force_unique=?
                  AND sanitize=?
            ''',
            [smiles, num_samples, scaled_radius, force_unique, sanitize])
        id = cursor.fetchone()

        if not id:
            return None

        cursor.execute('SELECT smiles FROM smiles_samples WHERE input_id=?',
                       [id[0]])
        generated_smiles = cursor.fetchall()
        generated_smiles = [x[0] for x in generated_smiles]
        return generated_smiles

    def fetch_n_sampling_data(self,
                              smiles,
                              num_samples,
                              scaled_radius,
                              force_unique,
                              sanitize):
        """
        Fetch the benchmark data for a given set of parameters.
        :param data:
        :return:
        """
        logger.debug('Fetching benchmark data...')
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT id FROM smiles
            WHERE smiles=?
                  AND scaled_radius=?
                  AND force_unique=?
                  AND sanitize=?
            ''',
            [smiles, scaled_radius, force_unique, sanitize])
        id = cursor.fetchone()

        if not id:
            return None

        cursor.execute(
            '''
            SELECT smiles, embedding, embedding_dim
            FROM smiles_samples WHERE input_id=?
            LIMIT ?
            ''',
            [id[0], num_samples])
        generated_smiles = cursor.fetchall()
        # generated_smiles = [x for x in generated_smiles]

        return generated_smiles
