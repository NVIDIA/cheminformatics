import os
import pickle
import sqlite3
import logging

from typing import List

from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.context import Context

__all__ = ['SampleCacheData']

logger = logging.getLogger(__name__)


class SampleCacheData(object, metaclass=Singleton):

    def __init__(self, db_file=None):

        context = Context()
        if db_file is None:
            db_file = context.get_config('data_mount_path', default='/data')
            db_file = os.path.join(db_file, 'db/embedding_cache.sqlite3')

        logger.info(f'Embedding cache database {db_file}...')
        self.conn = sqlite3.connect(db_file, check_same_thread=False)

        cursor = self.conn.cursor()

        sql_file = open("/workspace/benchmark/scripts/embedding_cache.sql")
        sql_as_string = sql_file.read()
        cursor.executescript(sql_as_string)

    def insert_sampling_data(self,
                             model_name,
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
            INSERT INTO smiles(model_name, smiles, num_samples, scaled_radius,
                                force_unique, sanitize)
            VALUES(?, ?,?,?,?,?)
            ''',
            [model_name, smiles, num_samples, scaled_radius, force_unique, sanitize]).lastrowid

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
                            model_name,
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
            WHERE model_name=?
                  AND smiles=?
                  AND num_samples=?
                  AND scaled_radius=?
                  AND force_unique=?
                  AND sanitize=?
            ''',
            [model_name, smiles, num_samples, scaled_radius, force_unique, sanitize])
        id = cursor.fetchone()

        if not id:
            return None

        cursor.execute('SELECT smiles FROM smiles_samples WHERE input_id=?',
                       [id[0]])
        generated_smiles = cursor.fetchall()
        generated_smiles = [x[0] for x in generated_smiles]
        return generated_smiles

    def fetch_n_sampling_data(self,
                              model_name,
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
            SELECT smiles, embedding, embedding_dim
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
        logger.debug('Fetching benchmark data...')
        cursor = self.conn.cursor()
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
