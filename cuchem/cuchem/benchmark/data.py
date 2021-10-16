import os
import pickle
import sqlite3
import logging

from typing import List

from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.context import Context
from cuchem.datasets.molecules import PHYSCHEM_TABLE_LIST
from cuchem.datasets.bioactivity import BIOACTIVITY_TABLE_LIST

logger = logging.getLogger(__name__)

class ZINC15TrainData(object, metaclass=Singleton):

    def __init__(self):
        """Store training split from ZINC15 for calculation of novelty"""

        context = Context()
        db_file = context.get_config('data_mount_path', default='/data')
        db_file = os.path.join(db_file, 'db', 'zinc_train.sqlite3')

        db_url = f'file:{db_file}?mode=ro'
        logger.info(f'Train database {db_url}...')
        self.conn = sqlite3.connect(db_url, uri=True)

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


class ZINC15TestSamplingData(object, metaclass=Singleton):

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
        # generated_smiles = [x for x in generated_smiles]

        return generated_smiles


class PhysChemEmbeddingData(object, metaclass=Singleton):
    # TODO RAJESH there is a bug upon retriving data from the SQL databases -- the dimensions are not the same as those input

    def __init__(self):

        context = Context()
        db_file = context.get_config('data_mount_path', default='/data')
        db_file = os.path.join(db_file, 'db/physchem.sqlite3')

        logger.info(f'Physchem properties database {db_file}...')
        self.conn = sqlite3.connect(db_file)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        for table_name in PHYSCHEM_TABLE_LIST:
            table_creation = '''
                            CREATE TABLE IF NOT EXISTS ''' + table_name + ''' (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            input_id INTEGER NOT NULL,
                            smiles TEXT NOT NULL,
                            model_name TEXT NOT NULL,
                            embedding TEXT NOT NULL,
                            embedding_dim TEXT NOT NULL);
                            '''
            cursor.execute(table_creation)

    def insert_embedding_data(self,
                             table_name,
                             model_name,
                             smiles,
                             smiles_index,
                             embeddings: List,
                             embeddings_dim: List):
        """
        Inserts a list of dicts into the benchmark data table.
        :param data:
        :return:
        """
        cursor = self.conn.cursor()
        
        # Add embedding
        logger.debug('Inserting benchmark data...')
        embedding = list(embeddings)
        embedding = pickle.dumps(embedding)

        embedding_dim = list(embeddings_dim)
        embedding_dim = pickle.dumps(embedding_dim)
        
        id = cursor.execute(
            '''
            INSERT INTO ''' + table_name + '''(input_id, smiles, model_name, embedding, embedding_dim)
            VALUES(?,?,?,?,?)
            ''',
            [smiles_index, smiles, model_name, sqlite3.Binary(embedding), sqlite3.Binary(embedding_dim)])
        self.conn.commit()

    def fetch_embedding_data(self,
                            table_name,
                            model_name,
                            smiles):
        """
        Fetch the embedding data for a given dataset and smiles
        :param data:
        :return:
        """

        logger.debug('Fetching embedding data...')

        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT embedding, embedding_dim FROM ''' + table_name + '''
            WHERE model_name=?
                  AND smiles=?
            ''',
            [model_name, smiles])
        embedding_results = cursor.fetchone()

        if not embedding_results:
            return None

        return embedding_results


class BioActivityEmbeddingData(object, metaclass=Singleton):
    # TODO RAJESH this needs testing

    def __init__(self):

        context = Context()
        db_file = context.get_config('data_mount_path', default='/data')
        db_file = os.path.join(db_file, 'db/bioactivity.sqlite3')

        logger.info(f'Bioactivities database {db_file}...')
        self.conn = sqlite3.connect(db_file)
        self._create_tables()
        # TODO how to handle fingerprints -- should they be stored in separate table since there's so much data

    def _create_tables(self):
        cursor = self.conn.cursor()

        # TODO update as appropriate for fingerprints
        for table_name in BIOACTIVITY_TABLE_LIST:
            table_creation = '''
                            CREATE TABLE IF NOT EXISTS ''' + table_name + ''' (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            input_id INTEGER NOT NULL,
                            smiles TEXT NOT NULL,
                            model_name TEXT NOT NULL,
                            embedding TEXT NOT NULL,
                            embedding_dim TEXT NOT NULL);
                            '''
            cursor.execute(table_creation)

    def insert_embedding_data(self,
                             table_name,
                             model_name,
                             smiles,
                             smiles_index,
                             embeddings: List,
                             embeddings_dim: List):
        """
        Inserts a list of dicts into the benchmark data table.
        :param data:
        :return:
        """
        cursor = self.conn.cursor()
        
        # Add embedding
        logger.debug('Inserting benchmark data...')
        embedding = list(embeddings)
        embedding = pickle.dumps(embedding)

        embedding_dim = list(embeddings_dim)
        embedding_dim = pickle.dumps(embedding_dim)
        
        id = cursor.execute(
            '''
            INSERT INTO ''' + table_name + '''(input_id, smiles, model_name, embedding, embedding_dim)
            VALUES(?,?,?,?,?)
            ''',
            [smiles_index, smiles, model_name, sqlite3.Binary(embedding), sqlite3.Binary(embedding_dim)])
        self.conn.commit()

    def fetch_embedding_data(self,
                            table_name,
                            model_name,
                            smiles):
        """
        Fetch the embedding data for a given dataset and smiles
        :param data:
        :return:
        """

        logger.debug('Fetching embedding data...')

        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT embedding, embedding_dim FROM ''' + table_name + '''
            WHERE model_name=?
                  AND smiles=?
            ''',
            [model_name, smiles])
        embedding_results = cursor.fetchone()

        if not embedding_results:
            return None

        return embedding_results


