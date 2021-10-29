import os
import pickle
import sqlite3
import logging
from typing import List
from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.context import Context


logger = logging.getLogger(__name__)

__all__ = ['ChEMBLApprovedDrugsEmbeddingData', 'PhysChemEmbeddingData', 'BioActivityEmbeddingData']

class EmbeddingData(object, metaclass=Singleton):

    def __init__(self, sql_path):

        context = Context()
        db_file = context.get_config('data_mount_path', default='/data')
        db_file = os.path.join(db_file, sql_path)
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        table_creation = '''
                        CREATE TABLE IF NOT EXISTS emb (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        smiles TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        embedding_dim TEXT NOT NULL);
                        '''
        cursor.execute(table_creation)

    def insert_embedding_data(self,
                              smiles,
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
            INSERT INTO emb(smiles, embedding, embedding_dim)
            VALUES(?,?,?)
            ''',
            [smiles, sqlite3.Binary(embedding), sqlite3.Binary(embedding_dim)])
        self.conn.commit()

    def fetch_embedding_data(self, smiles):
        """
        Fetch the embedding data for a given dataset and smiles
        :param data:
        :return:
        """

        logger.debug('Fetching embedding data...')

        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT embedding, embedding_dim
            FROM emb
            WHERE smiles=?
            ''',
            [smiles])
        embedding_results = cursor.fetchone()

        if not embedding_results:
            return None

        embedding, embedding_dim = embedding_results
        embedding = pickle.loads(embedding)
        embedding_dim = pickle.loads(embedding_dim)
        return (embedding, embedding_dim)


class ChEMBLApprovedDrugsEmbeddingData(EmbeddingData):

    def __init__(self, sql_path='db/chembl_approved_drugs.sqlite3'):
        super().__init__(sql_path=sql_path)
        logger.info(f'ChEMBL approved drugs database {self.db_file}...')


class PhysChemEmbeddingData(EmbeddingData):

    def __init__(self, sql_path='db/physchem.sqlite3'):
        super().__init__(sql_path=sql_path)
        logger.info(f'Physchem properties database {self.db_file}...')


class BioActivityEmbeddingData(EmbeddingData):

    def __init__(self, sql_path='db/bioactivity.sqlite3'):
        super().__init__(sql_path=sql_path)
        logger.info(f'Bioactivities properties database {self.db_file}...')
        # TODO how to handle fingerprints -- store in table since there's so many?

    # def fetch_embedding_data(self, smiles): # TODO THIS IS PROBABLY WRONG -- remove it
    #     """
    #     Fetch the embedding data for a given dataset and smiles
    #     :param data:
    #     :return:
    #     """

    #     logger.debug('Fetching embedding data...')

    #     cursor = self.conn.cursor()
    #     cursor.execute(
    #         '''
    #         SELECT embedding, embedding_dim FROM emb
    #         WHERE smiles=?
    #         ''', [smiles])
    #     embedding_results = cursor.fetchone()

    #     if not embedding_results:
    #         return None

    #     return embedding_results