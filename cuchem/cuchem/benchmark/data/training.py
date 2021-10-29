import os
import sqlite3
import logging
import pickle
from typing import List

from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.context import Context

__all__ = ['ZINC15TrainDataset']

logger = logging.getLogger(__name__)


class ZINC15TrainDataset(object, metaclass=Singleton):

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