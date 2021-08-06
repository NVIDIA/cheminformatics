import logging

from cuchemportal.utils.singleton import Singleton
from cuchemportal.data.db_client import DBClient

logger = logging.getLogger(__name__)


class Context(metaclass=Singleton):
    """
    A singleton class to store globale variables and settings.
    """
    def __init__(self):
        self._db_client = None
        self._db_config = None

    @property
    def db_client(self):
        if not self._db_client:

            if self._db_config is None:
                raise ValueError('Database host not set in context.')

            # TODO: Replace with values from config file
            self.conn_str: str = (
                "mysql+pymysql://{0}:{1}@{2}/{3}".format(self._db_config.username,
                                                         self._db_config.password,
                                                         self._db_config.server,
                                                         self._db_config.database))
            self._db_client = DBClient(self.conn_str)
        return self._db_client

    @property
    def db_config(self):
        return self._db_config

    @db_config.setter
    def db_config(self, value):
        self._db_config = value
