import logging
from chembench.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class Cache(metaclass=Singleton):

    def __init__(self):
        self._data = {}

    def delete(self, property):
        if property in self._data:
            self._data.pop(property)

    def set_data(self, property, value):
        """
        Stores data in local cache.
        """
        self._data[property] = value

    def get_data(self, property):
        """
        Returns values from local configuration.
        """
        try:
            return self._data[property]
        except KeyError:
            logger.warning('%s not found.', property)
            return None
