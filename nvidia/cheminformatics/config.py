import os
import logging
from io import StringIO
from configparser import RawConfigParser

logger = logging.getLogger(__name__)


CONFIG_FILE = '.cheminf_local_environment'


class Context(object):
    from nvidia.cheminformatics.utils.singleton import Singleton
    __metaclass__ = Singleton


    def __init__(self):

        self.config = None
        if os.path.exists(CONFIG_FILE):
            logger.info('Reading properties from %s...', CONFIG_FILE)
            self.config = self._load_properties_file(CONFIG_FILE)
        else:
            logger.warn('Could not locate %s', CONFIG_FILE)

    def _load_properties_file(self, properties_file):
        """
        Reads a properties file using ConfigParser.

        :param propertiesFile/configFile:
        """
        config_file = open(properties_file, 'r')
        config_content = StringIO('[root]\n' + config_file.read())
        config = RawConfigParser()
        config.readfp(config_content)

        return config._sections['root']

    def get_config(self, config_name):
        """
        Returns values from local configuration.
        """
        return self.config[config_name]
