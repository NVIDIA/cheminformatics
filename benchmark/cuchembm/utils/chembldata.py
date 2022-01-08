import os
import sqlite3
import logging

from contextlib import closing
from configparser import RawConfigParser
from io import StringIO

logger = logging.getLogger(__name__)

class ChEmblData(object):

    def __init__(self):

        config_file = '.env'
        if not os.path.exists('.env'):
            config_file = '/workspace/.env'

        if os.path.exists(config_file):
            logger.info('Reading properties from %s...', config_file)
            config = self._load_properties_file(config_file)
        else:
            logger.warning('Could not locate .env file')

        db_file = getattr(config, 'data_mount_path', '/data')
        db_file = os.path.join(db_file, 'db', 'chembl_27.db')

        if not os.path.exists(db_file):
            logger.error('%s not found', db_file)
            raise Exception('{} not found'.format(db_file))

        self.chembl_db = 'file:%s?mode=ro' % db_file

        logger.info('ChEMBL database: %s...' % self.chembl_db)

    def _load_properties_file(self, properties_file):
        """
        Reads a properties file using ConfigParser.

        :param propertiesFile/configFile:
        """
        config_file = open(properties_file, 'r')
        config_content = StringIO('[root]\n' + config_file.read())
        config = RawConfigParser()
        config.read_file(config_content)

        return config._sections['root']

    def fetch_approved_drugs(self, with_labels=False):
        """Fetch approved drugs with phase >=3 as dataframe

        Args:
            with_labels (bool): return column labels as list, Default: False
        Returns:
            pd.DataFrame: dataframe containing SMILES strings and molecule index
        """
        logger.debug('Fetching ChEMBL approved drugs...')
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
                closing(con.cursor()) as cur:
            select_stmt = """SELECT
                di.molregno,
                cs.canonical_smiles,
                di.max_phase_for_ind
            FROM
                drug_indication AS di
            LEFT JOIN compound_structures AS cs ON di.molregno = cs.molregno
            WHERE
                di.max_phase_for_ind >= 3
                AND cs.canonical_smiles IS NOT NULL;"""
            cur.execute(select_stmt)
            if with_labels:
                labels = [x[0] for x in cur.description]
                return cur.fetchall(), labels
            else:
                return cur.fetchall()

    def fetch_approved_drugs_physchem(self, with_labels=False):
        """Fetch approved drugs with phase >=3 as dataframe, merging in physchem properties

        Args:
            with_labels (bool): return column labels as list, Default: False
        Returns:
            pd.DataFrame: dataframe containing SMILES strings and molecule index
        """
        logger.debug('Fetching ChEMBL approved drugs with physchem properties...')
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
                closing(con.cursor()) as cur:

            select_stmt = '''
                SELECT cs.canonical_smiles,
                di.max_phase_for_ind,
                cp.*
            FROM
                drug_indication AS di,
                compound_structures AS cs,
                compound_properties AS cp
            WHERE
                di.molregno = cs.molregno
                AND di.molregno = cp.molregno
                AND di.max_phase_for_ind >= 3
                AND canonical_smiles is not null
            GROUP BY cp.molregno,
                cs.canonical_smiles
            HAVING di.max_phase_for_ind = max(di.max_phase_for_ind);
            '''
            cur.execute(select_stmt)
            if with_labels:
                labels = [x[0] for x in cur.description]
                return cur.fetchall(), labels
            else:
                return cur.fetchall()

    def fetch_random_samples(self, num_samples, max_len):
        """Fetch random samples from ChEMBL as dataframe

        Args:
            num_samples (int): number of samples to select
            chembl_db_path (string): path to chembl sqlite database
        Returns:
            pd.DataFrame: dataframe containing SMILES strings and molecule index
        """
        logger.debug('Fetching ChEMBL random samples...')
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
                closing(con.cursor()) as cur:
            select_stmt = """SELECT
                cs.molregno,
                cs.canonical_smiles,
                LENGTH(cs.canonical_smiles) as len
            FROM
                compound_structures AS cs
            WHERE
                cs.canonical_smiles IS NOT NULL
            AND
                len <= """ + f'{max_len}' + """
            ORDER BY RANDOM()
            LIMIT """ + f'{num_samples};'

            cur.execute(select_stmt)
            return cur.fetchall()