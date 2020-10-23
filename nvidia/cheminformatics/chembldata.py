import cudf
import pandas
import sqlite3
import logging

from contextlib import closing
from nvidia.cheminformatics.utils.singleton import Singleton

SQL_MOLECULAR_PROP = """
SELECT md.chembl_id, cp.*, cs.*
    FROM compound_properties cp,
            compound_structures cs,
            molecule_dictionary md
    WHERE cp.molregno = md.molregno
        AND md.molregno = cs.molregno
        AND md.chembl_id in (%s);
"""

logger = logging.getLogger(__name__)

CHEMBL_DB='/data/db/chembl_27.db'

class ChEmblData(object, metaclass=Singleton):


    def fetch_props_by_chembl_ids(self, chembl_ids):
        """
        Returns compound properties and structure filtered by ChEMBL ids along
        with a list of columns.
        """
        with closing(sqlite3.connect(CHEMBL_DB)) as con, con,  \
                closing(con.cursor()) as cur:
            select_stmt = SQL_MOLECULAR_PROP % "'%s'" % "','".join(chembl_ids)
            cur.execute(select_stmt)

            cols = list(map(lambda x: x[0], cur.description))
            return cols, cur.fetchall()

    def fetch_props_df_by_chembl_ids(self, chemblIDs) -> cudf.DataFrame:
        """
        Returns compound properties and structure filtered by ChEMBL ids in a
        dataframe.
        """
        with closing(sqlite3.connect(CHEMBL_DB)) as con:
            select_stmt = SQL_MOLECULAR_PROP % "'%s'" % "','".join(chemblIDs)
            df = pandas.read_sql(select_stmt, con)
            if not self.enable_gpu:
                return df

            df = cudf.from_pandas(df)
            return df.sort_values('chembl_id')

    def fetch_props(self, record_cnt):
        """
        Returns compound properties and structure for the first N number of
        records in a dataframe.
        """
        logger.info('Fetching %d records...' % record_cnt)
        with closing(sqlite3.connect(CHEMBL_DB)) as con:
            select_stmt = '''
                SELECT md.molregno as mol_id, md.chembl_id, cp.*, cs.*
                FROM compound_properties cp,
                     molecule_dictionary md,
                     compound_structures cs
                WHERE cp.molregno = md.molregno
                    AND md.molregno = cs.molregno
                LIMIT %d;
            ''' % record_cnt

            df = cudf.from_pandas(pandas.read_sql(select_stmt, con))
            return df.sort_values('chembl_id')
