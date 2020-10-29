import cudf
import pandas
import sqlite3
import logging

from dask import delayed, dataframe

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


class ChEmblData(object, metaclass=Singleton):

    CHEMBL_DB='/data/db/chembl_27.db'

    def fetch_props_by_chembl_ids(self, chembl_ids):
        """
        Returns compound properties and structure filtered by ChEMBL ids along
        with a list of columns.
        """
        with closing(sqlite3.connect(ChEmblData.CHEMBL_DB)) as con, con,  \
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
        with closing(sqlite3.connect(ChEmblData.CHEMBL_DB)) as con:
            select_stmt = SQL_MOLECULAR_PROP % "'%s'" % "','".join(chemblIDs)
            df = pandas.read_sql(select_stmt, con)
            if not self.enable_gpu:
                return df

            df = cudf.from_pandas(df)
            return df.sort_values('chembl_id')


def fetch_molecule_cnt():
    logger.debug('Finding number of molecules...')
    with closing(sqlite3.connect(ChEmblData.CHEMBL_DB)) as con, con,  \
            closing(con.cursor()) as cur:
        select_stmt = '''
            SELECT count(*)
            FROM molecule_dictionary md
        '''
        cur.execute(select_stmt)

        return cur.fetchone()[0]

@delayed
def fetch_molecular_props(start, batch_size):
    """
    Returns compound properties and structure for the first N number of
    records in a dataframe.
    """

    logger.info('Fetching %d records starting %d...' % (batch_size, start))

    select_stmt = '''
        SELECT md.molregno, md.chembl_id, cs.canonical_smiles
        FROM compound_properties cp,
                molecule_dictionary md,
                compound_structures cs
        WHERE cp.molregno = md.molregno
            AND md.molregno = cs.molregno
        LIMIT %d, %d
    ''' % (start, batch_size)
    df = pandas.read_sql(select_stmt,
                         sqlite3.connect(ChEmblData.CHEMBL_DB),
                         index_col='molregno')
    return df

def fetch_all_props(num_recs=None, batch_size=10000):
    """
    Returns compound properties and structure for the first N number of
    records in a dataframe.
    """
    logger.info('Fetching properties for all molecules...')

    if not num_recs:
        num_recs = fetch_molecule_cnt()

    div = [start for start in range(0, num_recs, batch_size)]
    dls = []

    for start in div:
        bsize = min(num_recs - start, batch_size)
        dls.append(fetch_molecular_props(start, bsize))

    return dataframe.from_delayed(dls)
