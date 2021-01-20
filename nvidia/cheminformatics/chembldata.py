import cudf
import pandas
import sqlite3
import logging

from dask import delayed, dataframe

from contextlib import closing
from nvidia.cheminformatics.utils.singleton import Singleton
from nvidia.cheminformatics.fingerprint import MorganFingerprint

SQL_MOLECULAR_PROP = """
SELECT md.molregno as molregno, md.chembl_id, cp.*, cs.*
    FROM compound_properties cp,
            compound_structures cs,
            molecule_dictionary md
    WHERE cp.molregno = md.molregno
        AND md.molregno = cs.molregno
        AND md.molregno in (%s)
"""


logger = logging.getLogger(__name__)


class ChEmblData(object, metaclass=Singleton):

    def __init__(self, 
                 db_file='/data/db/chembl_27.db', 
                 fp_type=MorganFingerprint):
        self.chembl_db = 'file:%s?mode=ro' % db_file
        self.fp_type = fp_type

    def fetch_props_by_molregno(self, molregnos):
        """
        Returns compound properties and structure filtered by ChEMBL ids along
        with a list of columns.
        """
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con,  \
                closing(con.cursor()) as cur:
            select_stmt = SQL_MOLECULAR_PROP % " ,".join(list(map(str, molregnos)))
            logger.info(select_stmt)
            cur.execute(select_stmt)

            cols = list(map(lambda x: x[0], cur.description))
            return cols, cur.fetchall()

    def fetch_props_df_by_molregno(self, molregnos, gpu=True):
        """
        Returns compound properties and structure filtered by ChEMBL ids in a
        dataframe.
        """
        select_stmt = SQL_MOLECULAR_PROP % " ,".join(molregnos)
        df = pandas.read_sql(select_stmt,
                             sqlite3.connect(self.chembl_db, uri=True),
                             index_col='molregno')
        if gpu:
            return cudf.from_pandas(df)
        else:
            return df


    def fetch_molregno_by_chemblId(self, chemblIds):
        logger.debug('Fetch chemblId using molregno...')
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con,  \
                closing(con.cursor()) as cur:
            select_stmt = '''
                SELECT md.molregno as molregno
                FROM compound_properties cp,
                        compound_structures cs,
                        molecule_dictionary md
                WHERE cp.molregno = md.molregno
                    AND md.molregno = cs.molregno
                    AND md.chembl_id in (%s)
            ''' %  "'%s'" %"','".join(chemblIds)
            cur.execute(select_stmt)
            return cur.fetchall()


    def fetch_molecule_cnt(self):
        logger.debug('Finding number of molecules...')
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con,  \
                closing(con.cursor()) as cur:
            select_stmt = '''
                SELECT count(*)
                FROM compound_properties cp,
                     molecule_dictionary md,
                     compound_structures cs
                WHERE cp.molregno = md.molregno
                      AND md.molregno = cs.molregno
            '''
            cur.execute(select_stmt)

            return cur.fetchone()[0]

    @delayed
    def fetch_molecular_props(self,
                              start,
                              batch_size=30000,
                              **transformation_kwargs):
        """
        Returns compound properties and structure for the first N number of
        records in a dataframe.
        """

        logger.info('Fetching %d records starting %d...' % (batch_size, start))

        select_stmt = '''
            SELECT md.molregno, cs.canonical_smiles
            FROM compound_properties cp,
                 molecule_dictionary md,
                 compound_structures cs
            WHERE cp.molregno = md.molregno
                  AND md.molregno = cs.molregno
            LIMIT %d, %d
        ''' % (start, batch_size)
        df = pandas.read_sql(select_stmt,
                            sqlite3.connect(self.chembl_db, uri=True),
                            index_col='molregno')

        transformation = self.fp_type(**transformation_kwargs)
        result_df = transformation.transform(df)
        return result_df

    def fetch_all_props(self,
                        num_recs=None,
                        batch_size=30000,
                        **transformation_kwargs):
        """
        Returns compound properties and structure for the first N number of
        records in a dataframe.
        """
        logger.debug('Fetching properties for all molecules...')

        if not num_recs or num_recs < 0:
            num_recs = self.fetch_molecule_cnt()

        transformation = self.fp_type(**transformation_kwargs)
        prop_meta = {i: pandas.Series([], dtype='float32') for i in range(len(transformation))}
        meta_df = pandas.DataFrame(prop_meta)

        dls = []
        for start in range(0, num_recs, batch_size):
            bsize = min(num_recs - start, batch_size)
            dls.append(self.fetch_molecular_props(
                                start,
                                batch_size=bsize,
                                **transformation_kwargs))

        return dataframe.from_delayed(dls, meta=meta_df)

    def save_fingerprints(self, hdf_path='data/filter_*.h5'):
        """
        Generates fingerprints for all ChEmblId's in the database
        """
        logger.debug('Fetching molecules from database for fingerprints...')

        mol_df = self.fetch_all_props()
        mol_df.to_hdf(hdf_path, 'fingerprints')
