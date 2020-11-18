import cudf
import pandas
import sqlite3
import logging

from dask import delayed, dataframe

from contextlib import closing
from nvidia.cheminformatics.utils.singleton import Singleton
from nvidia.cheminformatics.fingerprint import morgan_fingerprint


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

    CHEMBL_DB='file:/data/db/chembl_27.db?mode=ro'

    def fetch_props_by_chembl_ids(self, chembl_ids):
        """
        Returns compound properties and structure filtered by ChEMBL ids along
        with a list of columns.
        """
        with closing(sqlite3.connect(ChEmblData.CHEMBL_DB, uri=True)) as con, con,  \
                closing(con.cursor()) as cur:
            select_stmt = SQL_MOLECULAR_PROP % "'%s'" % "','".join(chembl_ids)
            cur.execute(select_stmt)

            cols = list(map(lambda x: x[0], cur.description))
            return cols, cur.fetchall()

    def fetch_props_df_by_chembl_ids(self, chemblIDs, gpu=True):
        """
        Returns compound properties and structure filtered by ChEMBL ids in a
        dataframe.
        """
        with closing(sqlite3.connect(ChEmblData.CHEMBL_DB, uri=True)) as con:
            select_stmt = SQL_MOLECULAR_PROP % "'%s'" % "','".join(chemblIDs)
            df = pandas.read_sql(select_stmt, con)

            if gpu:
                df = cudf.from_pandas(df)
                return df.sort_values('chembl_id')
            else:
                return df


    def fetch_molecule_cnt(self):
        logger.debug('Finding number of molecules...')
        with closing(sqlite3.connect(ChEmblData.CHEMBL_DB, uri=True)) as con, con,  \
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
    def fetch_molecular_props(self, start, batch_size=30000, radius=2, nBits=512):
        """
        Returns compound properties and structure for the first N number of
        records in a dataframe.
        """

        logger.debug('Fetching %d records starting %d...' % (batch_size, start))

        select_stmt = '''
            SELECT md.chembl_id, cs.canonical_smiles
            FROM compound_properties cp,
                 molecule_dictionary md,
                 compound_structures cs
            WHERE cp.molregno = md.molregno
                  AND md.molregno = cs.molregno
            LIMIT %d, %d
        ''' % (start, batch_size)
        df = pandas.read_sql(select_stmt,
                            sqlite3.connect(ChEmblData.CHEMBL_DB, uri=True),
                            index_col='chembl_id')

        df['fp'] = df.apply(lambda row: morgan_fingerprint(
                       row.canonical_smiles, radius=radius, nBits=nBits),
                       axis=1)

        return df['fp'].str.split(pat=', ', n=nBits+1, expand=True).astype('float32')

    def fetch_all_props(self, num_recs=None, batch_size=30000, radius=2, nBits=512):
        """
        Returns compound properties and structure for the first N number of
        records in a dataframe.
        """
        logger.debug('Fetching properties for all molecules...')

        if not num_recs or num_recs < 0:
            num_recs = self.fetch_molecule_cnt()

        prop_meta = {i: pandas.Series([], dtype='float32') for i in range(nBits)}
        meta_df = pandas.DataFrame(prop_meta)

        dls = []
        for start in range(0, num_recs, batch_size):
            bsize = min(num_recs - start, batch_size)
            dls.append(self.fetch_molecular_props(
                start, batch_size=bsize, radius=radius, nBits=nBits))

        return dataframe.from_delayed(dls, meta=meta_df)

    def save_fingerprints(hdf_path='data/filter_*.h5'):
        """
        Generates fingerprints for all ChEmblId's in the database
        """
        logger.debug('Fetching molecules from database for fingerprints...')

        chem_data = ChEmblData()
        mol_df = chem_data.fetch_all_props()

        mol_df.to_hdf(hdf_path, 'fingerprints')
