import os
import warnings
import pandas
import sqlite3
import logging

from typing import List
from dask import delayed, dataframe

from contextlib import closing
from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.context import Context

warnings.filterwarnings("ignore", message=r"deprecated", category=FutureWarning)
logger = logging.getLogger(__name__)

BATCH_SIZE = 100000
ADDITIONAL_FEILD = ['canonical_smiles', 'transformed_smiles']
IMP_PROPS = [
    'alogp',
    'aromatic_rings',
    'full_mwt',
    'psa',
    'rtb']
IMP_PROPS_TYPE = [pandas.Series([], dtype='float64'),
                  pandas.Series([], dtype='int64'),
                  pandas.Series([], dtype='float64'),
                  pandas.Series([], dtype='float64'),
                  pandas.Series([], dtype='int64')]
ADDITIONAL_FEILD_TYPE = [pandas.Series([], dtype='object'),
                         pandas.Series([], dtype='object')]

SQL_MOLECULAR_PROP = """
SELECT md.molregno as molregno, md.chembl_id, cp.*, cs.*
FROM molecule_dictionary as md
    join compound_structures cs on md.molregno = cs.molregno
    left join compound_properties cp on cs.molregno = cp.molregno
WHERE  md.molregno in (%s)
"""


# DEPRECATED. Please add code to DAO classes.
class ChEmblData(object, metaclass=Singleton):

    def __init__(self, fp_type):

        context = Context()
        db_file = context.get_config('data_mount_path', default='/data')
        db_file = os.path.join(db_file, 'db/chembl_27.db')

        if not os.path.exists(db_file):
            logger.error('%s not found', db_file)
            raise Exception('{} not found'.format(db_file))

        self.fp_type = fp_type
        self.chembl_db = 'file:%s?mode=ro' % db_file

        logger.info('ChEMBL database: %s...' % self.chembl_db)

    def fetch_props_by_molregno(self, molregnos):
        """
        Returns compound properties and structure filtered by ChEMBL IDs along
        with a list of columns.
        """
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
                closing(con.cursor()) as cur:
            select_stmt = SQL_MOLECULAR_PROP % " ,".join(list(map(str, molregnos)))
            cur.execute(select_stmt)

            cols = list(map(lambda x: x[0], cur.description))
            return cols, cur.fetchall()

    def fetch_props_by_chemble(self, chemble_ids):
        """
        Returns compound properties and structure filtered by ChEMBL IDs along
        with a list of columns.
        """
        sql_stml = """
            SELECT md.molregno as molregno, md.chembl_id, cp.*, cs.*
            FROM molecule_dictionary as md
                join compound_properties cp on md.molregno = cp.molregno
                left join compound_structures cs on cp.molregno = cs.molregno
            WHERE  md.chembl_id in (%s)
            """
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
                closing(con.cursor()) as cur:
            select_stmt = sql_stml % "'%s'" % "','".join([x.strip().upper() for x in chemble_ids])
            cur.execute(select_stmt)

            cols = list(map(lambda x: x[0], cur.description))
            return cols, cur.fetchall()

    def fetch_molregno_by_chemblId(self, chemblIds):
        logger.debug('Fetch ChEMBL ID using molregno...')
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
                closing(con.cursor()) as cur:
            select_stmt = '''
                SELECT md.molregno as molregno
                FROM molecule_dictionary as md
                    join compound_properties cp on md.molregno = cp.molregno
                    left join compound_structures cs on cp.molregno = cs.molregno
                WHERE  md.chembl_id in in (%s)
            ''' % "'%s'" % "','".join(chemblIds)
            cur.execute(select_stmt)
            return cur.fetchall()

    def fetch_id_from_chembl(self, new_molecules: List):
        logger.debug('Fetch ChEMBL ID using molregno...')

        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
                closing(con.cursor()) as cur:
            select_stmt = '''
                SELECT md.molregno as molregno, md.chembl_id as chembl_id,
                    cs.canonical_smiles as smiles
                FROM molecule_dictionary as md
                    join compound_structures cs on md.molregno = cs.molregno
                WHERE md.chembl_id in (%s)
            ''' % "'%s'" % "','".join([x.strip().upper() for x in new_molecules])
            cur.execute(select_stmt)

            return cur.fetchall()

    def fetch_chemblId_by_molregno(self, molregnos):
        logger.debug('Fetch ChEMBL ID using molregno...')
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
                closing(con.cursor()) as cur:
            select_stmt = '''
                SELECT md.chembl_id as chembl_id
                FROM molecule_dictionary md
                WHERE md.molregno in (%s)
            ''' % ", ".join(list(map(str, molregnos)))
            cur.execute(select_stmt)
            return cur.fetchall()

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
            labels = [x[0] for x in cur.description]
            if with_labels:
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
            labels = [x[0] for x in cur.description]
            if with_labels:
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

    def fetch_molecule_cnt(self):
        logger.debug('Finding number of molecules...')
        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
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

    def _meta_df(self, **transformation_kwargs):
        transformation = self.fp_type(**transformation_kwargs)

        prop_meta = {'id': pandas.Series([], dtype='int64')}
        prop_meta.update(dict(zip(IMP_PROPS + ADDITIONAL_FEILD,
                                  IMP_PROPS_TYPE + ADDITIONAL_FEILD_TYPE)))
        prop_meta.update({i: pandas.Series([], dtype='float32') for i in range(len(transformation))})

        return pandas.DataFrame(prop_meta)

    def _fetch_mol_embedding(self,
                             start=0,
                             batch_size=BATCH_SIZE,
                             molregnos=None,
                             **transformation_kwargs):
        """
        Returns compound properties and structure for the first N number of
        records in a dataframe.
        """

        logger.info('Fetching %d records starting %d...' % (batch_size, start))

        imp_cols = ['cp.' + col for col in IMP_PROPS]

        if molregnos is None:
            select_stmt = '''
                SELECT md.molregno, %s, cs.canonical_smiles
                FROM compound_properties cp,
                    molecule_dictionary md,
                    compound_structures cs
                WHERE cp.molregno = md.molregno
                    AND md.molregno = cs.molregno
                LIMIT %d, %d
            ''' % (', '.join(imp_cols), start, batch_size)
        else:
            select_stmt = '''
                SELECT md.molregno, %s, cs.canonical_smiles
                FROM compound_properties cp,
                    molecule_dictionary md,
                    compound_structures cs
                WHERE cp.molregno = md.molregno
                    AND md.molregno = cs.molregno
                    AND md.molregno in (%s)
                LIMIT %d, %d
            ''' % (', '.join(imp_cols), " ,".join(list(map(str, molregnos))), start, batch_size)

        df = pandas.read_sql(select_stmt,
                             sqlite3.connect(self.chembl_db, uri=True))

        # Smiles -> Smiles transformation and filtering
        # TODO: Discuss internally to find use or refactor this code to remove
        # model specific filtering
        df['transformed_smiles'] = df['canonical_smiles']
        # if smiles_transforms is not None:
        #     if len(smiles_transforms) > 0:
        #         for xf in smiles_transforms:
        #             df['transformed_smiles'] = df['transformed_smiles'].map(xf.transform)
        #             df.dropna(subset=['transformed_smiles'], axis=0, inplace=True)

        # Conversion to fingerprints or embeddings
        # transformed_smiles = df['transformed_smiles']
        transformation = self.fp_type(**transformation_kwargs)
        cache_data = transformation.transform(df)
        return_df = pandas.DataFrame(cache_data)

        return_df = pandas.DataFrame(
            return_df,
            columns=pandas.RangeIndex(start=0,
                                      stop=len(transformation))).astype('float32')

        return_df = df.merge(return_df, left_index=True, right_index=True)
        return_df.rename(columns={'molregno': 'id'}, inplace=True)
        return return_df

    def fetch_mol_embedding(self,
                            num_recs=None,
                            batch_size=BATCH_SIZE,
                            molregnos=None,
                            **transformation_kwargs):
        """
        Returns compound properties and structure for the first N number of
        records in a dataframe.
        """
        logger.debug('Fetching properties for all molecules...')

        if num_recs is None or num_recs < 0:
            num_recs = self.fetch_molecule_cnt()

        logger.info('num_recs %d', num_recs)
        logger.info('batch_size %d', batch_size)
        meta_df = self._meta_df(**transformation_kwargs)

        dls = []
        for start in range(0, num_recs, batch_size):
            bsize = min(num_recs - start, batch_size)
            dl_data = delayed(self._fetch_mol_embedding)(start=start,
                                                         batch_size=bsize,
                                                         molregnos=molregnos,
                                                         **transformation_kwargs)
            dls.append(dl_data)

        return dataframe.from_delayed(dls, meta=meta_df)

    def save_fingerprints(self, hdf_path='data/filter_*.h5', num_recs=None, batch_size=5000):
        """
        Generates fingerprints for all ChEMBL ID's in the database
        """
        logger.debug('Fetching molecules from database for fingerprints...')

        mol_df = self.fetch_mol_embedding(num_recs=num_recs, batch_size=batch_size)
        mol_df.to_hdf(hdf_path, 'fingerprints')


    def is_valid_chemble_smiles(self, smiles, con=None):

        if con is None:
            with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con, \
                    closing(con.cursor()) as cur:
                select_stmt = '''
                    SELECT count(*)
                    FROM compound_structures cs
                    WHERE canonical_smiles = ?
                '''
                cur.execute(select_stmt, (smiles))
                return cur.fetchone()[0]
        else:
            cur = con.cursor()
            select_stmt = '''
                    SELECT count(*)
                    FROM compound_structures cs
                    WHERE canonical_smiles = ?
                '''
            cur.execute(select_stmt, (smiles,))

            return cur.fetchone()[0]
