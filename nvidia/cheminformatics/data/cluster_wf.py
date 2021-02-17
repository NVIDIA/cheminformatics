from nvidia.cheminformatics.config import Context
from nvidia.cheminformatics.utils.singleton import Singleton
import os
import cudf
import dask_cudf
import dask
import logging
import sqlite3
from contextlib import closing

from typing import List

from . import ClusterWfDAO
from nvidia.cheminformatics.data.helper.chembldata import ChEmblData

logger = logging.getLogger(__name__)

FINGER_PRINT_FILES = 'filter_*.h5'


class ChemblClusterWfDao(ClusterWfDAO, metaclass=Singleton):

    def __init__(self):

        context = Context()
        db_file = context.get_config('data_mount_path', default='/data')

        self.chembl_db = 'file:%s/db/chembl_27.db?mode=ro' % db_file
        logger.info('Reading ChEMBL database at %s...' % self.chembl_db)

    def meta_df(self):
        chem_data = ChEmblData()
        return chem_data._meta_df()

    def fetch_molecular_embedding(self,
                                  n_molecules:int,
                                  cache_directory:str=None):
        chem_data = ChEmblData()
        if cache_directory:
            hdf_path = os.path.join(cache_directory, FINGER_PRINT_FILES)
            logger.info('Reading molecules from %s...' % hdf_path)
            mol_df = dask.dataframe.read_hdf(hdf_path, 'fingerprints')

            if n_molecules > 0:
                mol_df = mol_df.head(n_molecules, compute=False, npartitions=-1)
        else:
            logger.info('Reading molecules from database...')
            mol_df = chem_data.fetch_mol_embedding(num_recs=n_molecules)

        return mol_df

    def fetch_molecular_embedding_by_id(self, molecule_id:List):
        chem_data = ChEmblData()
        meta = chem_data._meta_df()
        fp_df = chem_data._fetch_mol_embedding(molregnos=molecule_id) \
                         .astype(meta.dtypes)

        fp_df = cudf.from_pandas(fp_df)
        fp_df = dask_cudf.from_cudf(fp_df, npartitions=1).reset_index()
        return fp_df

    def fetch_id_from_chembl(self, new_molecules: List):
        logger.debug('Fetch ChEMBL ID using molregno...')

        with closing(sqlite3.connect(self.chembl_db, uri=True)) as con, con,  \
                closing(con.cursor()) as cur:
            select_stmt = '''
                SELECT cs.molregno as molregno, md.chembl_id as chembl_id
                FROM compound_structures cs,
                    molecule_dictionary md
                WHERE md.molregno = cs.molregno
                    AND md.chembl_id in (%s)
            ''' %  "'%s'" %"','".join(new_molecules)
            cur.execute(select_stmt)

            return cur.fetchall()