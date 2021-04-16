import os
import math
import cudf
import dask_cudf
import dask
import logging
import sqlite3
from contextlib import closing

from typing import List

from . import ClusterWfDAO
from nvidia.cheminformatics.data.helper.chembldata import BATCH_SIZE, ChEmblData
from nvidia.cheminformatics.config import Context
from nvidia.cheminformatics.utils.singleton import Singleton

logger = logging.getLogger(__name__)

FINGER_PRINT_FILES = 'filter_*.h5'


class ChemblClusterWfDao(ClusterWfDAO, metaclass=Singleton):

    def meta_df(self):
        chem_data = ChEmblData()
        return chem_data._meta_df()

    def fetch_molecular_embedding(self,
                                  n_molecules:int,
                                  cache_directory:str=None):
        chem_data = ChEmblData()
        context = Context()
        if cache_directory:
            hdf_path = os.path.join(cache_directory, FINGER_PRINT_FILES)
            logger.info('Reading %d rows from %s...', n_molecules, hdf_path)
            mol_df = dask.dataframe.read_hdf(hdf_path, 'fingerprints')

            if n_molecules > 0:
                npartitions = math.ceil(n_molecules / BATCH_SIZE)
                mol_df = mol_df.head(n_molecules, compute=False, npartitions=npartitions)
        else:
            logger.info('Reading molecules from database...')
            mol_df = chem_data.fetch_mol_embedding(num_recs=n_molecules,
                                                   batch_size=context.batch_size)

        return mol_df

    def fetch_molecular_embedding_by_id(self, molecule_id:List):
        chem_data = ChEmblData()
        context = Context()
        meta = chem_data._meta_df()
        fp_df = chem_data._fetch_mol_embedding(molregnos=molecule_id,
                                               batch_size=context.batch_size) \
                         .astype(meta.dtypes)

        fp_df = cudf.from_pandas(fp_df)
        fp_df = dask_cudf.from_cudf(fp_df, npartitions=1).reset_index()
        return fp_df

    def fetch_id_from_chembl(self, new_molecules: List):
        logger.debug('Fetch ChEMBL ID using molregno...')
        chem_data = ChEmblData()
        return chem_data.fetch_id_from_chembl(new_molecules)
