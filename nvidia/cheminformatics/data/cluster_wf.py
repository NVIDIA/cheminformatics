import os
import dask
import logging
import dask_cudf

from typing import List

from . import ClusterWfDAO
from nvidia.cheminformatics.data.helper.chembldata import ChEmblData

logger = logging.getLogger(__name__)

FINGER_PRINT_FILES = 'filter_*.h5'


class ChemblClusterWfDao(ClusterWfDAO):

    def fetch_molecular_embedding(self, n_molecules:int, cache_directory:str=None):
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

    def fetch_new_molecules(self, new_molecules: List):
        pass
