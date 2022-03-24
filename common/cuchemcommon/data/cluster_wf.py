import logging
import math
import os
from typing import List

import cudf
import dask
import dask_cudf
import sys
from cuchemcommon.context import Context
from cuchemcommon.data.helper.chembldata import BATCH_SIZE, ChEmblData
from cuchemcommon.utils.singleton import Singleton

from . import ClusterWfDAO

logger = logging.getLogger(__name__)

FINGER_PRINT_FILES = 'filter_*.h5'


class ChemblClusterWfDao(ClusterWfDAO, metaclass=Singleton):

    def __init__(self, fp_type, radius=2, nBits=512):
        logger.info(f'ChemblClusterWfDao({fp_type})')
        self.chem_data = ChEmblData(fp_type)
        self.radius = radius
        self.nBits = nBits

    def meta_df(self):
        chem_data = ChEmblData()
        return chem_data._meta_df()

    def fetch_molecular_embedding(self,
                                  n_molecules: int,
                                  cache_directory: str = None, 
                                  radius=2, 
                                  nBits=512):
        # Since we allow the user to change the fingerprint radius and length (nBits),
        # the fingerprints need to be cached in separate subdirectories.
        # Note: the precomputed ones are not presumed to be of a specific radius or length
        context = Context()
        if cache_directory:
            cache_subdir = f'{cache_dir}/fp_r{radius}_n{nBits}'
            hdf_path = os.path.join(cache_subdir, FINGER_PRINT_FILES)
        else:
            cache_subdir = None
            hdf_path = None
        if cache_directory and os.path.isdir(cache_subdir): # and (self.radius == radius) and (self.nBits == nBits):
            logger.info('Reading %d rows from %s...', n_molecules, hdf_path)
            mol_df = dask.dataframe.read_hdf(hdf_path, 'fingerprints')
            if len(mol_df) == 0:
                logger.info(f'Zero molecules found in {hdf_path}! Caching error?')
            if n_molecules > 0:
                npartitions = math.ceil(n_molecules / BATCH_SIZE)
                mol_df = mol_df.head(n_molecules, compute=False, npartitions=npartitions)
        else:
            self.radius = radius
            self.nBits = nBits
            logger.info(f'Reading molecules from database and computing fingerprints (radius={self.radius}, nBits={self.nBits})...')
            sys.stdout.flush()
            mol_df = self.chem_data.fetch_mol_embedding(
                num_recs=n_molecules,
                batch_size=context.batch_size,
                radius=radius,
                nBits=nBits
            )
            if cache_directory:
                os.mkdir(cache_subdir)
                logger.info(f'Caching mol_df fingerprints to {hdf_path}')
                mol_df.to_hdf(hdf_path, 'fingerprints')
            else:
                logging.info(f'cache_directory={cache_directory}, not caching!')
        sys.stdout.flush()
        return mol_df

    def fetch_molecular_embedding_by_id(self, molecule_id: List, radius=2, nBits=512):
        context = Context()
        meta = self.chem_data._meta_df(
            f'fetch_molecular_embedding_by_id({molecule_id}): MISMATCH!!! radius: {radius} != {self.radius}, nBits: {nBits} != {self.nBits}')
        if (self.radius != radius) or (self.nBits != nBits):
            logger.info('Something broken?')
        fp_df = self.chem_data._fetch_mol_embedding(
            molregnos=molecule_id,
            batch_size=context.batch_size,
            radius=radius,
            nBits=nBits
        ).astype(meta.dtypes)

        fp_df = cudf.from_pandas(fp_df)
        fp_df = dask_cudf.from_cudf(fp_df, npartitions=1).reset_index()
        return fp_df

    def fetch_id_from_chembl(self, new_molecules: List):
        logger.debug('Fetch ChEMBL ID using molregno...')
        return self.chem_data.fetch_id_from_chembl(new_molecules)
