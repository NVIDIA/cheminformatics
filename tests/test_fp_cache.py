import os
import logging
import tempfile
import dask

from nvidia.cheminformatics.data.helper.chembldata import ChEmblData
from nvidia.cheminformatics.data.cluster_wf import FINGER_PRINT_FILES
from nvidia.cheminformatics.fingerprint import MorganFingerprint, Embeddings

logger = logging.getLogger(__name__)


def test_cache_morganfingerprint():
    """
    Verify fetching data from chemblDB.
    """
    num_recs = 1000

    logger.info('Morgan Fingerprinting Check!')

    cache_dir = os.path.join(tempfile.mkdtemp())
    logger.info('Creating cache at %s' % cache_dir)
    logger.info(type(cache_dir))

    # Write to cache
    chem_data = ChEmblData(fp_type = MorganFingerprint)
    chem_data.save_fingerprints(os.path.join(cache_dir, FINGER_PRINT_FILES),
                                num_recs=num_recs)

    # Verify cache
    hdf_path = os.path.join(cache_dir, FINGER_PRINT_FILES)
    logger.info('Reading molecules from %s...' % hdf_path)
    mol_df = dask.dataframe.read_hdf(hdf_path, 'fingerprints')
    mol_df = mol_df.compute()

    logger.info('Expected %s rec found %s.', num_recs, mol_df.shape[0])
    assert mol_df.shape[0] <= num_recs, \
        ('Expected %d rec found %d.' % (num_recs, mol_df.shape[0]))

def test_cache_cddd_embeddings():
    """
    Verify fetching data from chemblDB.
    """
    num_recs = 1000

    logger.info('CDDD Embeddings Check!')

    cache_dir = os.path.join(tempfile.mkdtemp())
    logger.info('Creating cache at %s' % cache_dir)
    logger.info(type(cache_dir))

    # Write to cache
    chem_data = ChEmblData(fp_type = Embeddings)
    chem_data.save_fingerprints(os.path.join(cache_dir, FINGER_PRINT_FILES),
                                num_recs=num_recs)

    # Verify cache
    hdf_path = os.path.join(cache_dir, FINGER_PRINT_FILES)
    logger.info('Reading molecules from %s...' % hdf_path)
    mol_df = dask.dataframe.read_hdf(hdf_path, 'fingerprints')
    mol_df = mol_df.compute()

    logger.info('Expected %s rec found %s.', num_recs, mol_df.shape[0])
    assert mol_df.shape[0] <= num_recs, \
        ('Expected %d rec found %d.' % (num_recs, mol_df.shape[0]))