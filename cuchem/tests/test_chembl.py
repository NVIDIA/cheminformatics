import logging
from cuchemcommon.data.helper.chembldata import IMP_PROPS
from nvidia.cheminformatics.fingerprint import TransformationDefaults
from nvidia.cheminformatics.fingerprint import MorganFingerprint
import dask

from cuchemcommon.data.cluster_wf import ChemblClusterWfDao

from tests.utils import _create_context

logger = logging.getLogger(__name__)


def test_dataframe():
    """
    Verify fetching data from chemblDB.
    """
    context = _create_context()

    dao = ChemblClusterWfDao(MorganFingerprint)
    mol_df = dao.fetch_molecular_embedding(n_molecules=100)
    assert isinstance(mol_df, dask.dataframe.core.DataFrame),\
        'Incorrect data structure from DAO'
    fp_size = MorganFingerprint.DEFAULTS.value['nBits']

    # Fingerprint size + Important Columns + ID + (smile + tranformed smile)
    df_size = fp_size + len(IMP_PROPS) + 3

    logger.info(df_size)
    logger.info(fp_size)
    logger.info(mol_df.columns)
    logger.info(mol_df.head())

    assert mol_df.shape[1] == df_size, \
        'Expected dataframe size is %d found %d.' % (df_size, mol_df.shape[1])
