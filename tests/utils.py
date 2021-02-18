import os
import dask
import tempfile

from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.utils.dask import initialize_cluster
from nvidia.cheminformatics.config import Context


def _fetch_chembl_test_dataset(n_molecules=None):
    if n_molecules is None:
        n_molecules = 1000

    dao = ChemblClusterWfDao()
    mol_df = dao.fetch_molecular_embedding(n_molecules=n_molecules)
    assert isinstance(mol_df, dask.dataframe.core.DataFrame),\
        'Incorrect data structure from DAO'

    return n_molecules, dao, mol_df


def _create_context():
    context = Context()
    if context.dask_client is None:
        context.dask_client = initialize_cluster()
    context.is_benchmark = False
    context.benchmark_file = os.path.join(tempfile.tempdir, 'benchmark.csv')
    context.cache_directory=tempfile.tempdir

    return context

