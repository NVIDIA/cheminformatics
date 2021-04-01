import os
import dask
import tempfile
import logging

from nvidia.cheminformatics.utils.dask import initialize_cluster
from nvidia.cheminformatics.config import Context


logger = logging.getLogger(__name__)


def _fetch_chembl_test_dataset(n_molecules=None):
    if n_molecules is None:
        n_molecules = 1000

    from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
    dao = ChemblClusterWfDao()
    mol_df = dao.fetch_molecular_embedding(n_molecules=n_molecules)
    assert isinstance(mol_df, dask.dataframe.core.DataFrame),\
        'Incorrect data structure from DAO'

    return n_molecules, dao, mol_df


def _create_context(use_gpu=True,
                    n_workers=-1,
                    benchmark_file=None,
                    cache_directory=None,
                    batch_size=None):
    context = Context()
    if context.dask_client is None:
        context.dask_client = initialize_cluster(use_gpu=use_gpu,
                                                 n_gpu=n_workers,
                                                 n_cpu=n_workers)
    context.is_benchmark = False

    context.cache_directory = cache_directory
    if cache_directory is None:
        context.cache_directory = tempfile.tempdir

    context.benchmark_file = benchmark_file
    if benchmark_file is None:
        context.benchmark_file = os.path.join(tempfile.tempdir, 'benchmark.csv')

    context.batch_size = batch_size
    if batch_size is None:
        context.batch_size = 10000

    return context
