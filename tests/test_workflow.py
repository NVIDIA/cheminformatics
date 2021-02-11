import os
import dask
import logging
import tempfile
from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.wf.cluster.gpukmeansumap import GpuKmeansUmap
from nvidia.cheminformatics.utils.dask import initialize_cluster
from nvidia.cheminformatics.config import Context


logger = logging.getLogger(__name__)


def _create_context():
    context = Context()
    if context.dask_client is None:
        context.dask_client = initialize_cluster()
    context.is_benchmark = False
    context.benchmark_file = os.path.join(tempfile.tempdir, 'benchmark.csv')
    context.cache_directory=tempfile.tempdir

    return context


def test_gpukmeansumap_dask():
    """
    Verify fetching data from chemblDB when the input is a pandas df.
    """
    n_molecules = 1000

    dao = ChemblClusterWfDao()
    mol_df = dao.fetch_molecular_embedding(n_molecules=n_molecules)
    assert isinstance(mol_df, dask.dataframe.core.DataFrame),\
        'Incorrect data structure from DAO'

    context = _create_context()
    with context.dask_client:
        wf = GpuKmeansUmap(n_molecules=n_molecules,
                        dao=dao, pca_comps=64)
        wf.cluster(df_mol_embedding=mol_df)


def test_gpukmeansumap_cudf():
    """
    Verify fetching data from chemblDB when the input is a cudf df.
    """
    n_molecules = 1000

    dao = ChemblClusterWfDao()
    mol_df = dao.fetch_molecular_embedding(n_molecules=n_molecules)
    assert isinstance(mol_df, dask.dataframe.core.DataFrame),\
        'Incorrect data structure from DAO'

    context = _create_context()

    with context.dask_client:
        wf = GpuKmeansUmap(n_molecules=n_molecules,
                        dao=dao, pca_comps=64)
        mol_df = mol_df.compute()
        wf.cluster(df_mol_embedding=mol_df)

