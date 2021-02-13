import os
import cudf
import dask
import logging
import tempfile

from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.wf.cluster.gpukmeansumap import GpuKmeansUmap
from nvidia.cheminformatics.utils.dask import initialize_cluster
from nvidia.cheminformatics.config import Context


logger = logging.getLogger(__name__)


def _fetch_test_dataset(n_molecules=None):
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


# def test_gpukmeansumap_dask():
#     """
#     Verify fetching data from chemblDB when the input is a pandas df.
#     """
#     n_molecules, dao, mol_df = _fetch_test_dataset()

#     context = _create_context()
#     wf = GpuKmeansUmap(n_molecules=n_molecules,
#                     dao=dao, pca_comps=64)
#     wf.cluster(df_mol_embedding=mol_df)


# def test_gpukmeansumap_cudf():
#     """
#     Verify fetching data from chemblDB when the input is a cudf df.
#     """
#     context = _create_context()

#     n_molecules, dao, mol_df = _fetch_test_dataset()
#     wf = GpuKmeansUmap(n_molecules=n_molecules,
#                     dao=dao, pca_comps=64)
#     mol_df = mol_df.compute()
#     wf.cluster(df_mol_embedding=mol_df)


def test_add_molecule():
    """
    Verify fetching data from chemblDB when the input is a cudf df.
    """
    context = _create_context()

    n_molecules, dao, mol_df = _fetch_test_dataset()

    if hasattr(mol_df, 'compute'):
        mol_df = mol_df.compute()

    mol_df = cudf.from_pandas(mol_df)
    n_molecules = mol_df.shape[0]

    # test mol should container aviable and new molecules
    test_mol = mol_df[n_molecules - 20:]
    mols_tobe_added = test_mol['canonical_smiles'].to_array().tolist()

    # Molecules to be used for clustering
    mol_df = mol_df[:n_molecules - 10]
    logger.info('Number of molecules to be added: %d', len(mols_tobe_added))

    wf = GpuKmeansUmap(n_molecules=n_molecules,
                       dao=dao, pca_comps=64)
    wf.cluster(df_mol_embedding=mol_df)

    missing_molregno, molregnos, df_embedding = wf.add_molecules(mols_tobe_added)
    assert len(missing_molregno) == 10, 'Expected 10 missing molecules found %d' % len(missing_molregno)

    missing_molregno, molregnos, df_embedding = wf.add_molecules(mols_tobe_added)
    assert len(missing_molregno) == 0, 'Expected no missing molecules found %d' % len(missing_molregno)
