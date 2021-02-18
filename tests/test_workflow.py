from nvidia.cheminformatics.wf.cluster.gpurandomprojection import GpuWorkflowRandomProjection
from nvidia.cheminformatics.data.helper.chembldata import ChEmblData
import os
import cudf
import dask
import logging
import tempfile

from tests.utils import _fetch_chembl_test_dataset, _create_context

from nvidia.cheminformatics.data.cluster_wf import ChemblClusterWfDao
from nvidia.cheminformatics.wf.cluster.gpukmeansumap import GpuKmeansUmap
from nvidia.cheminformatics.utils.dask import initialize_cluster
from nvidia.cheminformatics.config import Context


logger = logging.getLogger(__name__)


def test_random_proj():
    """
    Verify fetching data from chemblDB when the input is a pandas df.
    """
    n_molecules, dao, mol_df = _fetch_chembl_test_dataset()

    context = _create_context()
    wf = GpuWorkflowRandomProjection(n_molecules=n_molecules,
                                     dao=dao)
    wf.cluster(df_mol_embedding=mol_df)


def test_gpukmeansumap_dask():
    """
    Verify fetching data from chemblDB when the input is a pandas df.
    """
    n_molecules, dao, mol_df = _fetch_chembl_test_dataset()

    context = _create_context()
    wf = GpuKmeansUmap(n_molecules=n_molecules,
                       dao=dao, pca_comps=64)
    wf.cluster(df_mol_embedding=mol_df)


def test_gpukmeansumap_cudf():
    """
    Verify fetching data from chemblDB when the input is a cudf df.
    """
    context = _create_context()

    n_molecules, dao, mol_df = _fetch_chembl_test_dataset()
    wf = GpuKmeansUmap(n_molecules=n_molecules,
                       dao=dao, pca_comps=64)
    mol_df = mol_df.compute()
    wf.cluster(df_mol_embedding=mol_df)


def test_add_molecule():
    """
    Verify fetching data from chemblDB when the input is a cudf df.
    """
    context = _create_context()

    n_molecules, dao, mol_df = _fetch_chembl_test_dataset()

    if hasattr(mol_df, 'compute'):
        mol_df = mol_df.compute()

    mol_df = cudf.from_pandas(mol_df)
    n_molecules = mol_df.shape[0]

    # test mol should container aviable and new molecules
    test_mol = mol_df[n_molecules - 20:]
    mols_tobe_added = test_mol['id'].to_array().tolist()

    chData = ChEmblData()
    logger.info('Fetching ChEMBLLE id for %s', mols_tobe_added)
    mols_tobe_added = [str(row[0]) for row in chData.fetch_chemblId_by_molregno(mols_tobe_added)]
    logger.info('ChEMBL ids to be added %s', mols_tobe_added)

    # Molecules to be used for clustering
    mol_df = mol_df[:n_molecules - 10]

    wf = GpuKmeansUmap(n_molecules=n_molecules, dao=dao, pca_comps=64)
    wf.cluster(df_mol_embedding=mol_df)

    missing_mols, molregnos, df_embedding = wf.add_molecules(mols_tobe_added)
    assert len(missing_mols) == 10, 'Expected 10 missing molecules found %d' % len(missing_mols)

    # TODO: Once the issue with add_molecule in multi-gpu env. is fixed, the
    # number of missing_molregno found should be 0
    missing_mols, molregnos, df_embedding = wf.add_molecules(mols_tobe_added)
    assert len(missing_mols) == 0, 'Expected no missing molecules found %d' % len(missing_mols)
    # assert len(missing_mols) == 10, 'Expected 10 missing molecules found %d' % len(missing_mols)
