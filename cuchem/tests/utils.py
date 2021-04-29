import os
import dask
import time
import inspect
import tempfile
import logging

from locust import events

from nvidia.cheminformatics.utils.dask import initialize_cluster
from cuchemcommon.context import Context


logger = logging.getLogger(__name__)


def _fetch_chembl_test_dataset(n_molecules=None):
    if n_molecules is None:
        n_molecules = 1000

    from cuchemcommon.data.cluster_wf import ChemblClusterWfDao
    from nvidia.cheminformatics.fingerprint import MorganFingerprint

    dao = ChemblClusterWfDao(MorganFingerprint)
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


def stopwatch(request_type):
    def _stopwatch(func):

        def wrapper(*args, **kwargs):
            previous_frame = inspect.currentframe().f_back
            _, _, task_name, _, _ = inspect.getframeinfo(previous_frame)

            _start_time = time.time()
            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                total = int((time.time() - _start_time) * 1000)
                events.request_failure.fire(request_type=request_type,
                                            name=task_name,
                                            response_time=total,
                                            response_length=0,
                                            exception=e)
            else:
                total = int((time.time() - _start_time) * 1000)
                events.request_success.fire(request_type=request_type,
                                            name=task_name,
                                            response_time=total,
                                            response_length=0)
            print(total, _start_time, time.time())
            return result

        return wrapper
    return _stopwatch
