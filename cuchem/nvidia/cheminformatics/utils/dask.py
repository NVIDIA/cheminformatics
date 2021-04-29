import logging

import cupy

from dask_cuda.local_cuda_cluster import cuda_visible_devices
from dask_cuda.utils import get_n_gpus
from dask_cuda import initialize, LocalCUDACluster
from dask.distributed import Client, LocalCluster


logger = logging.getLogger(__name__)


def initialize_cluster(use_gpu=True, n_cpu=None, n_gpu=-1):
    enable_tcp_over_ucx = True
    enable_nvlink = True
    enable_infiniband = True

    logger.info('Starting dash cluster...')
    if use_gpu:
        initialize.initialize(create_cuda_context=True,
                              enable_tcp_over_ucx=enable_tcp_over_ucx,
                              enable_nvlink=enable_nvlink,
                              enable_infiniband=enable_infiniband)
        if n_gpu == -1:
            n_gpu = get_n_gpus()

        device_list = cuda_visible_devices(1, range(n_gpu)).split(',')
        CUDA_VISIBLE_DEVICES = []
        for device in device_list:
            try:
                CUDA_VISIBLE_DEVICES.append(int(device))
            except ValueError as vex:
                logger.warn(vex)

        logger.info('Using GPUs {} ...'.format(CUDA_VISIBLE_DEVICES))

        cluster = LocalCUDACluster(protocol="ucx",
                                   dashboard_address=':8787',
                                   CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
                                   enable_tcp_over_ucx=enable_tcp_over_ucx,
                                   enable_nvlink=enable_nvlink,
                                   enable_infiniband=enable_infiniband)
    else:
        logger.info('Using {} CPUs ...'.format(n_cpu))
        cluster = LocalCluster(dashboard_address=':8787',
                                n_workers=n_cpu,
                                threads_per_worker=4)

    client = Client(cluster)
    client.run(cupy.cuda.set_allocator)
    return client
