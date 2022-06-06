#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import logging
import pynvml as nv

from subprocess import run, PIPE

from nemo.core.config import hydra_runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('megamolbart')
formatter = logging.Formatter('%(asctime)s %(name)s [%(levelname)s]: %(message)s')


def _fetch_gpu_counts():
    nv.nvmlInit()
    return nv.nvmlDeviceGetCount()

def _set_cuda_device():
    """
    Fetch the container ID.
    """
    result = run(['bash', '-c',
                    'docker ps -a --format "table {{.ID}}\t{{.Names}}" | grep $HOSTNAME'],
                    stdout=PIPE, stderr=PIPE)
    result_lines = result.stdout.decode("utf-8")
    logger.info(f'Container info result: {result_lines}')
    if result.returncode != 0:
        logger.warning(f'Using default GPU device')
    else:
        container_id = int(result_lines.split('_')[-1])
        gpu_cnt = _fetch_gpu_counts()
        if gpu_cnt > 1:
            if container_id is not None:
                gpu_to_use = container_id % gpu_cnt
                logger.info(f'Using GPU {gpu_to_use}')
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_to_use)

_set_cuda_device()


import pathlib
import grpc
import generativesampler_pb2_grpc

from concurrent import futures
from megamolbart.service import GenerativeSampler


class Launcher(object):
    """
    Application launcher. This class can execute the workflows in headless (for
    benchmarking and testing) and with UI.
    """

    def __init__(self, cfg):
        logger.setLevel(logging.DEBUG)

        logger.info(f'Using checkpoint: {cfg.model.model_path}')

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server(
            GenerativeSampler(cfg),
            server)
        server.add_insecure_port(f'[::]:{cfg.service.port}')
        server.start()
        server.wait_for_termination()

    def _fetch_model_path(self, search_loc='/models'):
        """
        Fetch the model path from the model server.
        """
        checkpoints = sorted(pathlib.Path(search_loc).glob('**/*.nemo'))

        if not checkpoints or len(checkpoints) == 0:
            logger.info(f'Model not found. Downloading...')
            self.download_megamolbart_model()
            checkpoints = sorted(pathlib.Path(search_loc).glob('**/*.nemo'))
        logger.info(f'Found {len(checkpoints)} checkpoints in {search_loc}')

        checkpoint_dir = checkpoints[-1].absolute().parent.as_posix()
        return checkpoint_dir, checkpoints[-1]

    def download_megamolbart_model(self):
        """
        Downloads MegaMolBART model from NGC.
        """
        download_script = '/opt/nvidia/cheminfomatics/cuchemcommon/launch'
        if os.path.exists(download_script):
            logger.info('Triggering model download...')
            result = run(['bash', '-c',
                          'cd /opt/nvidia/cheminfomatics/cuchemcommon/ && /opt/nvidia/cheminfomatics/cuchemcommon/launch download_model'])
            logger.info(f'Model download result: {result.stdout}')
            logger.info(f'Model download result: {result.stderr}')
            if result.returncode != 0:
                raise Exception('Error downloading model')


@hydra_runner(config_path="../conf", config_name="default")
def main(cfg):
    Launcher(cfg)


if __name__ == '__main__':
    main()
