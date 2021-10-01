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
import sys
import shutil
import logging
import argparse
import pathlib

import grpc
import generativesampler_pb2_grpc

from concurrent import futures
from megamolbart.service import GenerativeSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('megamolbart')
formatter = logging.Formatter('%(asctime)s %(name)s [%(levelname)s]: %(message)s')


class Launcher(object):
    """
    Application launcher. This class can execute the workflows in headless (for
    benchmarking and testing) and with UI.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description='MegaMolBART gRPC Service')
        parser.add_argument('-p', '--port',
                            dest='port',
                            type=int,
                            default=50051,
                            help='GRPC server Port')
        parser.add_argument('-d', '--debug',
                            dest='debug',
                            action='store_true',
                            default=False,
                            help='Show debug messages')

        parser.add_argument('-m', '--model_path',
                            dest='model_path',
                            default=None,
                            help='Path to model content.')

        args = parser.parse_args(sys.argv[1:])

        if args.debug:
            logger.setLevel(logging.DEBUG)

        if args.model_path is None:
            model_dir = self._fetch_model_path()
        else:
            model_dir = args.model_path
        logger.info(f'Using checkpoint: {model_dir}')

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server(
            GenerativeSampler(model_dir=model_dir),
            server)
        server.add_insecure_port(f'[::]:{args.port}')
        server.start()
        server.wait_for_termination()

    def _fetch_model_path(self, search_loc='/models'):
        """
        Fetch the model path from the model server.
        """
        checkpoints = sorted(pathlib.Path(search_loc).glob('**/megamolbart_checkpoint.nemo'))
        logger.info(f'Found {len(checkpoints)} checkpoints in {search_loc}')
        
        if not checkpoints or len(checkpoints) == 0:
            raise Exception('Model not found')
        else:
            checkpoint_dir = checkpoints[-1].absolute().parent.as_posix()

            # TODO: This is a hack to place the vocab file where the model is expecting it.
            vocab_path = '/workspace/nemo/nemo/collections/chem/vocab/'
            os.makedirs(vocab_path, exist_ok=True)
            shutil.copy(os.path.join(checkpoint_dir, 'bart_vocab.txt'), 
                        os.path.join(vocab_path, 'megamolbart_pretrain_vocab.txt'))
            return checkpoint_dir

def main():
    Launcher()


if __name__ == '__main__':
    main()
