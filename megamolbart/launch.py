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
import atexit
import logging

import logging
import warnings
import argparse

from datetime import datetime

import grpc
import generativesampler_pb2_grpc
from concurrent import futures
from megamolbart.service import GenerativeSampler


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('megamolbart')
formatter = logging.Formatter(
    '%(asctime)s %(name)s [%(levelname)s]: %(message)s')


class Launcher(object):
    """
    Application launcher. This class can execute the workflows in headless (for
    benchmarking and testing) and with UI.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description='Mega-molbart gRPC Service')
        parser.add_argument('-p', '--port',
                            dest='port',
                            type=int,
                            default=50051,
                            help='GRPC server Port')
        parser.add_argument('-d', '--debug',
                            dest='debug',
                            action='store_true',
                            default=False,
                            help='Show debug message')

        args = parser.parse_args(sys.argv[2:])

        if args.debug:
            logger.setLevel(logging.DEBUG)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        # generativesampler_pb2_grpc.add_SimilaritySamplerServicer_to_server(GenerativeSampler(), server)
        # server.add_insecure_port(f'[::]:{args.port}')
        # server.start()
        # server.wait_for_termination()


def main():
    Launcher()


if __name__ == '__main__':
    main()
