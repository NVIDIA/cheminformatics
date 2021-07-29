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
from argparse import ArgumentParser
from waitress import serve
from .api import app
from .logger import initialize_log

logger = initialize_log(os.path.join('/var/log', 'cuChemPortal.log'), 'cuChemPortal')


def init_arg_parser():
    """
    Constructs command-line argument parser definition.
    """
    parser = ArgumentParser(description='cuChem Portal')

    parser.add_argument('-c', '--configFile',
                        dest='configFile',
                        nargs='+',
                        type=str,
                        required=False,
                        help='Application configuration file')
    return parser


def main():
    """
    Entry point to pipeline executor or the UI renderer.
    """
    logger.info('cuChem Portal')
    arg_parser = init_arg_parser()
    args = arg_parser.parse_args()

    config_file = '/etc/nvidia/cuChem/portalConfig.yaml'
    if hasattr(args, 'configFile') and args.configFile is not None:
        config_file = args.configFile

    logger.info(config_file)
    serve(app, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
