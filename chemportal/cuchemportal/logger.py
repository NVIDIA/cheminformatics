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

import logging
import sys

logger = logging.getLogger(__name__)


def initialize_log(log_file, logger_name):
    """
    Initializes logger.

    :param log_file:
    :param logger_name:
    :return:
    """
    logger = logging.getLogger(logger_name)

    # init the logger to print to console
    console_handle = logging.StreamHandler(sys.stdout)
    console_handle.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s [%(levelname)s]: %(message)s')
    console_handle.setFormatter(formatter)

    logger.addHandler(console_handle)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        file_handle = logging.FileHandler(log_file)
        file_handle.setLevel(logging.DEBUG)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    except IOError as ex:
        logger.error("Could not create log file handler")
        logger.exception(ex)

    return logger
