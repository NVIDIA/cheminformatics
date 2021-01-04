#!/bin/bash
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

BENCHMARK_FILE='benchmark.csv'
CACHE_DIR='/data/db'

if [ -e ${BENCHMARK_FILE} ]; then
    rm ${BENCHMARK_FILE}
fi

# if [ -e ${CACHE_DIR} ]; then
#     rm ${CACHE_DIR}/*.hdf5
#     ./startdash.py cache -c ${CACHE_DIR}
# fi

./startdash.py analyze -b --cache ${CACHE_DIR} --n_gpu 1
./startdash.py analyze -b --cache ${CACHE_DIR} --n_gpu 2
./startdash.py analyze -b --cache ${CACHE_DIR} --n_gpu 4
./startdash.py analyze -b --cache ${CACHE_DIR} --n_gpu 6
./startdash.py analyze -b --cache ${CACHE_DIR} --n_gpu 7

./startdash.py analyze -b --cache ${CACHE_DIR} --cpu --n_cpu 12

