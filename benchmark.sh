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

export PATH=/opt/conda/envs/rapids/bin:$PATH

BENCHMARK_DIR='/workspace/benchmark'
BENCHMARK_PATH=${BENCHMARK_DIR}/benchmark.csv
PLOT_PATH=${BENCHMARK_DIR}/benchmark.png
CLEAN_BENCHMARK=1

CACHE_DIR='/data/db'
REBUILD_CACHE=0

N_MOLECULES=(10000 20000 50000 100000 200000 -1)
N_GPUS=(1 2 4 6 7)
N_CPUS=(39)

if [[ $CLEAN_BENCHMARK -eq 1 ]]; then
    if [ -e ${BENCHMARK_DIR} ]; then
       rm -rf ${BENCHMARK_DIR}
    fi
fi

mkdir -p $BENCHMARK_DIR

if [[ $REBUILD_CACHE -eq 1 ]]; then
    if [ ! -e ${CACHE_DIR} ]; then
        mkdir -p ${CACHE_DIR}
    else
        rm ${CACHE_DIR}/*.hdf5
    fi
    python startdash.py cache -c ${CACHE_DIR}
fi

# Run the benchmarks
for n_molecules in ${N_MOLECULES[*]}; do
    for n_gpus in ${N_GPUS[*]}; do
        echo "Benchmarking $n_molecules molecules on $n_gpus GPUs"
        python startdash.py analyze -b --cache ${CACHE_DIR} --n_gpu $n_gpus --n_mol $n_molecules --output_dir $BENCHMARK_DIR
    done

    for n_cpus in ${N_CPUS[*]}; do
        echo "Benchmarking $n_molecules molecules on $n_cpus CPU cores"
        python startdash.py analyze -b --cache ${CACHE_DIR} --cpu --n_cpu $n_cpus --n_mol $n_molecules --output_dir $BENCHMARK_DIR
    done
done

# Plot the results
python nvidia/cheminformatics/utils/plot_benchmark_results.py --benchmark_file $BENCHMARK_PATH --output_path $PLOT_PATH
