#!/opt/conda/envs/rapids/bin/python3
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

import math
import cupy
from numba import cuda


@cuda.jit
def compute_norms(data, norms):
    """Compute norms

    Args:
        data (matrix): matrix with data and samples in rows
        norms (matrix): matrix for norms
    """
    i = cuda.grid(1)
    norms[i] = len(data[i])
    for j in range(len(data[i])):
        if data[i][j] != 0:
            value = j + 1
            data[i][j] = value
            norms[i] = norms[i] + (value**2)
    if norms[i] != 0:
        norms[i] = math.sqrt(norms[i])


@cuda.jit
def compute_tanimoto_matix(data, norms, dist_array, calc_distance=False):
    """Numba kernel to calculate

    Args:
        data (matrix): data with samples in rows
        norms (matrix): matrix with samples in rows
        dist_array (matrix): square matrix to hold pairwise distance
        calc_distance (bool, optional): Calculate distance metric. Defaults to False.
    """
    x = cuda.grid(1)
    rows = len(data)

    i = x // rows
    j = x % rows

    if i == j:
        dist_array[i][j] = 0 if calc_distance else 1
        return

    a = data[i]
    b = data[j]

    prod = 0
    for k in range(len(a)):
        prod = prod + (a[k] * b[k])

    a_norm = norms[i]
    b_norm = norms[j]

    tanimoto_calc = (prod / ((a_norm**2 + b_norm**2) - prod))
    tanimoto_calc = 1 - tanimoto_calc if calc_distance else tanimoto_calc
    dist_array[i][j] = tanimoto_calc


@cuda.jit
def compute_rdkit_tanimoto_matix(data, dist_array, calc_distance=False):
    x = cuda.grid(1)
    rows = len(data)

    i = x // rows
    j = x % rows

    if i == j:
        dist_array[i][j] = 0 if calc_distance else 1
        return

    a = data[i]
    b = data[j]

    intersections = 0
    total = 0
    for k in range(len(a)):
        if a[k] and b[k]:
            intersections += 1
            total += 2
        elif a[k] or b[k]:
            total += 1

    dist_array[i][j] = intersections / (total - intersections)


def tanimoto_calculate(fp, calc_distance=False):
    """Calculate tanimoto similarity or distance

    Args:
        fp (cupy array or cudf dataframe): fingerprints with samples in rows
        calc_distance (bool, optional): Calculate distance metric. Defaults to False.

    Returns:
        array: pairwise tanimoto distance
    """

    dist_array = cupy.zeros((fp.shape[0], fp.shape[0]), cupy.float32)
    compute_rdkit_tanimoto_matix.forall(fp.shape[0] * fp.shape[0], 1)(fp,
                                                                dist_array,
                                                                calc_distance)
    if calc_distance:
        dist_array = 1 - dist_array
    return dist_array
