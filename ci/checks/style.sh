#!/bin/bash

#
# Copyright 2019-2020 NVIDIA CORPORATION.
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
#


# Ignore errors and set path
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

PATH=/conda/bin:$PATH

# Set home to the job's workspace
export HOME=$WORKSPACE

cd ${WORKSPACE}

# Run copyright header check
logger "Run Copyright header check..."
./ci/checks/check_copyright.py

# Python
logger "Run Python/Cython formatting check..."
python -m pip install -r ./ci/commons/python-style-requirements.txt
source ci/commons/style_check
