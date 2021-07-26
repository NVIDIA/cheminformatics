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
import re

# ZINC dataset parameters
ZINC_CSV_DIR = '/data/zinc_csv'

# ZINC trie parameters
ZINC_TRIE_DIR = '/data/zinc_trie'
TRIE_FILENAME_LENGTH = 10
SHORT_TRIE_FILENAME = 'SHORT_SMILES'
TRIE_FILENAME_REGEX = re.compile(
    r'[/\\]+')  # re.compile(r'[^\w]+') # Alternative to strip all non-alphabet/non-numerical characters


def generate_trie_filename(smiles):
    """Generate appropriate filename for the trie"""

    # TODO smiles string should be cleaned before testing length -- will require regeneration of trie index
    filename_extractor = lambda x: re.sub(TRIE_FILENAME_REGEX, '', x)[:TRIE_FILENAME_LENGTH]

    if len(smiles) < TRIE_FILENAME_LENGTH:
        filename = SHORT_TRIE_FILENAME
    else:
        filename = filename_extractor(smiles)

    return filename + '.txt'
