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

# Parallel processing of ZINC15 data files to create a trie

import glob
import logging
import multiprocessing
import os
from pathlib import Path

import pandas as pd
from cuchem.utils.dataset import ZINC_CSV_DIR, \
                                 ZINC_TRIE_DIR, \
                                 generate_trie_filename, \
                                 TRIE_FILENAME_LENGTH, \
                                 SHORT_TRIE_FILENAME

### SETTINGS ###
LOG_PATH = os.path.join(ZINC_TRIE_DIR, 'processing.log')
FILE_LOG_PATH = os.path.join(ZINC_TRIE_DIR, 'processed_files.txt')

# Use number of processes and queue size to balance memory
# so there are just slightly more items in queue than processes
NUM_PROCESSES = (multiprocessing.cpu_count() * 2) - 1  # --> max num proceses, but needs more memory
QUEUE_SIZE = int(1e5)


def load_data(fil, trie_filename_length, short_trie_filename):
    """Load data as a pandas dataframe"""

    data = pd.read_csv(fil, usecols=['smiles', 'set'])
    data['filename'] = data['smiles'].map(generate_trie_filename)
    data['filename'] = data['set'] + '/' + data['filename']
    data.drop('set', axis=1, inplace=True)
    data = data.set_index('filename').sort_index()

    return data


def listener(queue, filelist, trie_filename_length, short_trie_filename):
    """
    Add batches to the queue
    Params:
        queue: multiprocessing queue of batches
        filelist: list of filenames to import
        trie_filename_length: integer length of filename to use for trie
        short_trie_filename: name to use for molecules shorter than minimum length
    """

    chunksize = 100
    logger = multiprocessing.get_logger()
    data_cleanup = lambda x: (x[0], x[1]['smiles'].tolist())

    for fil in filelist:
        logger.info(f'Reading {fil}')
        data = load_data(fil, trie_filename_length, short_trie_filename)

        data_grouper = [data_cleanup(x) for x in data.groupby(level=0)]
        num_groups = len(data_grouper)
        data_grouper = [data_grouper[i: i + chunksize] for i in range(0, len(data_grouper), chunksize)]
        num_chunks = len(data_grouper)
        logger.info(f'Finished processing {fil} with {num_groups} groups into {num_chunks} chunks')

        with open(FILE_LOG_PATH, 'a+') as fh:
            fh.write(fil + '\n')

        for chunk in data_grouper:
            queue.put(chunk)

    # queue.put('DONE')
    return


def process_data(base_filename, smiles_list, output_dir, lock):
    """Write SMILES to files.
    """
    logger = multiprocessing.get_logger()
    filename = os.path.join(output_dir, base_filename)

    num_entries = len(smiles_list)
    if num_entries >= 100:
        logger.info(f'Working on {filename} with {num_entries} entries')

    chunksize = 100
    smiles_list = [smiles_list[i: i + chunksize] for i in range(0, len(smiles_list), chunksize)]

    lock.acquire()
    with open(filename, 'a+') as fh:
        for sl in smiles_list:
            fh.write('\n'.join(sl) + '\n')
    lock.release()

    # if num_entries >= 100:
    #     logger.info(f'Saved {filename} with {num_entries} entries')

    return


def worker(queue, lock, output_dir):
    """
    Process batches of data from the queue
    Params:
        queue: multiprocessing queue of batches
        lock: ensure only one process modifies the file at a time
    """
    logger = multiprocessing.get_logger()

    while True:
        batch = queue.get(True)
        for data in batch:
            filename, smiles_list = data
            process_data(filename, smiles_list, output_dir, lock)


if __name__ == '__main__':

    for subdir in ['train', 'val', 'test']:
        dir = os.path.join(ZINC_TRIE_DIR, subdir)
        Path(dir).mkdir(parents=True, exist_ok=True)

    # Setup logging that is compatible with multiprocessing
    multiprocessing.log_to_stderr()
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Gather list of filenames
    filelist = sorted(glob.glob(os.path.join(ZINC_CSV_DIR, '*.csv')))
    logger.info(ZINC_CSV_DIR)
    logger.info(filelist)
    import sys
    sys.exit(0)
    n_files = len(filelist)
    logger.info(f'Identified {n_files} files')

    # Setup worker for multiprocessing
    manager = multiprocessing.Manager()
    queue = manager.Queue(QUEUE_SIZE)
    logger.info(f'Starting queue with maximum size of {QUEUE_SIZE}')
    producer = multiprocessing.Process(target=listener,
                                       args=(queue, filelist, TRIE_FILENAME_LENGTH, SHORT_TRIE_FILENAME))
    producer.start()

    # Setup listener
    logger.info(f'Starting {NUM_PROCESSES} listeners')
    pool = multiprocessing.Pool(NUM_PROCESSES)
    lock = manager.Lock()

    results = []
    for id_ in range(NUM_PROCESSES):
        results.append(pool.apply_async(worker, args=(queue, lock, ZINC_TRIE_DIR)))

    producer.join()
    pool.terminate()
    logger.info(f'Finished processing {n_files} files.')
