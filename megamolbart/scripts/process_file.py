#!/usr/bin/env python3

import os
import sys
import math
import time
import random
import argparse
from subprocess import run

from rdkit import Chem
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


MAX_LENGTH = 150
VAL_TEST_SPLIT = 0.005


def process_line(line):
    if line is None or line == "":
        return None

    splits = line.split("\t")
    if len(splits) < 2:
        return None

    smi, zinc_id = splits[0], splits[1]

    try:
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol, canonical=True)
    except RuntimeError:
        return None

    if len(smi) > MAX_LENGTH:
        return None

    output = f"{zinc_id},{smi}"

    return output


def process_file_text(text, executor):
    lines = text.split("\n")

    # Filter and canonicalise molecules
    futures = [executor.submit(process_line, line) for line in lines]
    outputs = [future.result() for future in futures]
    output_lines = [output for output in outputs if output is not None]

    # Assign set
    mol_sets = ["train" for _ in range(len(output_lines))]
    num_idxs = math.ceil(len(output_lines) * VAL_TEST_SPLIT)

    val_idxs = random.sample(range(len(output_lines)), k=num_idxs)
    for idx in val_idxs:
        mol_sets[idx] = "val"

    rem_idxs = set(range(len(output_lines))) - set(val_idxs)
    test_idxs = random.sample(list(rem_idxs), k=num_idxs)
    for idx in test_idxs:
        mol_sets[idx] = "test"

    # Generate file output
    completed_lines = [f"{line},{dataset}" for line, dataset in zip(output_lines, mol_sets)]
    output_text = "\n".join(completed_lines)
    output_text = f"zinc_id,smiles,set\n{output_text}"
    return output_text


def format_arr_idx(arr_idx):
    return "x" + str(arr_idx).rjust(3, "0")


def process_file(zinc_dir, out_dir, arr_idx):
    cpus = len(os.sched_getaffinity(0))
    executor = ProcessPoolExecutor(cpus)
    print(f"Using a pool of {str(cpus)} processes for execution.")

    zinc_path = Path(zinc_dir)
    filename = format_arr_idx(arr_idx)
    file_path = zinc_path / filename

    print(f"Processing file {str(file_path)}...")
    text = file_path.read_text()
    output_text = process_file_text(text, executor)
    print("Successfully processed file.")

    out_dir = Path(out_dir)
    out_path = out_dir / (filename + ".csv")
    out_path.write_text(output_text)
    print(f"Successfully written to {str(out_path)}")


def main(args):
    start_time = time.time()

    if args.download_list:
        # Download zinc files and perform all pre-processing steps. Following
        # are preprocessing steps:
        # - Merge all records
        # - Shuffle records randomly
        # - Spilt 100000 recs per file
        start = time.time()
        print('Downloading zinc database...')
        download_cmd = f'parallel -j {args.threads} --gnu "wget -q --no-clobber {{}} -P {args.download_dir}" < <(cat {args.download_list})'
        print(download_cmd)
        process = run(['bash', '-c', download_cmd])

        if process.returncode != 0:
            print('ERROR downloading zinc database. Please make sure "parallel" is installed and check disk space.')
            sys.exit(process.returncode)
        print('Download complete. Time ', time.time() - start)

        start = time.time()
        shuffled_data = args.zinc_dir
        split_cmd = f"mkdir -p {shuffled_data}; cd {shuffled_data}; tail -q -n +2 {args.download_dir}/** | cat | shuf | split -d -l 10000000 -a 3"
        process = run(['bash', '-c', split_cmd])

        if process.returncode != 0:
            print('ERROR downloading zinc database. Please make sure "parallel" is installed and check disk space.')
            sys.exit(process.returncode)
        print('Shuffling and spliting files complete. Time ', time.time() - start)


    print("Processing files...")
    process_file(args.zinc_dir, args.output_dir, args.arr_idx)
    print("Finished processing.")

    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zinc_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--arr_idx", type=str)
    parser.add_argument("--download_list", type=str, default=None)
    parser.add_argument("--download_dir", type=str, default=None)
    parser.add_argument("-t", "--threads", type=int, default=8)

    print("Running ZINC pre-processing script...")
    args = parser.parse_args()
    main(args)
    print("Complete.")
