ZINC Dataset

This dataset was downloaded on 24/02/2021 and contains approximately 1.463
billion molecules as SMILES strings with ZINC ids.

The raw_data directory contains the files as downloded from zinc15.docking.org.
The zinc-download-smiles.wget contains a list of wget commands used to download
the files - each command corresponds to one file. The command file was downloded
from the ZINC tranch tool, by selecting Reactivity = reactive,
Purchasability = annotated and removing molecules with molecular weight > 500 or
LogP > 5 (the rightmost and bottommost edges, respectively, of the matrix on
zinc15.docking.org/tranches/home).

The following files, however, were not available at the time of download, and
are therefore ommited from the raw_data directory:
BA/BAGC.smi
HA/HAAC.smi
CB/CBBC.smi
FB/FBCC.smi
IC/ICCC.smi
FD/FDEC.smi
HD/HDBC.smi
JG/JGBC.smi
JG/JGEC.smi
JH/JHEC.smi
II/IIBC.smi
II/IICC.smi

The files were downloaded using the download_zinc.py Python script in this
directory (and the download.sub slurm file, which was used to submit the compute
cluster).

The raw data files are then processed using the following command (from shuffled_data/ dir):
tail -q -n +2 ../raw_data/*/** | cat | shuf | split -d -l 10000000 -a 3

This produces 146 files with 10000000 molecules in each (except the last) with
no header in the file.

Finally, the file processed (canonicalised and dataset added) using the
process_file.py in a slurm array. Any SMILES that did not canonicalise or were
longer than 150 characters were stripped out. The final data is stored in
processed_data/

The folders raw_data/, shuffled_data/ and processed_data/ have been archived
using tar and compressed using lz4 compression and stored in raw.tar.lz4,
shuffled.tar.lz4 and processed.tar.lz4, respectively, to save disk space and
allow faster copying.
