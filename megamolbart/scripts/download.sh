#!/bin/bash

FILE="ZINC-downloader-AZ.txt"
N_THREADS=8

cat $FILE | parallel -j $N_THREADS --gnu "wget --no-clobber {}"
