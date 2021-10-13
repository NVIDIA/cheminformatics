#! /usr/bin/env python3
import os
import sys
import csv
from dask import dataframe as dd
from dask.distributed import LocalCluster, Client
import sqlite3 as lite

def upload(path='zinc_csv_split/train/*.csv'):
    print(f'Loading data from {path}...')
    db = 'sqlite:///zinc_train.sqlite3'

    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster, asynchronous=True)

    zinc_data = dd.read_csv(path)
    zinc_data = zinc_data.head()
    zinc_data.to_sql('train_data', db)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Please provide data location')
    
    upload(path=sys.argv[1])
