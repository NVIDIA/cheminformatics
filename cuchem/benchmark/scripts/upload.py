import sqlite3

from dask import dataframe as dd
from dask.distributed import LocalCluster, Client

def upload():
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster, asynchronous=True)

    path = '/clara/testData/chemInformatics/data/zinc_split/train/*.csv'
    db = 'sqlite:///zinc_train.db'

    print('Uploading data...')
    zinc_data = dd.read_csv(path)
    zinc_data.to_sql('train_data', db)

    conn = sqlite3.connect('zinc_train.db')
    cursor = conn.cursor()
    print('Creating index...')
    cursor.execute(
        '''CREATE INDEX IF NOT EXISTS smiles_index ON train_data (smiles)''')
    cursor.close()

if __name__ == '__main__':
    upload()
