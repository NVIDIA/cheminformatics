import logging
import sqlite3
from contextlib import closing

from cuchem.benchmark.data import ZINC15TrainData
from cuchemcommon.data.helper.chembldata import ChEmblData

logger = logging.getLogger(__name__)


def test_training_data_megatron_molbart():
    training_data = ZINC15TrainData()

    cursor = training_data.conn.cursor()
    cursor.execute('SELECT smiles FROM train_data limit 10')
    smiles_strs = cursor.fetchall()

    for smiles in smiles_strs:
        logger.info(f'Looking for {smiles} in known smiles database...')
        assert training_data.is_known_smiles(smiles[0]) == True

    smiles = 'adasdadsasdasd'
    logger.info(f'Looking for {smiles} in known smiles database...')
    assert training_data.is_known_smiles(smiles) == False



def test_training_data_cdd():
    chemble_db = ChEmblData(fp_type=None)

    with closing(sqlite3.connect(chemble_db.chembl_db, uri=True)) as con:
        assert chemble_db.is_valid_chemble_smiles('sabcd', con) == False

        assert chemble_db.is_valid_chemble_smiles(
            'Cc1ccc(C(=O)c2ccc(-n3ncc(=O)[nH]c3=O)cc2)cc1',
            con) == True
