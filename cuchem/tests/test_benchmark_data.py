import logging
from cuchem.benchmark.data import TrainingData

logger = logging.getLogger(__name__)


def test_training_data():
    training_data = TrainingData()

    cursor = training_data.conn.cursor()
    cursor.execute('SELECT smiles FROM train_data limit 10')
    smiles_strs = cursor.fetchall()

    for smiles in smiles_strs:
        logger.info(f'Looking for {smiles} in known smiles database...')
        assert training_data.is_known_smiles(smiles[0]) == True

    smiles = 'adasdadsasdasd'
    logger.info(f'Looking for {smiles} in known smiles database...')
    assert training_data.is_known_smiles(smiles) == False