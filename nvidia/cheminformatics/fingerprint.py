import logging

from datetime import datetime

from dask.distributed import Client, LocalCluster
from nvidia.cheminformatics.chembldata import ChEmblData


logger = logging.getLogger(__name__)


def save_fingerprints(hdf_path='data/filter_*.h5'):
    """
    Generates fingerprints for all ChEmblId's in the database
    """
    logger.info('Fetching molecules from database for fingerprints...')

    chem_data = ChEmblData()
    mol_df = chem_data.fetch_all_props()

    mol_df.to_hdf(hdf_path, 'fingerprints')


if __name__=='__main__':
    logger = logging.getLogger('nv_chem_viz')
    formatter = logging.Formatter(
            '%(asctime)s %(name)s [%(levelname)s]: %(message)s')

    logger.info('Starting dash cluster...')
    cluster = LocalCluster(dashboard_address=':9001',
                           n_workers=12,
                           threads_per_worker=4)
    client = Client(cluster)
    logger.info(client)
    logger.info('Fetching molecules from database for fingerprints...')

    task_start_time = datetime.now()
    save_fingerprints('data/filter_*.h5')
    logger.info('Fingerprint generated in (hh:mm:ss.ms) {}'.format(
        datetime.now() - task_start_time))
