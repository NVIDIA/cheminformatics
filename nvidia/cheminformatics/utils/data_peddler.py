import os
import logging
from subprocess import run, Popen


CDDD_DEFAULT_MODLE_LOC = '/data/cddd'


logger = logging.getLogger(__name__)


def download_cddd_models(target_dir=CDDD_DEFAULT_MODLE_LOC):
    """
    Downloads CDDD model
    """

    if os.path.exists(target_dir):
        logger.warning('Directory already exists. To re-download please delete %s', target_dir)
        return os.path.join(target_dir, 'default_model')

    run(['bash', '-c',
         'mkdir -p %s && cd %s; %s' % (target_dir, target_dir, '/opt/cddd/download_default_model.sh')],
        check=True)

    return os.path.join(target_dir, 'default_model')