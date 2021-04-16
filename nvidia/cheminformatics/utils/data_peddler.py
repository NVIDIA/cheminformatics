import os
import logging
import shutil
from subprocess import run, Popen


CDDD_DEFAULT_MODLE_LOC = '/model/cddd'


logger = logging.getLogger(__name__)


def download_cddd_models(target_dir=CDDD_DEFAULT_MODLE_LOC):
    """
    Downloads CDDD model
    """

    if os.path.exists(os.path.join(target_dir, 'default_model', 'hparams.json')):
        logger.warning('Directory already exists. To re-download please delete %s', target_dir)
        return os.path.join(target_dir, 'default_model')
    else:
        shutil.rmtree(os.path.join(target_dir, 'default_model'), ignore_errors=True)

    download_script = '/opt/cddd/download_default_model.sh'
    if not os.path.exists(download_script):
        download_script = '/tmp/download_default_model.sh'
        run(['bash', '-c',
             'wget --quiet -O %s https://raw.githubusercontent.com/jrwnter/cddd/master/download_default_model.sh && chmod +x %s' % (download_script, download_script)])

    run(['bash', '-c',
         'mkdir -p %s && cd %s; %s' % (target_dir, target_dir, download_script)],
        check=True)

    return os.path.join(target_dir, 'default_model')