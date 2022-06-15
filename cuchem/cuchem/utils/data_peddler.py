import logging
import os
import shutil
from subprocess import run
from cuchemcommon.context import Context

CDDD_MODEL_SCRIPT = 'https://raw.githubusercontent.com/jrwnter/cddd/master/download_default_model.sh'

logger = logging.getLogger(__name__)

# Depricated
def download_cddd_models():
    """
    Downloads CDDD model
    """

    context = Context()
    target_dir = context.get_config('data_mount_path', default='/data')
    target_dir = os.path.join(target_dir, 'mounts', 'cddd')

    if os.path.exists(os.path.join(target_dir, 'default_model', 'hparams.json')):
        logger.warning('Directory already exists. To re-download please delete %s', target_dir)
        return os.path.join(target_dir, 'default_model')
    else:
        shutil.rmtree(os.path.join(target_dir, 'default_model'), ignore_errors=True)

    # download_script = '/opt/cddd/download_default_model.sh'
    download_script = '/workspace/cddd/download_default_model.sh'
    if not os.path.exists(download_script):
        download_script = '/workspace/cddd/download_default_model.sh'
        run(['bash', '-c',
             'wget --quiet -O %s %s && chmod +x %s' % (download_script, CDDD_MODEL_SCRIPT, download_script)])

    run(['bash', '-c',
         'mkdir -p %s && cd %s; %s' % (target_dir, target_dir, download_script)],
        check=True)

    return os.path.join(target_dir, 'default_model')
