import os
import logging
import shutil
from subprocess import run

CDDD_MODEL_SCRIPT = 'https://raw.githubusercontent.com/jrwnter/cddd/master/download_default_model.sh'

log = logging.getLogger(__name__)

def download_cddd_models():
    """
    Downloads CDDD model
    """

    target_dir = os.path.join('/data', 'mounts', 'cddd')
    if os.path.exists(os.path.join(target_dir, 'default_model', 'hparams.json')):
        log.warning('Directory already exists. To re-download please delete %s', target_dir)
        return os.path.join(target_dir, 'default_model')
    else:
        shutil.rmtree(os.path.join(target_dir, 'default_model'), ignore_errors=True)

    download_script = '/opt/nvidia/cheminfomatics/cddd/download_default_model.sh'
    if not os.path.exists(download_script):
        download_script = '/tmp/download_default_model.sh'
        run(['bash', '-c',
             'wget --quiet -O %s %s && chmod +x %s' % (download_script, CDDD_MODEL_SCRIPT, download_script)])

    run(['bash', '-c',
         'mkdir -p %s && cd %s; %s' % (target_dir, target_dir, download_script)],
        check=True)

    return os.path.join(target_dir, 'default_model')


class Singleton(type):
    """
    Ensures single instance of a class.

    Example Usage:
        class MySingleton(metaclass=Singleton)
            pass
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]
