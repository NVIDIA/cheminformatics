import logging

from nvidia.cheminformatics.data import GenerativeWfDao
from nvidia.cheminformatics.data.generative_wf import ChemblGenerativeWfDao
from nvidia.cheminformatics.utils.singleton import Singleton
from nvidia.cheminformatics.wf.generative import BaseGenerativeWorkflow


logger = logging.getLogger(__name__)


class MolBART(BaseGenerativeWorkflow):

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao()) -> None:
        pass
