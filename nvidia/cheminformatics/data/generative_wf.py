from nvidia.cheminformatics.utils.singleton import Singleton
import logging

from typing import List

from . import GenerativeWfDao
from nvidia.cheminformatics.data.helper.chembldata import ChEmblData

logger = logging.getLogger(__name__)


class ChemblGenerativeWfDao(GenerativeWfDao, metaclass=Singleton):

    def fetch_id_from_chembl(self, id: List):
        logger.debug('Fetch ChEMBL ID using molregno...')
        chem_data = ChEmblData()
        return chem_data.fetch_id_from_chembl(id)
