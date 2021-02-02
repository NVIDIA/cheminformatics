from typing import List

class ClusterWfDAO(object):
    """
    Base class for all DAO for fetching data for Clustering Workflows
    """

    def fetch_molecular_embedding(self, n_molecules:int, cache_directory:str=None):
        """
        Fetch molecular properties from database/cache into a dask array.
        """
        pass

    def fetch_new_molecules(self, new_molecules: List):
        """
        Fetch molecular details for a list of molecules. The values in the list
        of molecules depends on database/service used. For e.g. it could be
        ChemblId or molreg_id for Chemble database.
        """
        pass
