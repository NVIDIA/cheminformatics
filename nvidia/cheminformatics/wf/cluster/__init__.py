from typing import List


class BaseClusterWorkflow:

    def is_gpu_enabled(self):
        return True

    def cluster(self, embedding):
        """
        Runs clustering workflow on the data fetched from database/cache.
        """
        NotImplemented

    def recluster(self,
                  filter_column=None,
                  filter_values=None,
                  n_clusters=None):
        """
        Runs reclustering on original dataframe or on the new dataframe passed.
        The new dataframe is usually a subset of the original dataframe.
        Caller may ask to include additional molecules.
        """
        NotImplemented

    def add_molecules(self, chemblids:List):
        """
        ChembleId's accepted as argument to the existing database. Duplicates
        must be ignored.
        """
        NotImplemented

    def compute_qa_matric(self):
        """
        Collects all quality matrix and log.
        """
        NotImplemented
