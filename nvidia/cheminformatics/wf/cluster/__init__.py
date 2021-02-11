class BaseClusterWorkflow:

    def is_gpu_enabled(self):
        return True

    def cluster(self, embedding):
        """
        Runs clustering workflow on the data fetched from database/cache.
        """
        pass

    def recluster(self, new_df=None, new_molecules=None):
        """
        Runs reclustering on original dataframe or on the new dataframe passed.
        The new dataframe is usually a subset of the original dataframe.
        Caller may ask to include additional molecules.
        """
        pass

    def compute_qa_matric(self):
        """
        Collects all quality matrix and log.
        """
        pass

