import cupy
import numpy
from typing import List

from cuml.metrics import pairwise_distances

from nvidia.cheminformatics.data.helper.chembldata import ADDITIONAL_FEILD, IMP_PROPS
from nvidia.cheminformatics.utils.distance import tanimoto_calculate
from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores, spearmanr


class BaseClusterWorkflow:

    def _remove_ui_columns(self, embedding):
        for col in ['x', 'y', 'cluster', 'filter_col', 'index', 'molregno']:
            if col in embedding.columns:
                embedding = embedding.drop([col], axis=1)

        return embedding

    def _remove_non_numerics(self, embedding):
        embedding = self._remove_ui_columns(embedding)

        other_props = ['id'] +  IMP_PROPS + ADDITIONAL_FEILD
        # Tempraryly store columns not required during processesing
        prop_series = {}
        for col in other_props:
            if col in embedding.columns:
                prop_series[col] = embedding[col]
        if len(prop_series) > 0:
            embedding = embedding.drop(other_props, axis=1)

        return embedding, prop_series


    def _compute_spearman_rho(self, embedding, X_train, Xt):
        n_indexes = min(self.n_spearman, X_train.shape[0])
        numpy.random.seed(self.seed)
        indexes = numpy.random.choice(numpy.array(range(X_train.shape[0])),
                                      size=n_indexes,
                                      replace=False)
        fp_sample = embedding.compute().values[indexes]
        Xt_sample = Xt.compute().values[indexes]

        dist_array_tani = tanimoto_calculate(fp_sample, calc_distance=True)
        dist_array_eucl = pairwise_distances(Xt_sample)
        return cupy.nanmean(spearmanr(dist_array_tani, dist_array_eucl, top_k=100))

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
