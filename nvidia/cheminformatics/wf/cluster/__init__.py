import cupy
import numpy
from typing import List

from cuml.metrics import pairwise_distances

from nvidia.cheminformatics.data.helper.chembldata import ADDITIONAL_FEILD, IMP_PROPS
from nvidia.cheminformatics.utils.distance import tanimoto_calculate
from nvidia.cheminformatics.utils.metrics import batched_silhouette_scores, spearmanr


class BaseClusterWorkflow:

    def __init__(self):
        self.df_embedding = None

    def _remove_ui_columns(self, embedding):
        for col in ['x', 'y', 'cluster', 'filter_col', 'index', 'molregno']:
            if col in embedding.columns:
                embedding = embedding.drop([col], axis=1)

        return embedding

    def _remove_non_numerics(self, embedding):
        embedding = self._remove_ui_columns(embedding)

        other_props = ['id'] + IMP_PROPS + ADDITIONAL_FEILD
        # Tempraryly store columns not required during processesing
        prop_series = {}
        for col in other_props:
            if col in embedding.columns:
                prop_series[col] = embedding[col]
        if len(prop_series) > 0:
            embedding = embedding.drop(other_props, axis=1)

        return embedding, prop_series

    def _random_sample_from_arrays(self, *input_array_list, n_samples=None, index=None):
        assert (n_samples is not None) != (index is not None)  # XOR -- must specify one or the other, but not both

        # Ensure array sizes are all the same
        sizes = []
        output_array_list = []
        for array in input_array_list:
            if hasattr(array, 'compute'):
                array = array.compute()
            sizes.append(array.shape[0])
            output_array_list.append(array)

        assert all([x == sizes[0] for x in sizes])
        size = sizes[0]

        if index is not None:
            assert (index.max() < size) & (len(index) <= size)
        else:
            # Sample from data / shuffle
            n_samples = min(size, n_samples)
            numpy.random.seed(self.seed)
            index = numpy.random.choice(numpy.arange(size), size=n_samples, replace=False)

        for pos, array in enumerate(output_array_list):
            if hasattr(array, 'values'):
                output_array_list[pos] = array.iloc[index]
            else:
                output_array_list[pos] = array[index]

        if len(output_array_list) == 1:
            output_array_list = output_array_list[0]

        return output_array_list, index

    def _compute_spearman_rho(self, fp_sample, Xt_sample, top_k=100):
        if hasattr(fp_sample, 'values'):
            fp_sample = fp_sample.values
        dist_array_tani = tanimoto_calculate(fp_sample, calc_distance=True)
        dist_array_eucl = pairwise_distances(Xt_sample)
        return cupy.nanmean(spearmanr(dist_array_tani, dist_array_eucl, top_k=top_k))

    def is_gpu_enabled(self):
        return True

    def cluster(self, embedding):
        """
        Runs clustering workflow on the data fetched from database/cache.
        """
        raise NotImplementedError

    def recluster(self,
                  filter_column=None,
                  filter_values=None,
                  n_clusters=None):
        """
        Runs reclustering on original dataframe or on the new dataframe passed.
        The new dataframe is usually a subset of the original dataframe.
        Caller may ask to include additional molecules.
        """
        raise NotImplementedError

    def add_molecules(self, chemblids: List):
        """
        ChembleId's accepted as argument to the existing database. Duplicates
        must be ignored.
        """
        raise NotImplementedError

    def compute_qa_matric(self):
        """
        Collects all quality matrix and log.
        """
        NotImplemented
