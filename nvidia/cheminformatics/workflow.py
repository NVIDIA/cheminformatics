import logging

from datetime import datetime

import dask_cudf

from cuml.manifold import UMAP as cuUMAP
from cuml.dask.decomposition import PCA
from cuml.dask.cluster import KMeans as cuKMeans
from cuml.dask.manifold import UMAP as Dist_cuUMAP

import sklearn.cluster
import sklearn.decomposition
import umap


logger = logging.getLogger(__name__)


class CpuWorkflow:

    def __init__(self,
                 client,
                 pca_comps=64,
                 n_clusters=7,):
        self.client = client
        self.pca_comps = pca_comps
        self.n_clusters = n_clusters

    def execute(self, mol_df):
        logger.info("Executing CPU workflow...")

        mol_df = mol_df.persist()

        logger.info('PCA...')
        if self.pca_comps:
            task_start_time = datetime.now()
            pca = sklearn.decomposition.PCA(n_components=self.pca_comps)
            df_fingerprints = pca.fit_transform(mol_df)
            logger.info('### Runtime PCA time (hh:mm:ss.ms) {}'.format(
                datetime.now() - task_start_time))
        else:
            df_fingerprints = mol_df.copy()

        logger.info('KMeans...')
        task_start_time = datetime.now()
        kmeans_float = sklearn.cluster.KMeans(n_clusters=self.n_clusters)
        kmeans_float.fit(df_fingerprints)
        logger.info('### Runtime Kmeans time (hh:mm:ss.ms) {}'.format(
            datetime.now() - task_start_time))

        logger.info('UMAP...')
        task_start_time = datetime.now()
        umap_model = umap.UMAP()

        Xt = umap_model.fit_transform(df_fingerprints)
        # TODO: Use dask to distribute umap. https://github.com/dask/dask/issues/5229
        mol_df = mol_df.compute()
        mol_df['x'] = Xt[:, 0]
        mol_df['y'] = Xt[:, 1]
        mol_df['cluster'] = kmeans_float.labels_
        logger.info('### Runtime UMAP time (hh:mm:ss.ms) {}'.format(
            datetime.now() - task_start_time))

        return mol_df


class GpuWorkflow:

    def __init__(self,
                 client,
                 pca_comps=64,
                 n_clusters=7,):
        self.client = client
        self.pca_comps = pca_comps
        self.n_clusters = n_clusters

    def execute(self, mol_df):
        logger.info("Executing GPU workflow...")

        mol_df = dask_cudf.from_dask_dataframe(mol_df)
        mol_df = mol_df.persist()

        logger.info('PCA...')
        if self.pca_comps:
            pca = PCA(client=self.client, n_components=self.pca_comps)
            df_fingerprints = pca.fit_transform(mol_df)
        else:
            df_fingerprints = mol_df.copy()

        logger.info('KMeans...')
        kmeans_cuml = cuKMeans(client=self.client, n_clusters=self.n_clusters)
        kmeans_cuml.fit(df_fingerprints)
        kmeans_labels = kmeans_cuml.predict(df_fingerprints)

        logger.info('UMAP...')
        local_model = cuUMAP()
        X_train = df_fingerprints.compute()
        local_model.fit(X_train)

        umap_model = Dist_cuUMAP(local_model,
                                 n_neighbors=100,
                                 a=1.0,
                                 b=1.0,
                                 learning_rate=1.0,
                                 client=self.client)
        Xt = umap_model.transform(df_fingerprints)

        mol_df['x'] = Xt[0]
        mol_df['y'] = Xt[1]
        mol_df['cluster'] = kmeans_labels

        return mol_df
