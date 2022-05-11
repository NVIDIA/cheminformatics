import logging
import sqlite3
from contextlib import closing
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem

from cuchemcommon.data import GenerativeWfDao
from cuchemcommon.data.generative_wf import ChemblGenerativeWfDao
from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.workflow import BaseGenerativeWorkflow
from cuchem.utils.data_peddler import download_cddd_models

logger = logging.getLogger(__name__)
from functools import singledispatch
@singledispatch
def add_jitter(embedding, radius, cnt, shape):
    return NotImplemented
@add_jitter.register(np.ndarray)
def _(embedding, radius, cnt, shape):
    distorteds = []
    for i in range(cnt):
        noise = np.random.normal(0, radius, embedding.shape)
        distorted = noise + embedding
        distorteds.append(distorted)
    return distorteds

class Cddd(BaseGenerativeWorkflow):
    __metaclass__ = Singleton

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao(None)) -> None:
        super().__init__(dao)
        self.default_model_loc = download_cddd_models()
        self.dao = dao
        self.min_jitter_radius = 0.5

        from cddd.inference import InferenceModel
        self.model = InferenceModel(self.default_model_loc,
                                   use_gpu=True,
                                   cpu_threads=5)

    def __len__(self):
        return self.model.hparams.emb_size

    def _compute_radius(self, scaled_radius):
        if scaled_radius:
            return float(scaled_radius * self.min_jitter_radius)
        else:
            return self.min_jitter_radius

    def addjitter(self,
                  embedding,
                  radius=None,
                  cnt=1,
                  shape=None):
        radius = radius if radius else self.min_jitter_radius
        return add_jitter(embedding, radius, cnt, shape)

    def transform(self, data):
        data = data['transformed_smiles']
        return self.model.seq_to_emb(data).squeeze()

    def is_known_smiles(self, smiles):
        with closing(sqlite3.connect(self.dao.chembl_db, uri=True)) as con:
            return self.dao.is_valid_chemble_smiles(smiles, con)

    def inverse_transform(self, embeddings, sanitize):
        embeddings = np.asarray(embeddings)
        mol_strs = self.model.emb_to_seq(embeddings)

        smiles_interp_list = []
        for smiles in mol_strs:
            if sanitize:
                mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
                if mol:
                    sanitized_smiles = Chem.MolToSmiles(mol)
                    smiles_interp_list.append(sanitized_smiles)
                else:
                    smiles_interp_list.append(smiles)
            else:
                smiles_interp_list.append(smiles)
        return smiles_interp_list

    def smiles_to_embedding(self, smiles: str, padding: int):
        embedding = self.model.seq_to_emb(smiles).squeeze()
        return embedding

    def embedding_to_smiles(self,
                            embedding,
                            dim: int,
                            pad_mask):
        return self.inverse_transform(embedding)

    def find_similars_smiles_list(self,
                                  smiles: str,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  sanitize=True,
                                  force_unique=False):

        radius = self._compute_radius(scaled_radius)
        embedding = self.model.seq_to_emb(smiles).squeeze()
        embeddings = self.addjitter(embedding, radius, cnt=num_requested)

        neighboring_embeddings = np.concatenate([embedding.reshape(1, embedding.shape[0]),
                                                 embeddings])
        embeddings = [embedding] + embeddings
        generated_mols = self.inverse_transform(neighboring_embeddings, sanitize)

        return generated_mols, embeddings

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             sanitize=True,
                             force_unique=False):
        generated_mols, neighboring_embeddings = self.find_similars_smiles_list(smiles,
                                                                                num_requested=num_requested,
                                                                                scaled_radius=scaled_radius,
                                                                                force_unique=force_unique,
                                                                                sanitize=sanitize)
        dims = []
        for neighboring_embedding in neighboring_embeddings:
            dims.append(neighboring_embedding.shape)

        generated_df = pd.DataFrame({'SMILES': generated_mols,
                                     'embeddings': neighboring_embeddings,
                                     'embeddings_dim': dims,
                                     'Generated': [True for i in range(len(generated_mols))]})
        generated_df.iat[0, 3] = False

        if force_unique:
            generated_df = self.compute_unique_smiles(generated_df,
                                                      self.inverse_transform,
                                                      scaled_radius=scaled_radius)
        return generated_df

    def interpolate_smiles(self,
                           smiles: List,
                           num_points: int = 10,
                           scaled_radius=None,
                           force_unique=False,
                           sanitize=True):

        num_points = int(num_points) + 2
        if len(smiles) < 2:
            raise Exception('At-least two or more smiles are expected')

        def linear_interpolate_points(embedding, num_points):
            return np.linspace(embedding[0], embedding[1], num_points)

        result_df = []
        for idx in range(len(smiles) - 1):
            data = pd.DataFrame({'transformed_smiles': [smiles[idx], smiles[idx + 1]]})
            input_embeddings = np.asarray(self.transform(data))

            interp_embeddings = np.apply_along_axis(linear_interpolate_points,
                                                    axis=0,
                                                    arr=input_embeddings,
                                                    num_points=num_points)
            generated_mols = self.inverse_transform(interp_embeddings, sanitize)
            interp_embeddings = interp_embeddings.tolist()

            dims = []
            embeddings = []
            for interp_embedding in interp_embeddings:
                dims.append(input_embeddings.shape)
                interp_embedding = np.asarray(interp_embedding)
                embeddings.append(interp_embedding)

            interp_df = pd.DataFrame({'SMILES': generated_mols,
                                      'embeddings': embeddings,
                                      'embeddings_dim': dims,
                                      'Generated': [True for i in range(num_points)]})

            # Mark the source and desinations as not generated
            interp_df.iat[0, 3] = False
            interp_df.iat[-1, 3] = False

            if force_unique:
                interp_df = self.compute_unique_smiles(interp_df,
                                                       self.inverse_transform,
                                                       scaled_radius=scaled_radius)

            result_df.append(interp_df)

        return pd.concat(result_df)

# import grpc
# import logging
# from typing import List

# import pandas as pd
# from generativesampler_pb2_grpc import GenerativeSamplerStub
# from generativesampler_pb2 import GenerativeSpec, EmbeddingList, GenerativeModel

# from cuchemcommon.data import GenerativeWfDao
# from cuchemcommon.data.generative_wf import ChemblGenerativeWfDao
# from cuchemcommon.utils.singleton import Singleton
# from cuchemcommon.workflow import BaseGenerativeWorkflow

# logger = logging.getLogger(__name__)


# class Cddd(BaseGenerativeWorkflow):
#     __metaclass__ = Singleton

#     def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao(None)) -> None:
#         super().__init__(dao)

#         self.min_jitter_radius = 1
#         self.channel = grpc.insecure_channel('cddd:50051')
#         self.stub = GenerativeSamplerStub(self.channel)

#     def is_ready(self, timeout: int = 10) -> bool:
#         try:
#             grpc.channel_ready_future(self.channel).result(timeout=timeout)
#             logger.info('CDDD is ready')
#             return True
#         except grpc.FutureTimeoutError:
#             logger.warning('CDDD is not reachable.')
#             return False

#     def smiles_to_embedding(self,
#                             smiles: str,
#                             padding: int,
#                             scaled_radius=None,
#                             num_requested: int = 10,
#                             sanitize=True):
#         spec = GenerativeSpec(smiles=[smiles],
#                               padding=padding,
#                               radius=scaled_radius,
#                               numRequested=num_requested,
#                               sanitize=sanitize)

#         result = self.stub.SmilesToEmbedding(spec)
#         return result

#     def embedding_to_smiles(self,
#                             embedding,
#                             dim: int,
#                             pad_mask):
#         spec = EmbeddingList(embedding=embedding,
#                              dim=dim,
#                              pad_mask=pad_mask)

#         return self.stub.EmbeddingToSmiles(spec)

#     def find_similars_smiles(self,
#                              smiles: str,
#                              num_requested: int = 10,
#                              scaled_radius=None,
#                              force_unique=False,
#                              sanitize=True):
#         spec = GenerativeSpec(model=GenerativeModel.MegaMolBART,
#                               smiles=smiles,
#                               radius=scaled_radius,
#                               numRequested=num_requested,
#                               forceUnique=force_unique,
#                               sanitize=sanitize)
#         result = self.stub.FindSimilars(spec)
#         generatedSmiles = list(result.generatedSmiles)
#         embeddings = []
#         dims = []
#         for embedding in result.embeddings:
#             embeddings.append(list(embedding.embedding))
#             dims.append(embedding.dim)

#         generated_df = pd.DataFrame({'SMILES': generatedSmiles,
#                                      'embeddings': embeddings,
#                                      'embeddings_dim': dims,
#                                      'Generated': [True for i in range(len(generatedSmiles))]})
#         generated_df['Generated'].iat[0] = False

#         return generated_df

#     def interpolate_smiles(self,
#                            smiles: List,
#                            num_points: int = 10,
#                            scaled_radius=None,
#                            force_unique=False,
#                            sanitize=True):
#         spec = GenerativeSpec(model=GenerativeModel.MegaMolBART,
#                               smiles=smiles,
#                               radius=scaled_radius,
#                               numRequested=num_points,
#                               forceUnique=force_unique,
#                               sanitize=sanitize)

#         result = self.stub.Interpolate(spec)
#         result = list(result.generatedSmiles)

#         generated_df = pd.DataFrame({'SMILES': result,
#                                      'Generated': [True for i in range(len(result))]})
#         generated_df.iat[0, 1] = False
#         generated_df.iat[-1, 1] = False
#         return generated_df
