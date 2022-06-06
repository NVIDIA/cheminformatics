import grpc
import logging
from typing import List

import pandas as pd
from generativesampler_pb2_grpc import GenerativeSamplerStub
from generativesampler_pb2 import GenerativeSpec, EmbeddingList, GenerativeModel

from cuchemcommon.data import GenerativeWfDao
from cuchemcommon.data.generative_wf import ChemblGenerativeWfDao
from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.workflow import BaseGenerativeWorkflow

logger = logging.getLogger(__name__)


class Cddd(BaseGenerativeWorkflow):
    __metaclass__ = Singleton

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao(None)) -> None:
        super().__init__(dao)

        self.min_jitter_radius = 1
        self.channel = grpc.insecure_channel('cddd:50051')
        self.stub = GenerativeSamplerStub(self.channel)

    def is_ready(self, timeout: int = 10) -> bool:
        try:
            grpc.channel_ready_future(self.channel).result(timeout=timeout)
            logger.info('CDDD is ready')
            return True
        except grpc.FutureTimeoutError:
            logger.warning('CDDD is not reachable.')
            return False

    def smiles_to_embedding(self,
                            smiles: str,
                            padding: int,
                            scaled_radius=None,
                            num_requested: int = 10,
                            sanitize=True):
        spec = GenerativeSpec(smiles=[smiles],
                              padding=padding,
                              radius=scaled_radius,
                              numRequested=num_requested,
                              sanitize=sanitize)

        result = self.stub.SmilesToEmbedding(spec)
        return result

    def embedding_to_smiles(self,
                            embedding,
                            dim: int,
                            pad_mask):
        spec = EmbeddingList(embedding=embedding,
                             dim=dim,
                             pad_mask=pad_mask)

        return self.stub.EmbeddingToSmiles(spec)

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             force_unique=False,
                             sanitize=True):
        spec = GenerativeSpec(model=GenerativeModel.MegaMolBART,
                              smiles=smiles,
                              radius=scaled_radius,
                              numRequested=num_requested,
                              forceUnique=force_unique,
                              sanitize=sanitize)
        result = self.stub.FindSimilars(spec)
        gsmis = list(result.generatedSmiles)
        embeddings = []
        dims = []
        smis = []
        for i in range(len(gsmis)):
            smi = gsmis[i]
            if smi in smis:
                continue

            smis.append(smi)
            embeddings.append(list(result.embeddings[i].embedding))
            dims.append(result.embeddings[i].dim)

        generated_df = pd.DataFrame({'SMILES': smis,
                                     'embeddings': embeddings,
                                     'embeddings_dim': dims,
                                     'Generated': [True for i in range(len(smis))]})
        generated_df['Generated'].iat[0] = False

        return generated_df

    def interpolate_smiles(self,
                           smiles: List,
                           num_points: int = 10,
                           scaled_radius=None,
                           force_unique=False,
                           sanitize=True):
        spec = GenerativeSpec(model=GenerativeModel.MegaMolBART,
                              smiles=smiles,
                              radius=scaled_radius,
                              numRequested=num_points,
                              forceUnique=force_unique,
                              sanitize=sanitize)

        result = self.stub.Interpolate(spec)
        result = list(set(result.generatedSmiles))

        generated_df = pd.DataFrame({'SMILES': result,
                                     'Generated': [True for i in range(len(result))]})
        generated_df.iat[0, 1] = False
        generated_df.iat[-1, 1] = False
        return generated_df
