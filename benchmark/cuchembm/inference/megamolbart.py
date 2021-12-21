import os
import grpc
import pathlib
import logging
import pandas as pd

from typing import List

from generativesampler_pb2_grpc import GenerativeSamplerStub
from generativesampler_pb2 import (GenerativeSpec,
                                   EmbeddingList,
                                   SmilesList,
                                   GenerativeModel,
                                   google_dot_protobuf_dot_empty__pb2)

from cuchemcommon.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class MegaMolBARTWrapper(metaclass=Singleton):

    def __init__(self) -> None:
        self.min_jitter_radius = 1

        # TODO: make this configurable/resuable accross all modules.
        checkpoint_file = 'megamolbart_checkpoint.nemo'
        files = sorted(pathlib.Path('/models').glob(f'**/{checkpoint_file}'))
        dir = files[-1].absolute().parent.as_posix()

        from megamolbart.inference import MegaMolBART

        logger.info(f'Loading model from {dir}/{checkpoint_file}')
        self.megamolbart = MegaMolBART(model_dir=os.path.join(dir, checkpoint_file))
        logger.info(f'Loaded Version {self.megamolbart.version}')

    def is_ready(self, timeout: int = 10) -> bool:
        return True

    def smiles_to_embedding(self,
                            smiles: str,
                            pad_length: int):

        embedding, pad_mask = self.megamolbart.smiles2embedding(smiles,
                                                                pad_length=pad_length)
        dim = embedding.shape
        embedding = embedding.flatten().tolist()
        return EmbeddingList(embedding=embedding,
                             dim=dim,
                             pad_mask=pad_mask)

    def embedding_to_smiles(self,
                            embedding,
                            dim: int,
                            pad_mask):
        '''
        Converts input embedding to SMILES.
        @param transform_spec: Input spec with embedding and mask.
        '''
        import torch

        embedding = torch.FloatTensor(list(embedding))
        pad_mask = torch.BoolTensor(list(pad_mask))
        dim = tuple(dim)

        embedding = torch.reshape(embedding, dim).cuda()
        pad_mask = torch.reshape(pad_mask, (dim[0], 1)).cuda()

        generated_mols = self.megamolbart.inverse_transform(embedding, pad_mask)
        return SmilesList(generatedSmiles=generated_mols)

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             force_unique=False,
                             sanitize=True):

        generated_df = self.megamolbart.find_similars_smiles(
                smiles,
                num_requested=num_requested,
                scaled_radius=scaled_radius,
                sanitize=sanitize,
                force_unique=force_unique)
        return generated_df

    def interpolate_smiles(self,
                           smiles: List,
                           num_points: int = 10,
                           scaled_radius=None,
                           force_unique=False,
                           sanitize=True):

        _, generated_smiles = self.megamolbart.interpolate_smiles(
            smiles,
            num_points=num_points,
            scaled_radius=scaled_radius,
            sanitize=sanitize,
            force_unique=force_unique)
        return SmilesList(generatedSmiles=generated_smiles)


class GrpcMegaMolBARTWrapper(metaclass=Singleton):

    def __init__(self) -> None:

        self.channel = grpc.insecure_channel('nginx:50052')
        self.stub = GenerativeSamplerStub(self.channel)

    def is_ready(self, timeout: int = 10) -> bool:
        try:
            self.find_similars_smiles(smiles='CC')
            # grpc.channel_ready_future(self.channel).result(timeout=timeout)
            logger.info('Megatron MolBART is ready')
            return True
        except (grpc.RpcError):
            logger.warning('Megatron MolBART is not reachable.')
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
        generatedSmiles = result.generatedSmiles
        embeddings = []
        dims = []
        for embedding in result.embeddings:
            embeddings.append(list(embedding.embedding))
            dims.append(embedding.dim)

        generated_df = pd.DataFrame({'SMILES': generatedSmiles,
                                     'embeddings': embeddings,
                                     'embeddings_dim': dims,
                                     'Generated': [True for i in range(len(generatedSmiles))]})
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
        result = result.generatedSmiles

        generated_df = pd.DataFrame({'SMILES': result,
                                     'Generated': [True for i in range(len(result))]})
        generated_df.iat[0, 1] = False
        generated_df.iat[-1, 1] = False
        return generated_df