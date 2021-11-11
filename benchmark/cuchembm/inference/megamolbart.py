import os
import pathlib
import logging
import torch

from typing import List

from generativesampler_pb2 import EmbeddingList, SmilesList

from cuchemcommon.utils.singleton import Singleton
from megamolbart.inference import MegaMolBART

logger = logging.getLogger(__name__)


class MegaMolBARTWrapper(metaclass=Singleton):

    def __init__(self) -> None:
        self.min_jitter_radius = 1

        # TODO: make this configurable/resuable accross all modules.
        checkpoint_file = 'megamolbart_checkpoint.nemo'
        files = sorted(pathlib.Path('/models').glob(f'**/{checkpoint_file}'))
        dir = files[-1].absolute().parent.as_posix()

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

        # embeddings = []

        # for _, row in generated_df.iterrows():
        #     embeddings.append(EmbeddingList(embedding=row.embeddings,
        #                                    dim=row.embeddings_dim))

        # return SmilesList(generatedSmiles=generated_df['SMILES'],
        #                   embeddings=embeddings)

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
