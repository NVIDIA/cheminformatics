import os
import pathlib
import logging

from typing import List

from generativesampler_pb2 import EmbeddingList, SmilesList

from cuchembench.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class CdddWrapper(metaclass=Singleton):

    def __init__(self) -> None:
        self.min_jitter_radius = 1

        from cuchem.wf.generative import Cddd
        self.inferrer = Cddd()

    def is_ready(self, timeout: int = 10) -> bool:
        return True

    def smiles_to_embedding(self,
                            smiles: str,
                            pad_length: int):

        embedding = self.inferrer.smiles_to_embedding(smiles,
                                                      padding=pad_length)
        dim = embedding.shape
        return EmbeddingList(embedding=embedding,
                             dim=dim)

    def embedding_to_smiles(self,
                            embedding,
                            dim: int,
                            pad_mask):
        '''
        Converts input embedding to SMILES.
        @param transform_spec: Input spec with embedding and mask.
        '''

        generated_mols = self.inferrer.embedding_to_smiles(embedding,
                                                           dim=dim,
                                                           pad_mask=pad_mask)
        return SmilesList(generatedSmiles=generated_mols)

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             force_unique=False,
                             sanitize=True):

        generated_df = self.inferrer.find_similars_smiles(smiles,
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

        _, generated_smiles = self.inferrer.interpolate_smiles(
            smiles,
            num_points=num_points,
            scaled_radius=scaled_radius,
            sanitize=sanitize,
            force_unique=force_unique)
        return SmilesList(generatedSmiles=generated_smiles)
