import logging
import numpy as np

from generativesampler_pb2 import EmbeddingList, SmilesList, Version
import generativesampler_pb2_grpc
from cdddinf import CdddInference

from cuchemcommon.utils import Singleton

logger = logging.getLogger(__name__)


class CdddService(generativesampler_pb2_grpc.GenerativeSampler, metaclass=Singleton):

    def __init__(self):
        self.cddd = CdddInference()

    def SmilesToEmbedding(self, spec, context):

        smile_str = ''.join(spec.smiles)

        embedding = self.cddd.smiles_to_embedding(
            smile_str,
            padding=spec.padding)

        embedding = embedding.flatten().tolist()
        return EmbeddingList(embedding=embedding,
                             dim=None,
                             pad_mask=None)

    def EmbeddingToSmiles(self, embedding_spec, context):
        '''
        Converts input embedding to SMILES.
        @param transform_spec: Input spec with embedding and mask.
        '''
        generated_mols = self.cddd.embedding_to_smiles(embedding_spec.embedding)
        return SmilesList(generatedSmiles=generated_mols)

    def FindSimilars(self, spec, context):

        smile_str = ''.join(spec.smiles)

        generated_df = self.cddd.find_similars_smiles(
                smile_str,
                num_requested=spec.numRequested,
                scaled_radius=spec.radius,
                sanitize=spec.sanitize,
                force_unique=False)

        embeddings = []
        for _, row in generated_df.iterrows():
            embeddings.append(EmbeddingList(embedding=row.embeddings,
                                           dim=row.embeddings_dim))

        return SmilesList(generatedSmiles=generated_df['SMILES'],
                          embeddings=embeddings)

    def Interpolate(self, spec, context):

        _, generated_smiles = self.cddd.interpolate_smiles(
            spec.smiles,
            num_points=spec.numRequested,
            scaled_radius=spec.radius,
            sanitize=spec.sanitize,
            force_unique=False)
        return SmilesList(generatedSmiles=generated_smiles)

    def GetVersion(self, spec, context):
        return Version(version='1.0.0')
