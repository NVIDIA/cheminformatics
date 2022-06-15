import logging
import torch

from generativesampler_pb2 import EmbeddingList, SmilesList, Version
import generativesampler_pb2_grpc
from megamolbart.inference import MegaMolBART

from cuchemcommon.utils import Singleton

logger = logging.getLogger(__name__)


class GenerativeSampler(generativesampler_pb2_grpc.GenerativeSampler,
                        metaclass=Singleton):

    def __init__(self, cfg, *args, **kwargs):
        torch.set_grad_enabled(False)

        self.megamolbart = MegaMolBART(cfg)
        logger.info(f'Loaded Version {self.megamolbart.version}')

    # TODO: Add the ability to process multiple smiles
    def SmilesToEmbedding(self, spec, context):
        '''
        Converts input SMILES to embedding.
        @param spec: Transform spec with embedding.
        '''
        smile_str = ''.join(spec.smiles)
        tokens, hidden_states, pad_masks = self.megamolbart.smiles2embedding(smile_str)
        return EmbeddingList(embedding=hidden_states.flatten().tolist(),
                             dim=hidden_states.shape,
                             tokens=tokens.flatten().tolist(),
                             pad_mask=pad_masks.flatten().tolist())

    def EmbeddingToSmiles(self, embedding_spec, context):
        '''
        Converts input embedding to SMILES.
        @param transform_spec: Input spec with embedding and mask.
        '''
        hidden_states = torch.FloatTensor(list(embedding_spec.embedding))
        pad_masks = torch.BoolTensor(list(embedding_spec.pad_mask))
        tokens = torch.IntTensor(list(embedding_spec.tokens))
        dim = tuple(embedding_spec.dim)

        hidden_states = torch.reshape(hidden_states, dim).cuda()
        pad_masks = torch.reshape(pad_masks, (dim[0], dim[1])).cuda()
        tokens = torch.reshape(tokens, (dim[0], dim[1])).cuda()

        generated_mols = self.megamolbart.embedding2smiles(tokens, hidden_states, pad_masks)
        return SmilesList(generatedSmiles=generated_mols)

    def FindSimilars(self, spec, context):
        smile_str = ''.join(spec.smiles)
        generated_df = self.megamolbart.find_similars_smiles(smile_str,
                                                             num_requested=spec.numRequested,
                                                             scaled_radius=spec.radius,
                                                             sanitize=spec.sanitize,
                                                             pad_length=spec.padding,
                                                             force_unique=False)
        embeddings = []

        for _, row in generated_df.iterrows():
            embeddings.append(EmbeddingList(embedding=row.embeddings,
                                            dim=row.embeddings_dim))

        return SmilesList(generatedSmiles=generated_df['SMILES'],
                          embeddings=embeddings)

    def Interpolate(self, spec, context):

        _, generated_smiles = self.megamolbart.interpolate_smiles(
            spec.smiles,
            num_points=spec.numRequested,
            scaled_radius=spec.radius,
            sanitize=spec.sanitize,
            force_unique=False)
        return SmilesList(generatedSmiles=generated_smiles)

    def GetVersion(self, spec, context):
        return Version(version='0.1.0_' + self.megamolbart.version)
