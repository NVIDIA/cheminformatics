import logging
import torch

import generativesampler_pb2
import generativesampler_pb2_grpc
from megamolbart.inference import MegaMolBART

logger = logging.getLogger(__name__)


class GenerativeSampler(generativesampler_pb2_grpc.GenerativeSampler):

    def __init__(self, *args, **kwargs):
        decoder_max_seq_len = kwargs['decoder_max_seq_len'] if 'decoder_max_seq_len' in kwargs else None
        self.megamolbart = MegaMolBART(decoder_max_seq_len=decoder_max_seq_len)

        try:
            iteration = int(self.megamolbart.iteration)
        except:
            iteration = 0
        self.iteration = iteration
        logger.info(f'Loaded iteration {self.iteration}')

    # TODO update to accept batched input if similes2embedding does
    # TODO how to handle length overrun for batch processing --> see also MegaMolBART.load_model in inference.py
    def SmilesToEmbedding(self, spec, context):

        smile_str = ''.join(spec.smiles)

        embedding, pad_mask = self.megamolbart.smiles2embedding(smile_str,
                                                                pad_length=spec.padding)
        dim = embedding.shape
        embedding = embedding.flatten().tolist()
        return generativesampler_pb2.EmbeddingList(embedding=embedding,
                                                   dim=dim,
                                                   pad_mask=pad_mask)

    def EmbeddingToSmiles(self, embedding_spec, context):
        '''
        Converts input embedding to SMILES.
        @param transform_spec: Input spec with embedding and mask.
        '''
        embedding = torch.FloatTensor(list(embedding_spec.embedding))
        pad_mask = torch.BoolTensor(list(embedding_spec.pad_mask))
        dim = tuple(embedding_spec.dim)

        embedding = torch.reshape(embedding, dim).cuda()
        pad_mask = torch.reshape(pad_mask, (dim[0], 1)).cuda()

        generated_mols = self.megamolbart.inverse_transform(embedding, pad_mask)
        return generativesampler_pb2.SmilesList(generatedSmiles=generated_mols)

    def FindSimilars(self, spec, context):

        smile_str = ''.join(spec.smiles)

        _, generated_smiles = \
            self.megamolbart.find_similars_smiles(
                smile_str,
                num_requested=spec.numRequested,
                scaled_radius=spec.radius)
        return generativesampler_pb2.SmilesList(generatedSmiles=generated_smiles)

    def Interpolate(self, spec, context):

        _, generated_smiles = self.megamolbart.interpolate_from_smiles(
            spec.smiles,
            num_points=spec.numRequested,
            scaled_radius=spec.radius)
        return generativesampler_pb2.SmilesList(generatedSmiles=generated_smiles)

    def GetIteration(self, spec, context):
        return generativesampler_pb2.IterationVal(iteration=self.iteration)
