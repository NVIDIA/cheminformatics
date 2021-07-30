import logging

import generativesampler_pb2
import generativesampler_pb2_grpc
from megamolbart.inference import MegaMolBART

logger = logging.getLogger(__name__)


class GenerativeSampler(generativesampler_pb2_grpc.GenerativeSampler):

    def __init__(self, *args, **kwargs):
        decoder_max_seq_len = kwargs['decoder_max_seq_len'] if 'decoder_max_seq_len' in kwargs else None
        vocab_path = kwargs['vocab_path'] if 'vocab_path' in kwargs else None
        checkpoints_dir = kwargs['checkpoints_dir'] if 'checkpoints_dir' in kwargs else None

        self.megamolbart = MegaMolBART(decoder_max_seq_len=decoder_max_seq_len,
                                       vocab_path=vocab_path,
                                       checkpoints_dir=checkpoints_dir)

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
        embedding = embedding.squeeze()
        shape = list(embedding.shape)
        assert len(shape) == 2

        embedding = shape + embedding.flatten().tolist()
        return generativesampler_pb2.EmbeddingList(embedding=embedding)

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
