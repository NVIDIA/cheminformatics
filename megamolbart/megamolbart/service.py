import logging

import generativesampler_pb2
import generativesampler_pb2_grpc

from megamolbart.inference import MegaMolBART


logger = logging.getLogger(__name__)


class GenerativeSampler(generativesampler_pb2_grpc.GenerativeSampler):

    def __init__(self, *args, **kwargs):
        self.megamolbart = MegaMolBART()


    # TODO update to accept batched input if similes2embedding does
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

        _, generated_smiles  = self.megamolbart.interpolate_from_smiles(
                    spec.smiles,
                    num_points=spec.numRequested,
                    scaled_radius=spec.radius)
        return generativesampler_pb2.SmilesList(generatedSmiles=generated_smiles)

