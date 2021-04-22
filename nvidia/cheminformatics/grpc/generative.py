import logging

import similaritysampler_pb2
import similaritysampler_pb2_grpc

from nvidia.cheminformatics.wf.generative import MolBART, Cddd


logger = logging.getLogger(__name__)


class SimilaritySampler(similaritysampler_pb2_grpc.SimilaritySampler):

    def __init__(self, *args, **kwargs):
        self.cddd = Cddd()
        self.molbart = MolBART()

    def FindSimilars(self, similarity_spec, context):
        if similarity_spec.model == similaritysampler_pb2.SimilarityModel.CDDD:
            generated_smiles = self.cddd.find_similars_smiles_list(
                similarity_spec.smiles,
                num_requested=similarity_spec.numRequested,
                radius=similarity_spec.radius)
        else:
            generated_smiles, neighboring_embeddings, pad_mask = \
                self.molbart.find_similars_smiles_list(
                    similarity_spec.smiles,
                    num_requested=similarity_spec.numRequested,
                    radius=similarity_spec.radius)
        return similaritysampler_pb2.SmilesList(generatedSmiles=generated_smiles)
