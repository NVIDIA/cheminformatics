import logging

import similaritysampler_pb2
import similaritysampler_pb2_grpc

from nvidia.cheminformatics.wf.generative import MolBART, Cddd


logger = logging.getLogger(__name__)


class SimilaritySampler(similaritysampler_pb2_grpc.SimilaritySampler):

    def __init__(self, *args, **kwargs):
        self.molbart = MolBART()
        self.cddd = Cddd()

    def FindSimilars(self, similarity_spec, context):

        if similarity_spec.model == similaritysampler_pb2.SimilarityModel.CDDD:
            genreated_smiles = self.cddd.find_similars_smiles_list(
                similarity_spec.smiles,
                num_requested=similarity_spec.numRequested,
                radius=similarity_spec.radius)
        else:
            genreated_smiles = self.molbart.find_similars_smiles_list(
                similarity_spec.smiles,
                num_requested=similarity_spec.numRequested,
                radius=similarity_spec.radius)
        return similaritysampler_pb2.SmilesList(generatedSmiles=genreated_smiles)
