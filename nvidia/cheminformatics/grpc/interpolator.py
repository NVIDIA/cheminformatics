import sys
import grpc
import logging

import interpolator_pb2_grpc
import interpolator_pb2

from nvidia.cheminformatics.wf.generate.latentspaceinterpolation import LatentSpaceInterpolation


logger = logging.getLogger(__name__)


class InterpolatorService(interpolator_pb2_grpc.InterpolatorServicer):

    def __init__(self, *args, **kwargs):
        pass

    def Interpolate(self, interpolation_spec, context):

        # get the string from the incoming request
        smiles = interpolation_spec.smiles

        genreated_df = LatentSpaceInterpolation().interpolate_from_smiles(interpolation_spec.smiles)

        generated_smiles = []
        for idx in range(genreated_df.shape[0]):
            generated_smiles.append(genreated_df.iat[idx, 0])


        return interpolator_pb2.SmilesList(generatedSmiles=generated_smiles)
