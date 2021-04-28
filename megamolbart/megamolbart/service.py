import logging

import generativesampler_pb2
import generativesampler_pb2_grpc


logger = logging.getLogger(__name__)


class GenerativeSampler(generativesampler_pb2_grpc.GenerativeSampler):

    def __init__(self, *args, **kwargs):
        pass


    def FindSimilars(self, spec, context):
        # Code for using generative model
        pass
