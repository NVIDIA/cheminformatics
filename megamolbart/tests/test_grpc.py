import sys
import grpc
import logging

from concurrent import futures
from contextlib import contextmanager

sys.path.insert(0, "generated")

from megamolbart.service import GenerativeSampler
import generativesampler_pb2
import generativesampler_pb2_grpc

logger = logging.getLogger(__name__)


@contextmanager
def similarity(add_server_method, service_cls, stub_cls):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_server_method(service_cls(), server)
    port = server.add_insecure_port('[::]:0')
    server.start()

    try:
        with grpc.insecure_channel('localhost:%d' % port) as channel:
            yield stub_cls(channel)
    finally:
        server.stop(None)


# def test_dataframe_similar():
#     sys.argv = [sys.argv[0]]
#     with similarity(generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server,
#                     GenerativeSampler,
#                     generativesampler_pb2_grpc.GenerativeSamplerStub) as stub:

#         logger.info('dfds')
#         spec = generativesampler_pb2.GenerativeSpec(
#             model=generativesampler_pb2.GenerativeModel.MegaMolBART,
#             smiles=['CC(=O)Nc1ccc(O)cc1'],
#             radius=0.0005,
#             numRequested=10)

#         result = stub.FindSimilars(spec)
#         logger.info(result)

def test_dataframe_interpolate():
    sys.argv = [sys.argv[0]]
    with similarity(generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server,
                    GenerativeSampler,
                    generativesampler_pb2_grpc.GenerativeSamplerStub) as stub:

        logger.info('dfds')
        spec = generativesampler_pb2.GenerativeSpec(
            model=generativesampler_pb2.GenerativeModel.MegaMolBART,
            smiles=['CC(=O)Nc1ccc(O)cc1', 'CC(=O)Nc1ccc(O)'],
            radius=0.0005,
            numPoints=10)

        result = stub.Interpolate(spec)
        logger.info(result)

