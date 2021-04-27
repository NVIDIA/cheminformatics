import sys
import grpc
import logging

from concurrent import futures
from contextlib import contextmanager

sys.path.insert(0, "generated")

from nvidia.cheminformatics.grpc import SimilaritySampler
import similaritysampler_pb2_grpc
import similaritysampler_pb2

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


def test_dataframe():
    with similarity(similaritysampler_pb2_grpc.add_SimilaritySamplerServicer_to_server,
                    SimilaritySampler,
                    similaritysampler_pb2_grpc.SimilaritySamplerStub) as stub:
        spec = similaritysampler_pb2.SimilaritySpec(
            model=similaritysampler_pb2.SimilarityModel.CDDD,
            smiles='CC(=O)Nc1ccc(O)cc1',
            radius=0.0005,
            numRequested=10)

        result = stub.FindSimilars(spec)
        logger.info(result)
