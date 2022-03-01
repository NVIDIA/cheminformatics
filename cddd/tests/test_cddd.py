import grpc
import logging
from concurrent import futures
from contextlib import contextmanager

import generativesampler_pb2
import generativesampler_pb2_grpc

from cdddinf import CdddService


log = logging.getLogger(__name__)


@contextmanager
def similarity(add_server_method, service_cls, stub_cls):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    add_server_method(service_cls(), server)
    port = server.add_insecure_port('[::]:0')
    server.start()
    try:
        with grpc.insecure_channel(f'localhost:{port}') as channel:
            yield stub_cls(channel)
    finally:
        server.stop(None)


def test_cddd_embedding():
    with similarity(generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server,
                    CdddService,
                    generativesampler_pb2_grpc.GenerativeSamplerStub) as stub:
        spec = generativesampler_pb2.GenerativeSpec(
            smiles=['CN1C=NC2=C1C(=O)N(C(=O)N2C)C'])

        result = stub.SmilesToEmbedding(spec)
        log.debug(result)

        # Output includes the original smiles
        assert result.embedding is not None

        smiles = stub.EmbeddingToSmiles(result)
        log.debug(smiles)


def test_cddd_sample():
    num_points = 20
    with similarity(generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server,
                    CdddService,
                    generativesampler_pb2_grpc.GenerativeSamplerStub) as stub:
        spec = generativesampler_pb2.GenerativeSpec(
            smiles=['CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
            radius=1.0,
            numRequested=num_points)

        result = stub.FindSimilars(spec)
        log.debug(result)

        # Output includes the original smiles
        assert len(result.generatedSmiles) == num_points + 1


def test_cddd_interpolation():
    with similarity(generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server,
                    CdddService,
                    generativesampler_pb2_grpc.GenerativeSamplerStub) as stub:
        num_points = 10
        spec = generativesampler_pb2.GenerativeSpec(
            smiles=['CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CCCCN1CCOC[C@@H]1C(=O)O[C@H](C)[C@H]1CCOC1'],
            radius=1.0,
            numRequested=num_points)

        result = stub.Interpolate(spec)
        log.debug(result)

        # Output includes the original smiles
        assert len(result.generatedSmiles) == num_points + 2
