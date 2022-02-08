import sys
import grpc
import logging
import pathlib

from concurrent import futures
from contextlib import contextmanager

from megamolbart.service import GenerativeSampler
import generativesampler_pb2
import generativesampler_pb2_grpc

logger = logging.getLogger(__name__)


@contextmanager
def similarity(add_server_method, service_cls, stub_cls):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    model_dir = sorted(pathlib.Path('/models/').glob('**/megamolbart_checkpoint.nemo'))[-1].absolute().parent.as_posix()
    add_server_method(service_cls(model_dir=model_dir),
                      server)
    port = server.add_insecure_port('[::]:0')
    server.start()
    try:
        with grpc.insecure_channel('localhost:%d' % port) as channel:
            yield stub_cls(channel)
    finally:
        server.stop(None)


def test_dataframe_similar():
    sys.argv = [sys.argv[0]]
    with similarity(generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server,
                    GenerativeSampler,
                    generativesampler_pb2_grpc.GenerativeSamplerStub) as stub:

        spec = generativesampler_pb2.GenerativeSpec(
            model=generativesampler_pb2.GenerativeModel.MegaMolBART,
            smiles=['CC(=O)Nc1ccc(O)cc1'],
            radius=2.0,
            numRequested=10)

        result = stub.FindSimilars(spec)


def test_dataframe_interpolate():
    sys.argv = [sys.argv[0]]
    with similarity(generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server,
                    GenerativeSampler,
                    generativesampler_pb2_grpc.GenerativeSamplerStub) as stub:

        spec = generativesampler_pb2.GenerativeSpec(
            model=generativesampler_pb2.GenerativeModel.MegaMolBART,
            smiles=['CC(=O)Nc1ccc(O)cc1', 'CC(=O)Nc1ccc(O)'],
            radius=0.0005,
            numRequested=10)

        result = stub.Interpolate(spec)


def test_transform():
    sys.argv = [sys.argv[0]]
    with similarity(generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server,
                    GenerativeSampler,
                    generativesampler_pb2_grpc.GenerativeSamplerStub) as stub:

        spec = generativesampler_pb2.GenerativeSpec(
            model=generativesampler_pb2.GenerativeModel.MegaMolBART,
            smiles=['CC(=O)Nc1ccc(O)'])

        result = stub.SmilesToEmbedding(spec)
        result = stub.EmbeddingToSmiles(result)
        logger.info(result)