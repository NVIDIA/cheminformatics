import sys
import grpc
import logging

from concurrent import futures
from contextlib import contextmanager

from megamolbart.service import GenerativeSampler
import generativesampler_pb2
import generativesampler_pb2_grpc
from util import (DEFAULT_NUM_LAYERS, DEFAULT_D_MODEL, DEFAULT_NUM_HEADS, CHECKPOINTS_DIR)

logger = logging.getLogger(__name__)


@contextmanager
def similarity(add_server_method, service_cls, stub_cls):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    add_server_method(service_cls(num_layers=DEFAULT_NUM_LAYERS,
                                  hidden_size=DEFAULT_D_MODEL,
                                  num_attention_heads=DEFAULT_NUM_HEADS,
                                  checkpoints_dir=CHECKPOINTS_DIR,
                                  vocab_path='/models/megamolbart/bart_vocab.txt',),
                      server)
    port = server.add_insecure_port('[::]:0')
    server.start()
    try:
        with grpc.insecure_channel('localhost:%d' % port) as channel:
            yield stub_cls(channel)
    finally:
        server.stop(None)

def test_fetch_iterations():
    sys.argv = [sys.argv[0]]
    with similarity(generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server,
                    GenerativeSampler,
                    generativesampler_pb2_grpc.GenerativeSamplerStub) as stub:

        result = stub.GetIteration(generativesampler_pb2.google_dot_protobuf_dot_empty__pb2.Empty())


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