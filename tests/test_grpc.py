import sys
import grpc
import logging

sys.path.insert(0, "generated")

import interpolator_pb2_grpc
import interpolator_pb2

logger = logging.getLogger(__name__)


def test_dataframe():

    host = 'localhost'
    server_port = 50051

    channel = grpc.insecure_channel('{}:{}'.format(host, server_port))

    # bind the client and the server
    stub = interpolator_pb2_grpc.InterpolatorStub(channel)

    """
    Client function to call the rpc for GetServerResponse
    """
    spec = interpolator_pb2.InterpolationSpec(model='test',
                                              smiles=['CHEMBL25', 'CHEMBL521'])
    result = stub.Interpolate(spec)
    logger.info(result)
