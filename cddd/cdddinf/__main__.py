#!/usr/bin/env python3

import logging
import sys
import argparse

from concurrent import futures

import grpc
from cdddinf.service import CdddService
import generativesampler_pb2_grpc


logging.basicConfig(level=logging.INFO)
log = logging.getLogger('cddd')
formatter = logging.Formatter('%(asctime)s %(name)s [%(levelname)s]: %(message)s')


class Launcher(object):
    """
    Application launcher. This class can execute the workflows in headless (for
    benchmarking and testing) and with UI.
    """

    def __init__(self):
        parser = argparse.ArgumentParser(description='CDDD gRPC Service')
        parser.add_argument('-p', '--port',
                            dest='port',
                            type=int,
                            default=50051,
                            help='GRPC server Port')
        parser.add_argument('-d', '--debug',
                            dest='debug',
                            action='store_true',
                            default=False,
                            help='Show debug messages')

        args = parser.parse_args(sys.argv[1:])

        if args.debug:
            log.setLevel(logging.DEBUG)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        generativesampler_pb2_grpc.add_GenerativeSamplerServicer_to_server(
            CdddService(),
            server)
        server.add_insecure_port(f'[::]:{args.port}')
        log.info(f'Server running on port {args.port}')
        server.start()
        server.wait_for_termination()


def main():
    Launcher()


if __name__ == '__main__':
    main()
