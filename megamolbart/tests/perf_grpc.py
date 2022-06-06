# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import sys

import grpc
from locust import task, User, constant

sys.path.insert(0, "generated")
import generativesampler_pb2
import generativesampler_pb2_grpc

from tests.utils import stopwatch


class GRPCLocust(User):
    host = 'http://192.167.100.2'
    wait_time = constant(.1)

    @task
    @stopwatch('GRPC_Sample')
    def client_task(self):
        with grpc.insecure_channel('127.0.0.1:50051') as channel:
            stub = generativesampler_pb2_grpc.GenerativeSamplerStub(channel)

            spec = generativesampler_pb2.GenerativeSpec(
                model=generativesampler_pb2.GenerativeModel.MegaMolBART,
                smiles='CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
                radius=0.0001,
                numRequested=10)

            response = stub.SmilesToEmbedding(spec)
