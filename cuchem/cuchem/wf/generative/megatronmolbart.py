import logging
from typing import List

import generativesampler_pb2
import generativesampler_pb2_grpc
import grpc
import pandas as pd
from cuchemcommon.data import GenerativeWfDao
from cuchemcommon.data.generative_wf import ChemblGenerativeWfDao
from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.workflow import BaseGenerativeWorkflow

logger = logging.getLogger(__name__)


class MegatronMolBART(BaseGenerativeWorkflow, metaclass=Singleton):

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao(None)) -> None:
        super().__init__(dao)

        self.min_jitter_radius = 1
        channel = grpc.insecure_channel('megamolbart:50051')
        self.stub = generativesampler_pb2_grpc.GenerativeSamplerStub(channel)

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             force_unique=False):
        spec = generativesampler_pb2.GenerativeSpec(
            model=generativesampler_pb2.GenerativeModel.MegaMolBART,
            smiles=smiles,
            radius=scaled_radius,
            numRequested=num_requested)

        result = self.stub.FindSimilars(spec)
        result = result.generatedSmiles

        generated_df = pd.DataFrame({'SMILES': result,
                                     'Generated': [True for i in range(len(result))]})
        generated_df['Generated'].iat[0] = False

        return generated_df

    def interpolate_from_smiles(self,
                                smiles: List,
                                num_points: int = 10,
                                scaled_radius=None,
                                force_unique=False):
        spec = generativesampler_pb2.GenerativeSpec(
            model=generativesampler_pb2.GenerativeModel.MegaMolBART,
            smiles=smiles,
            radius=scaled_radius,
            numRequested=num_points)

        result = self.stub.Interpolate(spec)
        result = result.generatedSmiles

        generated_df = pd.DataFrame({'SMILES': result,
                                     'Generated': [True for i in range(len(result))]})
        generated_df.iat[0, 1] = False
        generated_df.iat[0, 1] = False

        return generated_df
