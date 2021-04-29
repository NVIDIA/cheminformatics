import logging
import numpy as np
import pandas as pd
from typing import List

import torch
import torch.nn
import pickle
from pathlib import Path
import numpy as np
from functools import partial

from rdkit import Chem

from cuchemcommon.data import GenerativeWfDao
from cuchemcommon.data.generative_wf import ChemblGenerativeWfDao
from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.workflow import BaseGenerativeWorkflow
from nvidia.cheminformatics.fingerprint import MorganFingerprint


logger = logging.getLogger(__name__)


class MegatronMolBART(BaseGenerativeWorkflow, metaclass=Singleton):

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao(MorganFingerprint)) -> None:
        super().__init__(dao)


    def find_similars_smiles(self,
                             smiles:str,
                             num_requested:int=10,
                             radius=None,
                             force_unique=False):
        # Call gRPC implemetation of Megron molbart
        # Create the dataframe required by Caller
        return generated_df

    def interpolate_from_smiles(self,
                                smiles:List,
                                num_points:int=10,
                                radius=None,
                                force_unique=False):
        # Call gRPC implemetation of Megron molbart
        # Create the dataframe required by Caller

