import logging
import numpy as np
import pandas as pd
from typing import List

from nvidia.cheminformatics.fingerprint import Embeddings
from nvidia.cheminformatics.data import GenerativeWfDao
from nvidia.cheminformatics.data.generative_wf import ChemblGenerativeWfDao
from nvidia.cheminformatics.utils.singleton import Singleton
from nvidia.cheminformatics.utils.data_peddler import download_cddd_models
from nvidia.cheminformatics.wf.generative import BaseGenerativeWorkflow


logger = logging.getLogger(__name__)


class Cddd(BaseGenerativeWorkflow, metaclass=Singleton):

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao()) -> None:
        super().__init__(dao)
        self.default_model_loc = download_cddd_models()
        self.dao = dao
        self.cddd_embeddings = Embeddings(model_dir=self.default_model_loc)

    def find_similars_smiles_list(self, smiles:str, num_requested:int=10, radius=0.75):
        embedding = self.cddd_embeddings.func.seq_to_emb(smiles).squeeze()
        neighboring_embeddings = self.addjitter(embedding, radius, cnt=num_requested)

        return self.cddd_embeddings.inverse_transform(neighboring_embeddings)

    def find_similars_smiles(self, smiles:str, num_requested:int=10, radius=0.75):
        generated_mols = self.find_similars_smiles_list(smiles, num_requested=num_requested, radius=radius)
        generated_df = pd.DataFrame({'SMILES': generated_mols,
                                     'Generated': [True for i in range(len(generated_mols))]},
                                   )
        generated_df.iat[ 0, 1] = False
        return generated_df


    def interpolate_from_smiles(self, smiles:List, num_points:int=10, add_jitter=False, radius=0.75):
        num_points = int(num_points) + 2
        if len(smiles) < 2:
            raise Exception('At-least two or more smiles are expected')

        def linear_interpolate_points(embedding, num_points):
            return np.linspace(embedding[0], embedding[1], num_points+2)[1:-1]

        result_df = []
        for idx in range(len(smiles) - 1):
            data = pd.DataFrame({'transformed_smiles': [smiles[idx], smiles[idx + 1]]})
            embeddings = np.asarray(self.cddd_embeddings.transform(data))

            interp_embeddings = np.apply_along_axis(linear_interpolate_points,
                                                    axis=0,
                                                    arr=embeddings,
                                                    num_points=num_points)

            interp_df = pd.DataFrame({'SMILES': self.cddd_embeddings.inverse_transform(interp_embeddings),
                                      'Generated': [True for i in range(num_points)]},
                                      )
            if add_jitter:
                interp_df = self._jitter(interp_df,
                                         interp_embeddings,
                                         self.cddd_embeddings.inverse_transform,
                                         radius=radius)

            # Mark the source and desinations as not generated
            interp_df.iat[ 0, 1] = False
            interp_df.iat[-1, 1] = False

            result_df.append(interp_df)

        return pd.concat(result_df)