import os
import logging
import numpy as np
import pandas as pd
from typing import List
from cddd.inference import InferenceModel

from nvidia.cheminformatics.fingerprint import Embeddings
from nvidia.cheminformatics.data import GenerativeWfDao
from nvidia.cheminformatics.data.generative_wf import ChemblGenerativeWfDao
from nvidia.cheminformatics.utils.singleton import Singleton
from nvidia.cheminformatics.utils.data_peddler import download_cddd_models


logger = logging.getLogger(__name__)


class LatentSpaceInterpolation(metaclass=Singleton):

    def __init__(self,
                 dao: GenerativeWfDao = ChemblGenerativeWfDao()) -> None:
        self.default_model_loc = download_cddd_models()
        self.dao = dao

    def interpolate_from_id(self, ids:List, id_type:str='chembleid', num_points=10):
        smiles = None
        if id_type.lower() == 'chembleid':
            smiles = [row[2] for row in self.dao.fetch_id_from_chembl(ids)]
            if len(smiles) != len(ids):
                raise Exception('One of the ids is invalid %s', ids)
        else:
            raise Exception('id type %s not supported' % id_type)

        return self.interpolate_from_smiles(smiles, num_points=num_points)

    def _jitter(self, interp_df, embeddings, cddd_embeddings):
        """
        Add jitter to embedding if the generated SMILES are same.
        """

        def _addjitter(embeddings, idx):
            noise = np.random.normal(0, 0.5, embeddings[idx].shape)
            embeddings[idx] += noise

        for idx in range(1, interp_df.shape[0] - 1):
            if interp_df.iat[idx, 0] == interp_df.iat[idx + 1, 0]:
                regen = True
                _addjitter(embeddings, idx)

        regen = False
        if interp_df.shape[0] > 3:
            # If first three molecules are same, previous loop changes the sec
            # molecule. This block will fix the third one.
            if interp_df.iat[0, 0] == interp_df.iat[2, 0]:
                regen = True
                _addjitter(embeddings, 2)

        if regen:
            interp_df['SMILES'] = cddd_embeddings.inverse_transform(embeddings)
        return interp_df

    def interpolate_from_smiles(self, smiles:List, num_points:int=10):
        num_points = int(num_points) + 2
        if len(smiles) < 2:
            raise Exception('At-least two or more smiles are expected')

        cddd_embeddings = Embeddings(model_dir=self.default_model_loc)
        def linear_interpolate_points(embedding, num_points):
            return np.linspace(embedding[0], embedding[1], num_points+2)[1:-1]

        result_df = []
        for idx in range(len(smiles) - 1):
            data = pd.DataFrame({'transformed_smiles': [smiles[idx], smiles[idx + 1]]})
            embeddings = np.asarray(cddd_embeddings.transform(data))

            interp_embeddings = np.apply_along_axis(linear_interpolate_points,
                                                    axis=0,
                                                    arr=embeddings,
                                                    num_points=num_points)

            interp_df = pd.DataFrame({'SMILES': cddd_embeddings.inverse_transform(interp_embeddings),
                                      'Generated': [True for i in range(num_points)]},
                                      )
            interp_df = self._jitter(interp_df, interp_embeddings, cddd_embeddings)

            # Mark the source and desinations as not generated
            interp_df.iat[ 0, 1] = False
            interp_df.iat[-1, 1] = False

            result_df.append(interp_df)

        return pd.concat(result_df)
