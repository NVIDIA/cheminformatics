from typing import List
import numpy as np

from nvidia.cheminformatics.data.generative_wf import ChemblGenerativeWfDao
from nvidia.cheminformatics.data import GenerativeWfDao


class BaseGenerativeWorkflow:

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao()) -> None:
        self.dao = dao

    def _jitter(self, interp_df, embeddings, embedding_funct):
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
            interp_df['SMILES'] = embedding_funct(embeddings)
        return interp_df

    def interpolate_from_id(self, ids:List, id_type:str='chembleid', num_points=10, add_jitter=False):
        smiles = None
        if id_type.lower() == 'chembleid':
            smiles = [row[2] for row in self.dao.fetch_id_from_chembl(ids)]
            if len(smiles) != len(ids):
                raise Exception('One of the ids is invalid %s', ids)
        else:
            raise Exception('id type %s not supported' % id_type)

        return self.interpolate_from_smiles(smiles, num_points=num_points, add_jitter=add_jitter)

    def interpolate_from_smiles(self, smiles:List, num_points:int=10, add_jitter=False):
        NotImplemented


from nvidia.cheminformatics.wf.interpolation.molbart import MolBARTInterpolation
from nvidia.cheminformatics.wf.interpolation.latentspaceinterpolation import LatentSpaceInterpolation
