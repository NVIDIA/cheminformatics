import logging
from typing import List
import numpy as np
import torch
from functools import singledispatch

from nvidia.cheminformatics.data.generative_wf import ChemblGenerativeWfDao
from nvidia.cheminformatics.data import GenerativeWfDao


logger = logging.getLogger(__name__)


@singledispatch
def add_jitter(embedding, radius, cnt):
    return NotImplemented


@add_jitter.register(np.ndarray)
def _(embedding, radius, cnt):
    noise = np.random.normal(0, radius, (cnt,) + embedding.shape)

    return noise + embedding


@add_jitter.register(torch.Tensor)
def _(embedding, radius, cnt):
    permuted_emb = embedding.permute(1, 0, 2)
    noise = torch.normal(0,  radius, (cnt,) + permuted_emb.shape[1:]).to(embedding.device)

    return noise + permuted_emb


class BaseGenerativeWorkflow:

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao()) -> None:
        self.dao = dao

    def _jitter(self, interp_df, embeddings, embedding_funct, radius=0.5):
        """
        Add jitter to embedding if the generated SMILES are same.
        """
        for idx in range(1, interp_df.shape[0] - 1):
            if interp_df.iat[idx, 0] == interp_df.iat[idx + 1, 0]:
                regen = True
                add_jitter(embeddings[idx], radius, 1)

        regen = False
        if interp_df.shape[0] > 3:
            # If first three molecules are same, previous loop changes the sec
            # molecule. This block will fix the third one.
            if interp_df.iat[0, 0] == interp_df.iat[2, 0]:
                regen = True
                add_jitter(embeddings[2], radius, 1)

        if regen:
            interp_df['SMILES'] = embedding_funct(embeddings)
        return interp_df

    def addjitter(self, embedding, radius, cnt=1):
        return add_jitter(embedding, radius, cnt=cnt)

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

    def find_similars_smiles_from_id(self, chemble_id:str, id_type:str='chembleid', num_requested=10):
        smiles = None
        if id_type.lower() == 'chembleid':
            smiles = [row[2] for row in self.dao.fetch_id_from_chembl(chemble_id)]
            if len(smiles) != len(chemble_id):
                raise Exception('One of the ids is invalid %s', chemble_id)
        else:
            raise Exception('id type %s not supported' % id_type)

        return self.find_similars_smiles(smiles[0], num_requested=num_requested)

    def find_similars_smiles(self, smiles:str, num_requested:int=10, radius=0.5):
        NotImplemented


from nvidia.cheminformatics.wf.generative.molbart import MolBART as MolBART
from nvidia.cheminformatics.wf.generative.cddd import Cddd as Cddd
