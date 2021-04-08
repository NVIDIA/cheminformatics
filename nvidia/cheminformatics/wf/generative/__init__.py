import logging
from typing import List

from rdkit.Chem import Draw, PandasTools
import numpy as np
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


# @add_jitter.register(torch.Tensor)
# def _(embedding, radius, cnt):
#     permuted_emb = embedding.permute(1, 0, 2)
#     noise = torch.normal(0,  radius, (cnt,) + permuted_emb.shape[1:]).to(embedding.device)

#     return noise + permuted_emb


class BaseGenerativeWorkflow:

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao()) -> None:
        self.dao = dao
        self.radius_scale = None

    def interpolate_from_smiles(self,
                                smiles:List,
                                num_points:int=10,
                                radius=None,
                                force_unique=False):
        NotImplemented

    def find_similars_smiles_list(self,
                                  smiles:str,
                                  num_requested:int=10,
                                  radius=None,
                                  force_unique=False):
        NotImplemented

    def find_similars_smiles(self,
                             smiles:str,
                             num_requested:int=10,
                             radius=None,
                             force_unique=False):
        NotImplemented

    def addjitter(self,
                  embedding,
                  radius=None,
                  cnt=1):
        radius = radius if radius else self.radius_scale
        return add_jitter(embedding, radius, cnt)

    def compute_unique_smiles(self,
                              interp_df,
                              embeddings,
                              embedding_funct,
                              radius=None):
        """
        Identify duplicate SMILES and distorts the embedding. The input df
        must have columns 'SMILES' and 'Generated' at 0th and 1st position.
        'Generated' colunm must contain boolean to classify SMILES into input
        SMILES(False) and generated SMILES(True).

        This function does not make any assumptions about order of embeddings.
        Instead it simply orders the df by SMILES to identify the duplicates.
        """

        radius = radius if radius else self.radius_scale

        for i in range(5):
            smiles = interp_df['SMILES'].sort_values()
            duplicates = set()
            for idx in range(0, smiles.shape[0] - 1):
                if smiles.iat[idx] == smiles.iat[idx + 1]:
                    duplicates.add(smiles.index[idx])
                    duplicates.add(smiles.index[idx + 1])

            if len(duplicates) > 0:
                for dup_idx in duplicates:
                    if interp_df.iat[dup_idx, 1]:
                        # add jitter to generated molecules only
                        embeddings[dup_idx] = self.addjitter(
                            embeddings[dup_idx], radius, 1)
                interp_df['SMILES'] = embedding_funct(embeddings)
            else:
                break

        # Ensure all generated molecules are valid.
        for i in range(5):
            PandasTools.AddMoleculeColumnToFrame(interp_df,'SMILES')
            invalid_mol_df = interp_df[interp_df['ROMol'].isnull()]

            if not invalid_mol_df.empty:
                invalid_index = invalid_mol_df.index.to_list()
                for idx in invalid_index:
                    embeddings[idx] = self.addjitter(embeddings[idx],
                                                        radius,
                                                        cnt=1)
                interp_df['SMILES'] = embedding_funct(embeddings)
            else:
                break

        # Cleanup
        if 'ROMol' in interp_df.columns:
            interp_df = interp_df.drop('ROMol', axis=1)

        return interp_df

    def interpolate_from_id(self,
                            ids:List,
                            id_type:str='chembleid',
                            num_points=10,
                            force_unique=False,
                            scaled_radius:int=1):
        smiles = None

        if not self.radius_scale:
            raise Exception('Property `radius_scale` must be defined in model class.')
        else:
            radius = float(scaled_radius * self.radius_scale)

        if id_type.lower() == 'chembleid':
            smiles = [row[2] for row in self.dao.fetch_id_from_chembl(ids)]
            if len(smiles) != len(ids):
                raise Exception('One of the ids is invalid %s', ids)
        else:
            raise Exception('id type %s not supported' % id_type)

        return self.interpolate_from_smiles(smiles,
                                            num_points=num_points,
                                            radius=radius,
                                            force_unique=force_unique)

    def find_similars_smiles_from_id(self,
                                     chemble_id:str,
                                     id_type:str='chembleid',
                                     num_requested=10,
                                     force_unique=False,
                                     scaled_radius:int=1):
        smiles = None

        if not self.radius_scale:
            raise Exception('Property `radius_scale` must be defined in model class.')
        else:
            radius = float(scaled_radius * self.radius_scale)

        if id_type.lower() == 'chembleid':
            smiles = [row[2] for row in self.dao.fetch_id_from_chembl(chemble_id)]
            if len(smiles) != len(chemble_id):
                raise Exception('One of the ids is invalid %s' + chemble_id)
        else:
            raise Exception('id type %s not supported' % id_type)

        return self.find_similars_smiles(smiles[0],
                                         num_requested=num_requested,
                                         radius=radius,
                                         force_unique=force_unique)


from nvidia.cheminformatics.wf.generative.molbart import MolBART as MolBART
from nvidia.cheminformatics.wf.generative.cddd import Cddd as Cddd