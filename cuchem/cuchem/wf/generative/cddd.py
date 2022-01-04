import logging
import sqlite3
from contextlib import closing
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem

from cuchemcommon.data import GenerativeWfDao
from cuchemcommon.data.generative_wf import ChemblGenerativeWfDao
from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.workflow import BaseGenerativeWorkflow
from cuchem.utils.data_peddler import download_cddd_models

logger = logging.getLogger(__name__)


class Cddd(BaseGenerativeWorkflow):
    __metaclass__ = Singleton

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao(None)) -> None:
        super().__init__(dao)
        self.default_model_loc = download_cddd_models()
        self.dao = dao
        self.min_jitter_radius = 0.5

        from cddd.inference import InferenceModel
        self.model = InferenceModel(self.default_model_loc,
                                   use_gpu=True,
                                   cpu_threads=5)

    def __len__(self):
        return self.model.hparams.emb_size

    def transform(self, data):
        data = data['transformed_smiles']
        return self.model.seq_to_emb(data).squeeze()

    def is_known_smiles(self, smiles):
        with closing(sqlite3.connect(self.dao.chembl_db, uri=True)) as con:
            return self.dao.is_valid_chemble_smiles(smiles, con)

    def inverse_transform(self, embeddings, sanitize):
        embeddings = np.asarray(embeddings)
        mol_strs = self.model.emb_to_seq(embeddings)

        smiles_interp_list = []
        for smiles in mol_strs:
            if sanitize:
                mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
                if mol:
                    sanitized_smiles = Chem.MolToSmiles(mol)
                    smiles_interp_list.append(sanitized_smiles)
                else:
                    smiles_interp_list.append(smiles)
            else:
                smiles_interp_list.append(smiles)
        return smiles_interp_list

    def smiles_to_embedding(self, smiles: str, padding: int):
        embedding = self.model.seq_to_emb(smiles).squeeze()
        return embedding

    def embedding_to_smiles(self,
                            embedding,
                            dim: int,
                            pad_mask):
        return self.inverse_transform(embedding)

    def find_similars_smiles_list(self,
                                  smiles: str,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  sanitize=True,
                                  force_unique=False):

        radius = self._compute_radius(scaled_radius)
        embedding = self.model.seq_to_emb(smiles).squeeze()
        embeddings = self.addjitter(embedding, radius, cnt=num_requested)

        neighboring_embeddings = np.concatenate([embedding.reshape(1, embedding.shape[0]),
                                                 embeddings])
        embeddings = [embedding] + embeddings
        generated_mols = self.inverse_transform(neighboring_embeddings, sanitize)

        return generated_mols, embeddings

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             sanitize=True,
                             force_unique=False):
        generated_mols, neighboring_embeddings = self.find_similars_smiles_list(smiles,
                                                                                num_requested=num_requested,
                                                                                scaled_radius=scaled_radius,
                                                                                force_unique=force_unique,
                                                                                sanitize=sanitize)
        dims = []
        for neighboring_embedding in neighboring_embeddings:
            dims.append(neighboring_embedding.shape)

        generated_df = pd.DataFrame({'SMILES': generated_mols,
                                     'embeddings': neighboring_embeddings,
                                     'embeddings_dim': dims,
                                     'Generated': [True for i in range(len(generated_mols))]})
        generated_df.iat[0, 3] = False

        if force_unique:
            generated_df = self.compute_unique_smiles(generated_df,
                                                      self.inverse_transform,
                                                      scaled_radius=scaled_radius)
        return generated_df

    def interpolate_smiles(self,
                           smiles: List,
                           num_points: int = 10,
                           scaled_radius=None,
                           force_unique=False,
                           sanitize=True):

        num_points = int(num_points) + 2
        if len(smiles) < 2:
            raise Exception('At-least two or more smiles are expected')

        def linear_interpolate_points(embedding, num_points):
            return np.linspace(embedding[0], embedding[1], num_points)

        result_df = []
        for idx in range(len(smiles) - 1):
            data = pd.DataFrame({'transformed_smiles': [smiles[idx], smiles[idx + 1]]})
            input_embeddings = np.asarray(self.transform(data))

            interp_embeddings = np.apply_along_axis(linear_interpolate_points,
                                                    axis=0,
                                                    arr=input_embeddings,
                                                    num_points=num_points)
            generated_mols = self.inverse_transform(interp_embeddings, sanitize)
            interp_embeddings = interp_embeddings.tolist()

            dims = []
            embeddings = []
            for interp_embedding in interp_embeddings:
                dims.append(input_embeddings.shape)
                interp_embedding = np.asarray(interp_embedding)
                embeddings.append(interp_embedding)

            interp_df = pd.DataFrame({'SMILES': generated_mols,
                                      'embeddings': embeddings,
                                      'embeddings_dim': dims,
                                      'Generated': [True for i in range(num_points)]})

            # Mark the source and desinations as not generated
            interp_df.iat[0, 3] = False
            interp_df.iat[-1, 3] = False

            if force_unique:
                interp_df = self.compute_unique_smiles(interp_df,
                                                       self.inverse_transform,
                                                       scaled_radius=scaled_radius)

            result_df.append(interp_df)

        return pd.concat(result_df)
