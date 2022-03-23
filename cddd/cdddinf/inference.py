import logging
import itertools
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, CanonSmiles

from cddd.inference import InferenceModel

from .utils import download_cddd_models
from .utils import Singleton

log = logging.getLogger(__name__)


class CdddInference():
    '''
    Inferences CDDD model for generating SMILES
    '''
    __metaclass__ = Singleton

    def __init__(self) -> None:
        self.default_model_loc = download_cddd_models()
        self.min_jitter_radius = 0.5

        self.model = InferenceModel(self.default_model_loc,
                                   use_gpu=True,
                                   cpu_threads=5)

    def __len__(self):
        return self.model.hparams.emb_size

    def _compute_radius(self, scaled_radius):
        if scaled_radius:
            return float(scaled_radius * self.min_jitter_radius)
        else:
            return self.min_jitter_radius

    def add_jitter(self, embedding, radius, cnt):
        distorteds = []
        for _ in range(cnt):
            noise = np.random.normal(0, radius, embedding.shape)
            distorted = noise + embedding
            distorteds.append(distorted)
        return distorteds

    def embedding_to_smiles(self, embeddings, sanitize=True):
        embeddings = np.asarray(embeddings)
        smiles = self.model.emb_to_seq(embeddings)

        if isinstance(smiles, str):
            smiles = [smiles]
        if sanitize:
            gsmiles = []
            for smi in smiles:
                mol = Chem.MolFromSmiles(smi, sanitize=sanitize)
                if mol:
                    smi = Chem.MolToSmiles(mol)
                gsmiles.append(smi)

            smiles = gsmiles
        return smiles

    def smiles_to_embedding(self, smiles: str, padding: int):
        embedding = self.model.seq_to_emb(smiles).squeeze()
        return embedding

    def find_similars_smiles_list(self,
                                  smiles: str,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  sanitize=True,
                                  force_unique=False):

        radius = self._compute_radius(scaled_radius)
        embedding = self.model.seq_to_emb(smiles).squeeze()
        embeddings = self.add_jitter(embedding, radius, num_requested)

        neighboring_embeddings = np.concatenate(
            [embedding.reshape(1, embedding.shape[0]), embeddings])
        embeddings = [embedding] + embeddings
        generated_mols = self.embedding_to_smiles(neighboring_embeddings, sanitize)

        return generated_mols, embeddings

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             sanitize=True,
                             force_unique=False):
        generated_mols, neighboring_embeddings =\
            self.find_similars_smiles_list(smiles,
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
                                     'Generated': itertools.repeat(True, len(generated_mols))})
        generated_df.iat[0, 3] = False

        if force_unique:
            generated_df = self.compute_unique_smiles(generated_df,
                                                      self.embedding_to_smiles,
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
            raise Exception('Interpolation requires two or more smiles')

        def linear_interpolate_points(embedding, num_points):
            return np.linspace(embedding[0], embedding[1], num_points)

        result_df = []
        for idx in range(len(smiles) - 1):
            data = pd.DataFrame({'transformed_smiles': [smiles[idx], smiles[idx + 1]]})

            input_embeddings = np.asarray(
                self.model.seq_to_emb(data['transformed_smiles']).squeeze())
            interp_embeddings = np.apply_along_axis(
                linear_interpolate_points,
                axis=0,
                arr=input_embeddings,
                num_points=num_points)

            generated_mols = self.embedding_to_smiles(interp_embeddings, sanitize)
            interp_embeddings = interp_embeddings.tolist()

            dims = []
            embeddings = []
            for interp_embedding in interp_embeddings:
                dims.append(input_embeddings.shape)
                interp_embedding = np.asarray(interp_embedding)
                embeddings.append(interp_embedding)

            interp_df = pd.DataFrame(\
                {'SMILES': generated_mols,
                 'embeddings': embeddings,
                 'embeddings_dim': dims,
                 'Generated': [True for i in range(num_points)]})

            # Mark the source and desinations as not generated
            interp_df.iat[0, 3] = False
            interp_df.iat[-1, 3] = False

            if force_unique:
                interp_df = self.compute_unique_smiles(
                    interp_df,
                    scaled_radius=scaled_radius)

            result_df.append(interp_df)

        result_df = pd.concat(result_df)
        smile_list = list(result_df['SMILES'])

        return result_df, smile_list

    def compute_unique_smiles(self,
                              interp_df,
                              embedding_funct,
                              scaled_radius=None):
        """
        Identify duplicate SMILES and distorts the embedding. The input df
        must have columns 'SMILES' and 'Generated' at 0th and 1st position.
        'Generated' colunm must contain boolean to classify SMILES into input
        SMILES(False) and generated SMILES(True).

        This function does not make any assumptions about order of embeddings.
        Instead it simply orders the df by SMILES to identify the duplicates.
        """

        distance = self._compute_radius(scaled_radius)
        embeddings = interp_df['embeddings']
        embeddings_dim = interp_df['embeddings_dim']
        for _, row in interp_df.iterrows():
            smile_string = row['SMILES']
            try:
                canonical_smile = CanonSmiles(smile_string)
            except:
                # If a SMILES cannot be canonicalized, just use the original
                canonical_smile = smile_string

            row['SMILES'] = canonical_smile

        for i in range(5):
            smiles = interp_df['SMILES'].sort_values()
            duplicates = set()
            for idx in range(0, smiles.shape[0] - 1):
                if smiles.iat[idx] == smiles.iat[idx + 1]:
                    duplicates.add(smiles.index[idx])
                    duplicates.add(smiles.index[idx + 1])

            if len(duplicates) > 0:
                for dup_idx in duplicates:
                    if interp_df.iat[dup_idx, 3]:
                        # add jitter to generated molecules only
                        distored = self.add_jitter(embeddings[dup_idx],
                                                  distance,
                                                  1)
                        embeddings[dup_idx] = distored[0]
                interp_df['SMILES'] = self.embedding_to_smiles(embeddings.to_list())
                interp_df['embeddings'] = embeddings
            else:
                break

        # Ensure all generated molecules are valid.
        for i in range(5):
            PandasTools.AddMoleculeColumnToFrame(interp_df, 'SMILES')
            invalid_mol_df = interp_df[interp_df['ROMol'].isnull()]

            if not invalid_mol_df.empty:
                invalid_index = invalid_mol_df.index.to_list()
                for idx in invalid_index:
                    embeddings[idx] = self.add_jitter(embeddings[idx],
                                                     distance,
                                                     1)[0]
                interp_df['SMILES'] = self.embedding_to_smiles(embeddings.to_list())
                interp_df['embeddings'] = embeddings
            else:
                break

        # Cleanup
        if 'ROMol' in interp_df.columns:
            interp_df = interp_df.drop('ROMol', axis=1)

        return interp_df