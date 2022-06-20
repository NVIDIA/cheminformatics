#!/usr/bin/env python3

import logging
import concurrent.futures
from functools import partial
from typing import List

from rdkit import Chem
from rdkit.Chem import PandasTools, CanonSmiles

import torch
import pandas as pd

import sys
import numpy as np
from collections import defaultdict
import random
import logging
from functools import partial
from typing import List
import concurrent.futures

from nemo_chem.models.megamolbart.megatron_bart_model import MegaMolBARTModel
from nemo_chem.models.megamolbart.megatron_bart_latent_model import MegaMolBARTLatentModel

from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo

initialize_model_parallel_for_nemo(
    world_size=1,
    global_rank=0,
    local_rank=0,
    tensor_model_parallel_size=1,
    seed=1234,
)
logger = logging.getLogger(__name__)


class MegaMolBART():
    def __init__(self, model_dir) -> None:
        super().__init__()

        torch.set_grad_enabled(False)
        self.device = 'cuda'
        self.min_jitter_radius = 1.0
        self.model, self.version = self.load_model(model_dir)
        self.max_model_position_embeddings = self.model.max_seq_len
        self.tokenizer = self.model.tokenizer
        self.encoder_type = self.model.model.encoder_type

    def _compute_radius(self, scaled_radius):
        if isinstance(scaled_radius, (int, float)):
            return float(scaled_radius * self.min_jitter_radius)
        else:
            logger.debug(f'Scaled Radius {scaled_radius} Not being used')
            return self.min_jitter_radius

    def add_jitter(self, embedding, radius, cnt, shape=None):
        permuted_emb = embedding
        distorteds = []
        for _ in range(cnt):
            noise = torch.normal(0, radius, permuted_emb.shape).to(embedding.device)
            distorted = (noise + permuted_emb)
            distorteds.append(distorted)

        return distorteds

    def load_model(self, checkpoint_path):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            MegaMolBART trained model
        """
        model = MegaMolBARTModel.restore_from(checkpoint_path)
        model = model.cuda()
        model.eval()
        return model, '0.1.0'

    def smiles2embedding(self, smiles, pad_length=None):
        """Calculate embedding and padding mask for smiles with optional extra padding

        Params
            smiles: string, input SMILES molecule
            pad_length: optional extra

        Returns
            embedding array and boolean mask
        """

        if isinstance(smiles, str):
            smiles = [smiles]

        tokens = self.tokenizer.tokenize(smiles, pad=True)

        # # Append to tokens and mask if appropriate. No Padding for Perceiver
        if pad_length and self.encoder_type != 'perceiver':
            for i in range(len(tokens['original_tokens'])):
                n_pad = pad_length - len(tokens['original_tokens'][i])
                tokens['original_tokens'][i] += [self.tokenizer.pad_token] * n_pad
                tokens['masked_pad_masks'][i] += [1] * n_pad

        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens['original_tokens'])).cuda().T
        pad_mask = torch.tensor(tokens['masked_pad_masks']).bool().cuda().T
        encode_input = {"encoder_input": token_ids, "encoder_pad_mask": pad_mask}

        embedding = self.model.model.encode(encode_input)
        torch.cuda.empty_cache()
        return embedding, pad_mask

    def _inverse_transform_batch(self, memory, mem_pad_mask, k=1, sanitize=True):
        mem_pad_mask = mem_pad_mask.clone()
        smiles_interp_list = []
        (_, batch_size, _) = tuple(memory.size())
        if self.encoder_type == "perceiver":
            mem_pad_mask = torch.zeros((memory.shape[0:2]), dtype=mem_pad_mask.dtype, device=mem_pad_mask.device)
        with torch.no_grad():
            decode_fn = partial(self.model.model._decode_fn,
                                mem_pad_mask=mem_pad_mask.type(torch.LongTensor).cuda(),
                                memory=memory)

            mol_strs, _ = self.model.sampler.beam_decode(decode_fn,
                                                        batch_size=batch_size,
                                                        device='cuda',
                                                        k=k)
            
            mol_strs = sum(mol_strs, [])  # flatten list
            g_smiles = None
            for smiles in mol_strs:
                g_smiles = smiles
                if sanitize:
                    mol = Chem.MolFromSmiles(g_smiles, sanitize=sanitize)
                    if mol:
                        g_smiles = Chem.MolToSmiles(mol)
                smiles_interp_list.append(g_smiles)
        return smiles_interp_list
    
    # Added this function for easier local working with the model
    # TODO as the inverse_transform is used via the sql query for generation this function should take in 
    # either any sized embedding or a list of embeddings and process them via an appropriate batch size
    # z = model.smiles2embedding(smiles) out_smiles = model.embedding2smiles(z)
    def embedding2smiles(self, embeddings, mem_pad_mask, k=1, sanitize=True):
        return self.inverse_transform([embeddings], mem_pad_mask, k, sanitize)[0]

    def inverse_transform(self, embeddings, mem_pad_mask, k=1, sanitize=True):
        smiles_interp_list = []
        with torch.no_grad():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                if type(mem_pad_mask) == list: # To handle database level batching of cheminformatics/benchmark/cuchembench/inference/megamolbart.py embedding_to_smiles(...)
                    futures = {executor.submit(self._inverse_transform_batch, memory, mask.clone(), k, sanitize): memory for memory, mask in zip(embeddings, mem_pad_mask)}
                else:
                    futures = {executor.submit(self._inverse_transform_batch, memory, mem_pad_mask.clone(), k, sanitize): memory for memory in embeddings}
                for future in concurrent.futures.as_completed(futures):
                    smiles = futures[future]
                    try:
                        g_smiles = future.result()
                        smiles_interp_list.append(g_smiles)
                    except Exception as exc:
                        logger.info(f'{type(futures)}, {type(future)}')
                        logger.exception(exc)
        return smiles_interp_list

    def interpolate_molecules(self, smiles1, smiles2, num_interp, tokenizer, k=1, sanitize=True):
        """Interpolate between two molecules in embedding space.

        Params
            smiles1: str, input SMILES molecule
            smiles2: str, input SMILES molecule
            num_interp: int, number of molecules to interpolate
            tokenizer: MolEncTokenizer tokenizer object
            k: number of molecules for beam search, default 1. Can increase if there are issues with validity

        Returns
            list of interpolated smiles molecules
        """
        # add 2 for start / stop
        pad_length = max(len(smiles1), len(smiles2)) + 2
        embedding1, pad_mask1 = self.smiles2embedding(smiles1,
                                                      pad_length=pad_length)

        embedding2, pad_mask2 = self.smiles2embedding(smiles2,
                                                      pad_length=pad_length)
        # skip first and last because they're the selected molecules
        scale = torch.linspace(0.0, 1.0, num_interp + 2)[1:-1]
        scale = scale.unsqueeze(0).unsqueeze(-1).cuda()

        # dims: batch, tokens, embedding
        interpolated_emb = torch.lerp(embedding1, embedding2, scale).cuda()
        if self.encoder_type == 'perceiver':
            combined_mask = torch.zeros((embedding1.shape[0:2]), dtype=pad_mask1.dtype, device=pad_mask1.device).cuda()
        else:
            combined_mask = (pad_mask1 & pad_mask2).bool().cuda()
        embeddings = [interpolated_emb]
        generated_mols = self.inverse_transform(embeddings,
                                      combined_mask, #[combined_mask]*num_interp,
                                      k=k,
                                      sanitize=sanitize)[0] # list of num_interp molecules
        embeddings = []
        dims = []
        for idx in range(num_interp):
            emb = interpolated_emb[:, idx, :]
            emb = emb.unsqueeze(1)
            dims.append(tuple(emb.shape))
            embeddings.append(emb)
        generated_mols = [smiles1] + generated_mols + [smiles2]
        embeddings = [embedding1] + embeddings + [embedding2]
        dims = [tuple(embedding1.shape)] + dims + [tuple(embedding2.shape)]
        return generated_mols, embeddings, combined_mask, dims

    def find_similars_smiles_list(self,
                                  smiles: list,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  force_unique=False,
                                  sanitize=True,
                                  pad_length=None):
        distance = self._compute_radius(scaled_radius)
        logger.info(f'Sampling {num_requested} around {len(smiles)} SMILES with distance {distance}...')

        embedding, pad_mask = self.smiles2embedding(smiles, pad_length=pad_length)
        # emb = NxBxM
        num_molecules = embedding.shape[1]
        neighboring_embeddings = self.add_jitter(embedding, distance, num_requested)
        # neighboring_embeddings = [NxBxM] * num_requested with each using the same pad mask
        generated_mols = self.inverse_transform(neighboring_embeddings,
                                                pad_mask.bool().cuda(),
                                                k=1, sanitize=sanitize)
        # gens = [B] * num_requested
        # need to resize it to [num_requested] * B
        generated_molcules = []
        generated_embeddings = []
        for idx in range(num_molecules):
            molecules = [generated_mols[k][idx] for k in range(len(generated_mols))]
            embs = [nemb[:, idx,:] for nemb in neighboring_embeddings]
            if force_unique:
                molecules = list(set(molecules))
            generated_molcules.append(molecules)
            generated_embeddings.append(embs)

        total_molecules = []
        total_embeddings = []
        for smile, gens in zip(smiles, generated_molcules):
            total_molecules.append([smile] + gens)
        for smile_idx, gens in zip(list(range(embedding.shape[1])), generated_embeddings):
            total_embeddings.append([embedding[:, smile_idx, :]] + gens)

        return total_molecules, total_embeddings, pad_mask

    def find_similars_smiles(self,
                             smiles: list,
                             num_requested: int = 10,
                             scaled_radius=None,
                             force_unique=False,
                             sanitize=True,
                             pad_length=None):
        generated_mols, neighboring_embeddings, pad_mask = \
            self.find_similars_smiles_list(smiles,
                                           num_requested=num_requested,
                                           scaled_radius=scaled_radius,
                                           force_unique=force_unique,
                                           sanitize=sanitize,
                                           pad_length=pad_length)

        # Rest of the applications and libraries use RAPIDS and cuPY libraries.
        # For interoperability, we need to convert the embeddings to cupy.
        dfs = []
        for smile_idx in range(len(smiles)):
            embeddings = []
            dims = []
            for neighboring_embedding in neighboring_embeddings[smile_idx]:
                dims.append(tuple(neighboring_embedding.shape))
                embeddings.append(neighboring_embedding.flatten().tolist())

            generated_df = pd.DataFrame({'SMILES': generated_mols[smile_idx],
                                        'embeddings': embeddings,
                                        'embeddings_dim': dims,
                                        'Generated': [True for i in range(len(generated_mols[smile_idx]))]})
            generated_df.iat[0, 3] = False
            # Removed Force Unique
            dfs.append(generated_df)

        return dfs

    def interpolate_smiles(self,
                           smiles: List,
                           num_points: int = 10,
                           scaled_radius=None,
                           force_unique=False,
                           sanitize=True):
        num_points = int(num_points)
        if len(smiles) < 2:
            raise Exception('At-least two or more smiles are expected')

        k = 1
        result_df = []
        for idx in range(len(smiles) - 1):
            interpolated_mol, interpolated_embeddings, combined_mask, dims = \
                self.interpolate_molecules(smiles[idx],
                                           smiles[idx + 1],
                                           num_points,
                                           self.tokenizer,
                                           k=k,
                                           sanitize=sanitize)

            # Rest of the applications and libraries use RAPIDS and cuPY libraries.
            # For interoperability, we need to convert the embeddings to cupy.
            embeddings = []
            for interpolated_embedding in interpolated_embeddings:
                embeddings.append(interpolated_embedding.flatten().tolist())

            interp_df = pd.DataFrame({'SMILES': interpolated_mol,
                                      'embeddings': embeddings,
                                      'embeddings_dim': dims,
                                      'Generated': [True for i in range(len(interpolated_mol))]})

            inv_transform_funct = partial(self.inverse_transform, mem_pad_mask=combined_mask)

            # Mark the source and desinations as not generated
            interp_df.iat[0, 3] = False
            interp_df.iat[-1, 3] = False

            if force_unique:
                interp_df = self.compute_unique_smiles(interp_df,
                                                       inv_transform_funct,
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
                                                  1,
                                                  shape=embeddings_dim[dup_idx])
                        embeddings[dup_idx] = distored[0]
                interp_df['SMILES'] = embedding_funct(embeddings.to_list())
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
                                                     1,
                                                     shape=embeddings_dim[idx])[0]
                interp_df['SMILES'] = embedding_funct(embeddings.to_list())
                interp_df['embeddings'] = embeddings
            else:
                break

        # Cleanup
        if 'ROMol' in interp_df.columns:
            interp_df = interp_df.drop('ROMol', axis=1)

        return interp_df

#TODO: This class will change once Perceiver is integrated into Nemo_MegaMolBart/dev
# Only works with @(dreidenbach) local LVM implementation
class MegaMolBARTLatent(MegaMolBART):
    def __init__(self, model_dir, noise_mode = 0):
        # super().__init__()

        self.model, self.version = self.load_model(model_dir)
        self.max_model_position_embeddings = self.model.max_seq_len
        self.tokenizer = self.model.tokenizer
        self.encoder_type = self.model.model.encoder_type
        self.sample_logv = -6.0
        self.noise_mode = noise_mode
        # Noise Mode allows the user to set the sampling method
        # When using sample logv "radius" is used as the sample_logv value
        self.min_jitter_radius = 1.0

    def load_model(self, checkpoint_path):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            MegaMolBART trained model
        """
        model = MegaMolBARTLatentModel.restore_from(checkpoint_path)
        model = model.cuda()
        model.eval()
        return model, '0.1.0'
    
    def _set_logv(self, logv):
        self.sample_logv = logv
        return logv
    
    def _compute_radius(self, scaled_radius):
        if isinstance(scaled_radius, (int, float)) and self.noise_mode == 2:
            return self._set_logv(scaled_radius)
        elif isinstance(scaled_radius, (int, float)):
            return float(scaled_radius * self.min_jitter_radius)
        else:
            return self.min_jitter_radius

    def add_jitter(self,  emb_info, radius, cnt):
        distorteds = []
        embedding, z, z_mean, z_logv = emb_info
        if self.noise_mode == 0:
            # Standard Discrete Noise Radius Steps
            for _ in range(cnt):
                noise = torch.normal(0, radius, z_mean.shape).to(z_mean.device)
                distorted = (noise + z_mean)
                distorteds.append(distorted)
        elif self.noise_mode == 1:
            # Radius scales learned Standard Deviation
            for _ in range(cnt):
                e = torch.randn_like(z_mean)
                sample = radius * (e * torch.exp(0.5 * z_logv)) + z_mean
                distorteds.append(sample)
        elif self.noise_mode == 2:
            # Sampled logv based sampling (MIM Only)
            self.sample_logv = radius
            for _ in range(cnt):
                e = torch.randn_like(z_mean)
                sample =(e * torch.exp(0.5 * z_logv)) + z_mean
                distorteds.append(sample)
        elif self.noise_mode == 3:
            # Latent Reconstruction: No Noise
            for _ in range(cnt):
                sample = z_mean
                distorteds.append(sample)
        return distorteds

    def smiles2embedding(self, smiles, pad_length=None):
        """Calculate embedding and padding mask for smiles with optional extra padding

        Params
            smiles: string, input SMILES molecule
            pad_length: optional extra

        Returns
            embedding array and boolean mask
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        tokens = self.tokenizer.tokenize(smiles, mask = False, pad=True)
        # LVM never require padding
        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens['original_tokens'])).cuda().T
        pad_mask = torch.tensor(tokens['masked_pad_masks']).bool().cuda().T
        encode_input = {"encoder_input": token_ids, "encoder_pad_mask": pad_mask}

        embedding = self.model.model.encode(encode_input, chosen_logv = self.sample_logv)
        z, z_mean, z_logv = self.model.model.encode_latent(hidden=embedding, logv_clamp = self.sample_logv)
        torch.cuda.empty_cache()
        return (embedding, z, z_mean, z_logv), pad_mask

    def _inverse_transform_batch(self, memory, mem_pad_mask, k=1, sanitize=True):
        memory = self.model.model.latent2hidden(memory)
        #Above needed for LVM
        mem_pad_mask = mem_pad_mask.clone()
        smiles_interp_list = []
        (_, batch_size, _) = tuple(memory.size())
        if self.encoder_type == "perceiver":
            mem_pad_mask = torch.zeros((memory.shape[0:2]), dtype=mem_pad_mask.dtype, device=mem_pad_mask.device)
        with torch.no_grad():
            decode_fn = partial(self.model.model._decode_fn,
                                mem_pad_mask=mem_pad_mask.type(torch.LongTensor).cuda(),
                                memory=memory)

            mol_strs, _ = self.model.sampler.beam_decode(decode_fn,
                                                        batch_size=batch_size,
                                                        device='cuda',
                                                        k=k)
            
            mol_strs = sum(mol_strs, [])  # flatten list
            g_smiles = None
            for smiles in mol_strs:
                g_smiles = smiles
                if sanitize:
                    mol = Chem.MolFromSmiles(g_smiles, sanitize=sanitize)
                    if mol:
                        g_smiles = Chem.MolToSmiles(mol)
                smiles_interp_list.append(g_smiles)
        return smiles_interp_list

    def find_similars_smiles_list(self,
                                  smiles: list,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  force_unique=False,
                                  sanitize=True,
                                  pad_length=None):
        distance = self._compute_radius(scaled_radius)
        logger.info(f'Sampling {num_requested} around {len(smiles)} SMILES with distance {distance}...')

        embedding_info, pad_mask = self.smiles2embedding(smiles, pad_length=pad_length)
        embedding, z, z_mean, z_logv = embedding_info
        # emb = NxBxM
        num_molecules = z_mean.shape[1]
        neighboring_embeddings = self.add_jitter(embedding_info, distance, num_requested)
        # neigh = [NxBxM] * num_requested
        generated_mols = self.inverse_transform(neighboring_embeddings,
                                                pad_mask.bool().cuda(), #[pad_mask.bool().cuda()]*num_requested,
                                                k=1, sanitize=sanitize)
        # gens = [B] * num_requested
        # need to resize it to [num_requested] * B
        generated_molcules = []
        generated_embeddings = []
        for idx in range(num_molecules):
            molecules = [generated_mols[k][idx] for k in range(len(generated_mols))]
            embs = [nemb[:, idx,:] for nemb in neighboring_embeddings]
            if force_unique:
                molecules = list(set(molecules))
            generated_molcules.append(molecules)
            generated_embeddings.append(embs)

        total_molecules = []
        total_embeddings = []
        for smile, gens in zip(smiles, generated_molcules):
            total_molecules.append([smile] + gens)
        for smile_idx, gens in zip(list(range(embedding.shape[1])), generated_embeddings):
            total_embeddings.append([embedding[:, smile_idx, :]] + gens)

        return total_molecules, total_embeddings, pad_mask

    def interpolate_molecules(self, smiles1, smiles2, num_interp, tokenizer, k=1, sanitize=True):
        """Interpolate between two molecules in embedding space.

        Params
            smiles1: str, input SMILES molecule
            smiles2: str, input SMILES molecule
            num_interp: int, number of molecules to interpolate
            tokenizer: MolEncTokenizer tokenizer object
            k: number of molecules for beam search, default 1. Can increase if there are issues with validity

        Returns
            list of interpolated smiles molecules
        """
        embedding_info1, pad_mask1 = self.smiles2embedding(smiles1)
        _, _, embedding1, _ = embedding_info1
        embedding_info2, pad_mask2 = self.smiles2embedding(smiles2)
        _, _, embedding2, _ = embedding_info2
        # skip first and last because they're the selected molecules
        scale = torch.linspace(0.0, 1.0, num_interp + 2)[1:-1]
        scale = scale.unsqueeze(0).unsqueeze(-1).cuda()

        # dims: batch, tokens, embedding
        interpolated_emb = torch.lerp(embedding1, embedding2, scale).cuda()

        # # radial interpolation MIM only
        # # print('[Interpolated Shape]', interpolated_emb.shape)
        # interpolated_emb_2 = interpolated_emb.squeeze()
        # r = interpolated_emb_2.norm(dim=-1)
        # s = torch.linspace(r[0].item(), r[-1].item(), r.shape[0]).cuda()
        # interpolated_emb = interpolated_emb_2 / r.unsqueeze(-1) * s.unsqueeze(-1)
        # interpolated_emb = interpolated_emb.unsqueeze(0)


        combined_mask = torch.zeros((embedding1.shape[0:2]), dtype=pad_mask1.dtype, device=pad_mask1.device).cuda()
        embeddings = [interpolated_emb]
        generated_mols = self.inverse_transform(embeddings,
                                      combined_mask, #[combined_mask]*num_interp,
                                      k=k,
                                      sanitize=sanitize)[0] # list of num_interp molecules
        embeddings = []
        dims = []
        for idx in range(num_interp):
            emb = interpolated_emb[:, idx, :]
            emb = emb.unsqueeze(1)
            dims.append(tuple(emb.shape))
            embeddings.append(emb)
        generated_mols = [smiles1] + generated_mols + [smiles2]
        embeddings = [embedding1] + embeddings + [embedding2]
        dims = [tuple(embedding1.shape)] + dims + [tuple(embedding2.shape)]
        return generated_mols, embeddings, combined_mask, dims
