#!/usr/bin/env python3

import logging
from functools import partial
from typing import List
import concurrent.futures
from rdkit import Chem

import torch
import pandas as pd
from cuchemcommon.workflow import BaseGenerativeWorkflow, add_jitter

from nemo.collections.chem.models.megamolbart.megatron_bart_model import MegaMolBARTModel

logger = logging.getLogger(__name__)


@add_jitter.register(torch.Tensor)
def _(embedding, radius, cnt, shape):
    if shape is not None:
        embedding = torch.reshape(embedding, (1, shape[0], shape[1])).to(embedding.device)
    permuted_emb = embedding.permute(1, 0, 2)

    distorteds = []
    for i in range(cnt):
        noise = torch.normal(0, radius, permuted_emb.shape).to(embedding.device)
        distorted = (noise + permuted_emb).permute(1, 0, 2)
        distorteds.append(distorted)

    return distorteds


class MegaMolBART(BaseGenerativeWorkflow):

    def __init__(self, model_dir) -> None:
        super().__init__()

        torch.set_grad_enabled(False)  # Testing this instead of `with torch.no_grad():` context since it doesn't exit

        self.device = 'cuda'  # Megatron arg loading seems to only work with GPU
        self.min_jitter_radius = 1.0
        self.model, self.version = self.load_model(model_dir)
        self.max_model_position_embeddings = self.model.max_seq_len
        self.tokenizer = self.model.tokenizer

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

        # TODO: get version from model (self.megamolbart.model)
        return model, '0.2.0'

    def smiles2embedding(self, smiles, pad_length=None):
        """Calculate embedding and padding mask for smiles with optional extra padding

        Params
            smiles: string, input SMILES molecule
            pad_length: optional extra

        Returns
            embedding array and boolean mask
        """

        assert isinstance(smiles, str)
        if pad_length:
            assert pad_length >= len(smiles) + 2

        tokens = self.tokenizer.tokenize([smiles], pad=True)

        # Append to tokens and mask if appropriate
        if pad_length:
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

    def _inverse_transform(self, memory, mem_pad_mask, batch_size, sanitize, k):
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
                        break
                else:
                    break

            logger.debug(f'Sanitized SMILES {g_smiles} added...')
            return g_smiles

    def inverse_transform(self, embeddings, mem_pad_mask, k=1, sanitize=True):
        mem_pad_mask = mem_pad_mask.clone()
        smiles_interp_list = []

        batch_size = 1
        with torch.no_grad():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self._inverse_transform, memory, mem_pad_mask, batch_size, sanitize, k): \
                    memory for memory in embeddings}

                for future in concurrent.futures.as_completed(futures):
                    smiles = futures[future]

                    try:
                        g_smiles = future.result()
                        smiles_interp_list.append(g_smiles)
                    except Exception as exc:
                        logger.warning(f'{smiles.smiles} generated an exception: {exc}')

                # for memory in embeddings:
                #     if isinstance(memory, list):
                #         memory = torch.FloatTensor(memory).cuda()
                #     decode_fn = partial(self.model.model._decode_fn,
                #                         mem_pad_mask=mem_pad_mask.type(torch.LongTensor).cuda(),
                #                         memory=memory)
                #     mol_strs, _ = self.model.sampler.beam_decode(decode_fn,
                #                                                 batch_size=batch_size,
                #                                                 device='cuda',
                #                                                 k=k)
                #     mol_strs = sum(mol_strs, [])  # flatten list

                #     for smiles in mol_strs:
                #         if sanitize:
                #             mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
                #             if mol:
                #                 sanitized_smiles = Chem.MolToSmiles(mol)
                #                 smiles_interp_list.append(sanitized_smiles)
                #                 logger.debug(f'Sanitized SMILES {sanitized_smiles} added...')
                #                 break
                #         smiles_interp_list.append(smiles)

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

        pad_length = max(len(smiles1), len(smiles2)) + 2  # add 2 for start / stop
        embedding1, pad_mask1 = self.smiles2embedding(smiles1,
                                                      pad_length=pad_length)

        embedding2, pad_mask2 = self.smiles2embedding(smiles2,
                                                      pad_length=pad_length)

        scale = torch.linspace(0.0, 1.0, num_interp + 2)[
                1:-1]  # skip first and last because they're the selected molecules
        scale = scale.unsqueeze(0).unsqueeze(-1).cuda()

        interpolated_emb = torch.lerp(embedding1, embedding2, scale).cuda()  # dims: batch, tokens, embedding
        combined_mask = (pad_mask1 & pad_mask2).bool().cuda()

        embeddings = []
        dims = []
        for emb in interpolated_emb.permute(1, 0, 2):
            dims.append(tuple(emb.shape))
            embeddings.append(emb)

        generated_mols = self.inverse_transform(embeddings,
                                      combined_mask,
                                      k=k,
                                      sanitize=sanitize)
        generated_mols = [smiles1] + generated_mols + [smiles2]
        embeddings = [embedding1] + embeddings + [embedding2]
        dims = [tuple(embedding1.shape)] + dims + [tuple(embedding2.shape)]
        return generated_mols, embeddings, combined_mask, dims

    def find_similars_smiles_list(self,
                                  smiles: str,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  force_unique=False,
                                  sanitize=True):
        distance = self._compute_radius(scaled_radius)
        logger.info(f'Sampling {num_requested} around {smiles} with distance {distance}...')

        embedding, pad_mask = self.smiles2embedding(smiles)

        neighboring_embeddings = self.addjitter(embedding, distance, cnt=num_requested)

        generated_mols = self.inverse_transform(neighboring_embeddings,
                                                pad_mask.bool().cuda(),
                                                k=1, sanitize=sanitize)
        if force_unique:
            generated_mols = list(set(generated_mols))

        generated_mols = [smiles] + generated_mols
        neighboring_embeddings = [embedding] + neighboring_embeddings
        return generated_mols, neighboring_embeddings, pad_mask

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             force_unique=False,
                             sanitize=True):
        generated_mols, neighboring_embeddings, pad_mask = \
            self.find_similars_smiles_list(smiles,
                                           num_requested=num_requested,
                                           scaled_radius=scaled_radius,
                                           force_unique=force_unique,
                                           sanitize=sanitize)

        # Rest of the applications and libraries use RAPIDS and cuPY libraries.
        # For interoperability, we need to convert the embeddings to cupy.
        embeddings = []
        dims = []
        for neighboring_embedding in neighboring_embeddings:
            dims.append(tuple(neighboring_embedding.shape))
            embeddings.append(neighboring_embedding.flatten().tolist())

        # Rest of the applications and libraries use RAPIDS and cuPY libraries.
        # For interoperability, we need to convert the embeddings to cupy.
        embeddings = []
        dims = []
        for neighboring_embedding in neighboring_embeddings:
            dims.append(neighboring_embedding.shape)
            embeddings.append(neighboring_embedding.flatten().tolist())

        generated_df = pd.DataFrame({'SMILES': generated_mols,
                                     'embeddings': embeddings,
                                     'embeddings_dim': dims,
                                     'Generated': [True for i in range(len(generated_mols))]})
        generated_df.iat[0, 3] = False

        if force_unique:
            inv_transform_funct = partial(self.inverse_transform,
                                          mem_pad_mask=pad_mask)
            generated_df = self.compute_unique_smiles(generated_df,
                                                      inv_transform_funct,
                                                      scaled_radius=scaled_radius)
        return generated_df

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
        smiles_list = list(result_df['SMILES'])

        return result_df, smiles_list
