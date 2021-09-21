#!/usr/bin/env python3

import logging
from functools import partial
from pathlib import Path
from typing import List
from rdkit import Chem

import torch
import pandas as pd
from checkpointing import load_checkpoint
from cuchemcommon.workflow import BaseGenerativeWorkflow, add_jitter
from decoder import DecodeSampler
from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron_bart import MegatronBART
from tokenizer import MolEncTokenizer
from util import (REGEX, DEFAULT_CHEM_TOKEN_START, DEFAULT_MAX_SEQ_LEN,
                  DEFAULT_VOCAB_PATH, CHECKPOINTS_DIR,
                  DEFAULT_NUM_LAYERS, DEFAULT_D_MODEL, DEFAULT_NUM_HEADS)

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

    def __init__(self,
                 max_seq_len=DEFAULT_MAX_SEQ_LEN,
                 vocab_path=DEFAULT_VOCAB_PATH,
                 regex=REGEX,
                 default_chem_token_start=DEFAULT_CHEM_TOKEN_START,
                 checkpoints_dir=CHECKPOINTS_DIR,
                 num_layers=DEFAULT_NUM_LAYERS,
                 hidden_size=DEFAULT_D_MODEL,
                 num_attention_heads=DEFAULT_NUM_HEADS,
                 decoder_max_seq_len=None) -> None:
        super().__init__()

        torch.set_grad_enabled(False)  # Testing this instead of `with torch.no_grad():` context since it doesn't exit

        self.device = 'cuda'  # Megatron arg loading seems to only work with GPU
        self.min_jitter_radius = 1.0
        self.max_model_position_embeddings = max_seq_len

        args = {
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'max_position_embeddings': self.max_model_position_embeddings,
            'tokenizer_type': 'GPT2BPETokenizer',
            'vocab_file': vocab_path,
            'load': checkpoints_dir
        }

        with torch.no_grad():
            initialize_megatron(args_defaults=args, ignore_unknown_args=True)
            args = get_args()
            self.tokenizer = self.load_tokenizer(args.vocab_file, regex, default_chem_token_start)
            self.model = self.load_model(args, self.tokenizer, decoder_max_seq_len)

    def _compute_radius(self, scaled_radius):  # TODO REMOVE
        if scaled_radius:
            return float(scaled_radius * self.min_jitter_radius)
        else:
            return self.min_jitter_radius

    def load_tokenizer(self, tokenizer_vocab_path, regex, default_chem_token_start):
        """Load tokenizer from vocab file

        Params:
            tokenizer_vocab_path: str, path to tokenizer vocab

        Returns:
            MolEncTokenizer tokenizer object
        """

        tokenizer_vocab_path = Path(tokenizer_vocab_path)
        tokenizer = MolEncTokenizer.from_vocab_file(
            tokenizer_vocab_path,
            regex,
            default_chem_token_start)

        return tokenizer

    def load_model(self, args, tokenizer, decoder_max_seq_len=None):
        """Load saved model checkpoint

        Params:
            tokenizer: MolEncTokenizer tokenizer object
            decoder_max_seq_len: int, maximum sequence length
            args: Megatron initialized arguments

        Returns:
            MegaMolBART trained model
        """

        vocab_size = len(tokenizer)
        pad_token_idx = tokenizer.vocab[tokenizer.pad_token]

        # TODO how to handle length overrun for batch processing
        if not decoder_max_seq_len:
            decoder_max_seq_len = args.max_position_embeddings

        sampler = DecodeSampler(tokenizer, decoder_max_seq_len)
        model = MegatronBART(
            sampler,
            pad_token_idx,
            vocab_size,
            args.hidden_size,
            args.num_layers,
            args.num_attention_heads,
            args.hidden_size * 4,
            args.max_position_embeddings,
            dropout=0.1,
        )
        self.iteration = load_checkpoint(model, None, None)
        model = model.cuda()
        model.eval()
        return model

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

        embedding = self.model.encode(encode_input)
        torch.cuda.empty_cache()
        return embedding, pad_mask

    def inverse_transform(self, embeddings, mem_pad_mask, k=1, sanitize=True):
        mem_pad_mask = mem_pad_mask.clone()
        smiles_interp_list = []

        batch_size = 1  # TODO: parallelize this loop as a batch
        with torch.no_grad():
            for memory in embeddings:

                if isinstance(memory, list):
                    memory = torch.FloatTensor(memory).cuda()

                decode_fn = partial(self.model._decode_fn,
                                    mem_pad_mask=mem_pad_mask.type(torch.LongTensor).cuda(),
                                    memory=memory)

                mol_strs, _ = self.model.sampler.beam_decode(decode_fn,
                                                             batch_size=batch_size,
                                                             device='cuda',
                                                             k=k)
                mol_strs = sum(mol_strs, [])  # flatten list

                for smiles in mol_strs:
                    if sanitize:
                        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
                        if mol:
                            sanitized_smiles = Chem.MolToSmiles(mol)
                            smiles_interp_list.append(sanitized_smiles)
                            logger.debug(f'Sanitized SMILES {sanitized_smiles} added...')
                            break
                    smiles_interp_list.append(smiles)

        return smiles_interp_list

    def interpolate_molecules(self, smiles1, smiles2, num_interp, tokenizer, k=1, sanitize=False):
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
            dims.append(emb.shape)
            embeddings.append(emb)

        generated_mols = self.inverse_transform(embeddings,
                                      combined_mask,
                                      k=k,
                                      sanitize=sanitize)
        generated_mols = [smiles1] + generated_mols + [smiles2]
        embeddings = [embedding1] + embeddings + [embedding2]
        dims = [embedding1.shape] + dims + [embedding2.shape]
        return generated_mols, embeddings, combined_mask, dims

    def find_similars_smiles_list(self,
                                  smiles: str,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  force_unique=False,
                                  sanitize=False):
        distance = self._compute_radius(scaled_radius)
        logger.info(f'Computing with distance {distance}...')

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
                             sanitize=False):
        generated_mols, neighboring_embeddings, pad_mask = \
            self.find_similars_smiles_list(smiles,
                                           num_requested=num_requested,
                                           scaled_radius=scaled_radius,
                                           force_unique=force_unique)

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
                                sanitize=False):
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
                embeddings.append(interpolated_embedding.cpu())

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
