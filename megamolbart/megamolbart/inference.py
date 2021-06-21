#!/usr/bin/env python3

import logging
import pandas as pd
import torch

from typing import List
from functools import partial
from pathlib import Path

from megatron.initialize import initialize_megatron
from megatron import get_args

from megatron_bart import MegatronBART
from checkpointing import load_checkpoint
from decoder import DecodeSampler
from tokenizer import MolEncTokenizer
from util import (REGEX, DEFAULT_CHEM_TOKEN_START, DEFAULT_MAX_SEQ_LEN, 
                  DEFAULT_VOCAB_PATH, CHECKPOINTS_DIR, 
                  DEFAULT_NUM_LAYERS, DEFAULT_D_MODEL, DEFAULT_NUM_HEADS)

from cuchemcommon.workflow import BaseGenerativeWorkflow, add_jitter

logger = logging.getLogger(__name__)


@add_jitter.register(torch.Tensor)
def _(embedding, radius, cnt):
    permuted_emb = embedding.permute(1, 0, 2)
    noise = torch.normal(0,  radius, (cnt,) + permuted_emb.shape[1:]).to(embedding.device)

    return (noise + permuted_emb).permute(1, 0, 2)


def clean_smiles_list(smiles_list, standardize=True):
    """Ensure SMILES are valid and unique. Optionally standardize them."""

    smiles_clean_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, sanitize=standardize)
        if mol:
            sanitized_smiles = Chem.MolToSmiles(mol)
            if sanitized_smiles not in smiles_clean_list:
                smiles_clean_list.append(sanitized_smiles)

    if len(smiles_clean_list) == 0:
        smiles_clean_list = [np.NaN]
    return smiles_clean_list


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

        torch.set_grad_enabled(False) # Testing this instead of `with torch.no_grad():` context since it doesn't exit
        
        self.device = 'cuda' # Megatron arg loading seems to only work with GPU
        self.min_jitter_radius = 2.1 # TODO adjust this once model is trained
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

        initialize_megatron(args_defaults=args, ignore_unknown_args=True)
        args = get_args()
        self.tokenizer = self.load_tokenizer(args.vocab_file, regex, default_chem_token_start)
        self.model = self.load_model(args, self.tokenizer, decoder_max_seq_len)

    def _compute_radius(self, scaled_radius): # TODO REMOVE
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
        load_checkpoint(model, None, None)
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

    def inverse_transform(self, embeddings, model, mem_pad_mask, k=1, sanitize=True):
        mem_pad_mask = mem_pad_mask.clone()
        smiles_interp_list = []

        batch_size = 1 # TODO: parallelize this loop as a batch
        for memory in embeddings.permute(1, 0, 2):

            decode_fn = partial(model._decode_fn,
                                mem_pad_mask=mem_pad_mask.type(torch.LongTensor).cuda(),
                                memory=memory)

            mol_strs, log_lhs = model.sampler.beam_decode(decode_fn,
                                                                batch_size=batch_size,
                                                                device='cuda',
                                                                k=k)
            mol_strs = sum(mol_strs, []) # flatten list

            # TODO: add back sanitization and validity checking once model is trained
            logger.warn('WARNING: MOLECULE VALIDATION AND SANITIZATION CURRENTLY DISABLED')
            for smiles in mol_strs:
                # mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
                # if mol:
                #     sanitized_smiles = Chem.MolToSmiles(mol)
                #     if sanitized_smiles not in smiles_interp_list:
                #         smiles_interp_list.append(sanitized_smiles)
                #         break
                smiles_interp_list.append(smiles)

        return smiles_interp_list

    def interpolate_molecules(self, smiles1, smiles2, num_interp, tokenizer, k=1):
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

        scale = torch.linspace(0.0, 1.0, num_interp+2)[1:-1]  # skip first and last because they're the selected molecules
        scale = scale.unsqueeze(0).unsqueeze(-1).cuda()

        interpolated_emb = torch.lerp(embedding1, embedding2, scale).cuda()  # dims: batch, tokens, embedding
        combined_mask = (pad_mask1 & pad_mask2).bool().cuda()

        return self.inverse_transform(interpolated_emb, self.model, k=k, mem_pad_mask=combined_mask, sanitize=True), combined_mask

    def find_similars_smiles_list(self,
                                  smiles:str,
                                  num_requested:int=10,
                                  scaled_radius=None,
                                  force_unique=False):
        distance = self._compute_radius(scaled_radius)

        embedding, pad_mask = self.smiles2embedding(smiles)

        neighboring_embeddings = self.addjitter(embedding, distance, cnt=num_requested)

        generated_mols = self.inverse_transform(embeddings=neighboring_embeddings, model=self.model,
                                                k=1, mem_pad_mask=pad_mask.bool().cuda(), sanitize=True)

        generated_mols = [smiles] + generated_mols
        return generated_mols, neighboring_embeddings, pad_mask

    def find_similars_smiles(self,
                             smiles:str,
                             num_requested:int=10,
                             scaled_radius=None,
                             force_unique=False):
        distance = self._compute_radius(scaled_radius)

        generated_mols, neighboring_embeddings, pad_mask = \
            self.find_similars_smiles_list(smiles,
                                           num_requested=num_requested,
                                           scaled_radius=distance,
                                           force_unique=force_unique)

        generated_df = pd.DataFrame({'SMILES': generated_mols,
                                     'Generated': [True for i in range(len(generated_mols))]})
        generated_df.iat[0, 1] = False

        if force_unique:
            inv_transform_funct = partial(self.inverse_transform,
                                   mem_pad_mask=pad_mask)
            generated_df = self.compute_unique_smiles(generated_df,
                                                   neighboring_embeddings,
                                                   inv_transform_funct,
                                                   radius=distance)

        smile_list = list(generated_df['SMILES'])

        return generated_df, smile_list

    def interpolate_from_smiles(self,
                                smiles:List,
                                num_points:int=10,
                                scaled_radius=None,
                                force_unique=False):
        distance = self._compute_radius(scaled_radius)
        num_points = int(num_points)
        if len(smiles) < 2:
            raise Exception('At-least two or more smiles are expected')

        k = 1
        result_df = []
        for idx in range(len(smiles) - 1):
            interpolated_mol = [smiles[idx]]
            interpolated, combined_mask = self.interpolate_molecules(smiles[idx],
                                                                          smiles[idx + 1],
                                                                          num_points,
                                                                          self.tokenizer,
                                                                          k=k)
            interpolated_mol += interpolated
            interpolated_mol.append(smiles[idx + 1])

            interp_df = pd.DataFrame({'SMILES': interpolated_mol,
                                      'Generated': [True for i in range(len(interpolated_mol))]})

            inv_transform_funct = partial(self.inverse_transform, mem_pad_mask=combined_mask)

            # Mark the source and desinations as not generated
            interp_df.iat[ 0, 1] = False
            interp_df.iat[-1, 1] = False

            if force_unique:
                interp_df = self.compute_unique_smiles(interp_df,
                                                       interpolated_mol,
                                                       inv_transform_funct,
                                                       radius=distance)

            result_df.append(interp_df)

        result_df = pd.concat(result_df)
        smile_list = list(result_df['SMILES'])

        return result_df, smile_list

