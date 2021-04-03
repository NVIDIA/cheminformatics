import logging
import numpy as np
import pandas as pd
from typing import List

import torch
import torch.nn
import pickle
from pathlib import Path
import numpy as np
from functools import partial

from rdkit import Chem

from nvidia.cheminformatics.data import GenerativeWfDao
from nvidia.cheminformatics.data.generative_wf import ChemblGenerativeWfDao
from nvidia.cheminformatics.utils.singleton import Singleton
from nvidia.cheminformatics.wf.generative import BaseGenerativeWorkflow


logger = logging.getLogger(__name__)


class MolBART(BaseGenerativeWorkflow):

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao()) -> None:
        super().__init__(dao)
        max_seq_len = 64
        tokenizer_path = '/models/molbart/mol_opt_tokeniser.pickle'
        model_chk_path = '/models/molbart/az_molbart_pretrain.ckpt'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.bart_model = self.load_model(model_chk_path, self.tokenizer, max_seq_len)

        self.bart_model.to('cuda')

    def load_tokenizer(self, tokenizer_path):
        """Load pickled tokenizer

        Params:
            tokenizer_path: str, path to pickled tokenizer

        Returns:
            MolEncTokeniser tokenizer object
        """

        tokenizer_path = Path(tokenizer_path)

        with open(tokenizer_path, 'rb') as fh:
            tokenizer = pickle.load(fh)

        return tokenizer

    def load_model(self, model_checkpoint_path, tokenizer, max_seq_len):
        """Load saved model checkpoint

        Params:
            model_checkpoint_path: str, path to saved model checkpoint
            tokenizer: MolEncTokeniser tokenizer object
            max_seq_len: int, maximum sequence length

        Returns:
            MolBART trained model
        """
        from molbart.models import BARTModel
        from molbart.decode import DecodeSampler

        sampler = DecodeSampler(tokenizer, max_seq_len)
        pad_token_idx = tokenizer.vocab[tokenizer.pad_token]

        bart_model = BARTModel.load_from_checkpoint(model_checkpoint_path,
                                                    decode_sampler=sampler,
                                                    pad_token_idx=pad_token_idx)
        bart_model.sampler.device = "cuda"
        return bart_model.cuda()

    def smiles2embedding(self, bart_model, smiles, tokenizer, pad_length=None):
        """Calculate embedding and padding mask for smiles with optional extra padding

        Params
            smiles: string, input SMILES molecule
            tokenizer: MolEncTokeniser tokenizer object
            pad_length: optional extra

        Returns
            embedding array and boolean mask
        """

        assert isinstance(smiles, str)
        if pad_length:
            assert pad_length >= len(smiles) + 2

        tokens = tokenizer.tokenise([smiles], pad=True)

        # Append to tokens and mask if appropriate
        if pad_length:
            for i in range(len(tokens['original_tokens'])):
                n_pad = pad_length - len(tokens['original_tokens'][i])
                tokens['original_tokens'][i] += [tokenizer.pad_token] * n_pad
                tokens['pad_masks'][i] += [1] * n_pad

        token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens['original_tokens'])).cuda().T
        pad_mask = torch.tensor(tokens['pad_masks']).bool().cuda().T
        encode_input = {"encoder_input": token_ids, "encoder_pad_mask": pad_mask}

        embedding = bart_model.encode(encode_input)
        torch.cuda.empty_cache()

        return embedding, pad_mask

    def interpolate_molecules(self, smiles1, smiles2, num_interp, tokenizer, k=1):
        """Interpolate between two molecules in embedding space.

        Params
            smiles1: str, input SMILES molecule
            smiles2: str, input SMILES molecule
            num_interp: int, number of molecules to interpolate
            tokenizer: MolEncTokeniser tokenizer object
            k: number of molecules for beam search, default 1. Can increase if there are issues with validity

        Returns
            list of interpolated smiles molecules
        """

        pad_length = max(len(smiles1), len(smiles2)) + 2 # add 2 for start / stop
        embedding1, pad_mask1 = self.smiles2embedding(self.bart_model,
                                                      smiles1,
                                                      tokenizer,
                                                      pad_length=pad_length)

        embedding2, pad_mask2 = self.smiles2embedding(self.bart_model,
                                                      smiles2,
                                                      tokenizer,
                                                      pad_length=pad_length)

        scale = torch.linspace(0.0, 1.0, num_interp+2)[1:-1] # skip first and last because they're the selected molecules
        scale = scale.unsqueeze(0).unsqueeze(-1).cuda()
        interpolated_emb = torch.lerp(embedding1, embedding2, scale).permute(1, 0, 2).cuda()
        combined_mask = (pad_mask1 & pad_mask2).bool().cuda()

        return self.inverse_transform(interpolated_emb, k=k, mem_pad_mask=combined_mask), combined_mask

    def inverse_transform(self, embeddings, k=1, mem_pad_mask=None):
        smiles_interp_list = []

        batch_size = 1 # TODO: parallelize this loop as a batch
        for memory in embeddings:
            decode_fn = partial(self.bart_model._decode_fn,
                                mem_pad_mask=mem_pad_mask,
                                memory=memory)
            mol_strs, log_lhs = self.bart_model.sampler.beam_decode(decode_fn,
                                                                    batch_size=batch_size,
                                                                    k=k)
            mol_strs = sum(mol_strs, []) # flatten list
            for smiles in mol_strs:
                mol = Chem.MolFromSmiles(smiles)
                if (mol is not None) and (smiles not in smiles_interp_list):
                    smiles_interp_list.append(smiles)
                    break

        return smiles_interp_list

    def find_similars_smiles_list(self,
                                  smiles:str,
                                  num_requested:int=10,
                                  radius=0.0001,
                                  force_unique=False):
        embedding, pad_mask = self.smiles2embedding(self.bart_model,
                                                    smiles,
                                                    self.tokenizer)

        neighboring_embeddings = self.addjitter(embedding, radius, cnt=num_requested)

        generated_mols = self.inverse_transform(neighboring_embeddings, k=1, mem_pad_mask=pad_mask.bool().cuda())
        generated_mols = [smiles] + generated_mols
        return generated_mols, neighboring_embeddings, pad_mask

    def find_similars_smiles(self,
                             smiles:str,
                             num_requested:int=10,
                             radius=0.0001,
                             force_unique=False):

        generated_mols, neighboring_embeddings, pad_mask = \
            self.find_similars_smiles_list(smiles,
                                           num_requested=num_requested,
                                           radius=radius,
                                           force_unique=force_unique)

        generated_df = pd.DataFrame({'SMILES': generated_mols,
                                     'Generated': [True for i in range(len(generated_mols))]})
        generated_df.iat[ 0, 1] = False

        if force_unique:
            inv_transform_funct = partial(self.inverse_transform,
                                   mem_pad_mask=pad_mask)
            generated_df = self.compute_unique_smiles(generated_df,
                                                   neighboring_embeddings,
                                                   inv_transform_funct,
                                                   radius=radius)
        return generated_df

    def interpolate_from_smiles(self,
                                smiles:List,
                                num_points:int=10,
                                radius=0.0001,
                                force_unique=False):
        num_points = int(num_points)
        if len(smiles) < 2:
            raise Exception('At-least two or more smiles are expected')

        k = 1
        result_df = []
        for idx in range(len(smiles) - 1):
            interplocated_mol = [smiles[idx]]
            interplocated, combined_mask = self.interpolate_molecules(smiles[idx],
                                                                          smiles[idx + 1],
                                                                          num_points,
                                                                          self.tokenizer,
                                                                          k=k)
            interplocated_mol += interplocated
            interplocated_mol.append(smiles[idx + 1])

            interp_df = pd.DataFrame({'SMILES': interplocated_mol,
                                      'Generated': [True for i in range(len(interplocated_mol))]},
                                      )

            inv_transform_funct = partial(self.inverse_transform,
                                   mem_pad_mask=combined_mask)

            # Mark the source and desinations as not generated
            interp_df.iat[ 0, 1] = False
            interp_df.iat[-1, 1] = False

            if force_unique:
                interp_df = self.compute_unique_smiles(interp_df,
                                                       interplocated_mol,
                                                       inv_transform_funct,
                                                       radius=radius)


            result_df.append(interp_df)

        return pd.concat(result_df)