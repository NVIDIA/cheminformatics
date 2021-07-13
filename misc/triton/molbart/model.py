# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sys
import json

import os
import shutil
from subprocess import run, Popen

from typing import List

import numpy as np
import pandas as pd
import torch
from functools import singledispatch

import logging
import pandas as pd
from typing import List

import torch
import torch.nn
import pickle
from pathlib import Path
import numpy as np
from functools import partial

from rdkit import Chem
from rdkit.Chem import Draw, PandasTools


import triton_python_backend_utils as pb_utils

CDDD_DEFAULT_MODLE_LOC = '/models/cddd'

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



class TritonPythonModel:

    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        max_seq_len = 64
        self.download_cddd_models()
        tokenizer_path = '/models/molbart/mol_opt_tokeniser.pickle'
        model_chk_path = '/models/molbart/az_molbart_pretrain.ckpt'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.bart_model = self.load_model(model_chk_path, self.tokenizer, max_seq_len)
        self.bart_model.to('cuda')


    def execute(self, requests):

        output0_dtype = self.output0_dtype

        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_smiles = in_0.as_numpy()[0].decode()
            print('processing', input_smiles)
            generated_smiles, neighboring_embeddings, pad_mask = \
                self.find_similars_smiles_list(input_smiles,
                                               num_requested=10,
                                               force_unique=True)

            out_0 = np.array(generated_smiles).astype(np.object)

            out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                           out_0.astype(output0_dtype))

            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')


    def load_tokenizer(self, tokenizer_path):
        """Load pickled tokenizer

        Params:
            tokenizer_path: str, path to pickled tokenizer

        Returns:
            MolEncTokenizer tokenizer object
        """

        tokenizer_path = Path(tokenizer_path)

        with open(tokenizer_path, 'rb') as fh:
            tokenizer = pickle.load(fh)

        return tokenizer

    def load_model(self, model_checkpoint_path, tokenizer, max_seq_len):
        """Load saved model checkpoint

        Params:
            model_checkpoint_path: str, path to saved model checkpoint
            tokenizer: MolEncTokenizer tokenizer object
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

    def download_cddd_models(self, target_dir=CDDD_DEFAULT_MODLE_LOC):
        """
        Downloads CDDD model
        """

        if os.path.exists(os.path.join(target_dir, 'default_model', 'hparams.json')):
            print('Directory already exists. To re-download please delete', target_dir)
            return os.path.join(target_dir, 'default_model')
        else:
            shutil.rmtree(os.path.join(target_dir, 'default_model'), ignore_errors=True)

        download_script = '/opt/cddd/download_default_model.sh'
        if not os.path.exists(download_script):
            download_script = '/tmp/download_default_model.sh'
            run(['bash', '-c',
                'wget --quiet -O %s https://raw.githubusercontent.com/jrwnter/cddd/master/download_default_model.sh && chmod +x %s' % (download_script, download_script)])

        run(['bash', '-c',
            'mkdir -p %s && cd %s; %s' % (target_dir, target_dir, download_script)],
            check=True)

        return os.path.join(target_dir, 'default_model')

    def addjitter(self,
                  embedding,
                  radius,
                  cnt=1):
        return add_jitter(embedding, radius, cnt)

    def compute_unique_smiles(self,
                              interp_df,
                              embeddings,
                              embedding_funct,
                              radius=0.5):
        """
        Identify duplicate SMILES and distorts the embedding. The input df
        must have columns 'SMILES' and 'Generated' at 0th and 1st position.
        'Generated' colunm must contain boolean to classify SMILES into input
        SMILES(False) and generated SMILES(True).

        This function does not make any assumptions about order of embeddings.
        Instead it simply orders the df by SMILES to identify the duplicates.
        """

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
                smiles = embedding_funct(embeddings)
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
                smiles = embedding_funct(embeddings)
            else:
                break

        # Cleanup
        if 'ROMol' in interp_df.columns:
            interp_df = interp_df.drop('ROMol', axis=1)

        return interp_df