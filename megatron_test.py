#!/usr/bin/env python3

import time
import sys
sys.path.insert(0, "/opt/MolBART/megatron_molbart")
sys.path.insert(0, "/opt/MolBART/")

import torch
from torch import nn

from pathlib import Path
from functools import partial
from rdkit import Chem

from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from megatron import get_args

from megatron_bart import MegatronBART

from molbart.decoder import DecodeSampler
from molbart.tokeniser import MolEncTokeniser

# from molbart.util import *
REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
DEFAULT_CHEM_TOKEN_START = 272


def model_provider(args):
    tokenizer = MolEncTokeniser.from_vocab_file(
            '/models/molbart/bart_vocab.txt',
            REGEX,
            DEFAULT_CHEM_TOKEN_START)

    vocab_size = len(tokenizer)

    MAX_SEQ_LEN = 512
    pad_token_idx = tokenizer.vocab[tokenizer.pad_token]
    sampler = DecodeSampler(tokenizer, MAX_SEQ_LEN)

    model = MegatronBART(
        sampler,
        pad_token_idx,
        vocab_size,
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.hidden_size * 4,
        MAX_SEQ_LEN,
        dropout=0.1,
        )
    return model.cuda(), tokenizer

args = {
    'num_layers': 4,
    'hidden_size': 256,
    'num_attention_heads': 8,
    'max_position_embeddings': 512,
    'tokenizer_type': 'GPT2BPETokenizer',
    'vocab_file': '/models/molbart/bart_vocab.txt',
    'load': '/models/molbart/'
}

num_loops = 10
start_time = time.time()
with torch.no_grad():
    initialize_megatron(args_defaults=args)
    args = get_args()
    model, tokenizer = model_provider(args)

    load_checkpoint(model, None, None)

    cnt = num_loops
    while cnt > 0:
        loop_start = time.time()

        smiles = 'CC(=O)Nc1ccc(O)cc1'
        tokens = tokenizer.tokenise([smiles], pad=True)

        token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens['original_tokens'])).cuda()
        pad_mask = torch.tensor(tokens['masked_pad_masks']).bool().cuda()

        encode_input = {"encoder_input": token_ids, "encoder_pad_mask": pad_mask}
        embedding = model.encode(encode_input)
        mem_mask = pad_mask.clone()

        (_, batch_size, _) = tuple(embedding.size())
        smiles_interp_list = []

        decode_fn = partial(model._decode_fn,
                            memory=embedding,
                            mem_pad_mask=mem_mask.type(torch.LongTensor).cuda())

        mol_strs, log_lhs = model.sampler.beam_decode(decode_fn,
                                                      batch_size,
                                                      device='cuda',
                                                      k=1)
        print('mol_strs', len(mol_strs))

        # mol_strs = tokenizer.detokenise(mol_strs)
        for smiles in mol_strs:
            print('-----------', smiles[0][:20])
            #mol = Chem.MolFromSmiles(smiles)
            #if (mol is not None) and (smiles not in smiles_interp_list):
            #    smiles_interp_list.append(smiles)
            #    break

        print(smiles_interp_list)
        loop_time = time.time() - loop_start
        print('Single time', loop_time)

        #smiles = mol_strs[0]

        #print('-----------', smiles[:20])
        #mol = Chem.MolFromSmiles(smiles)
        #if (mol is not None) and (smiles not in smiles_interp_list):
        #    smiles_interp_list.append(smiles)

        #print(smiles_interp_list)
        cnt -= 1
total = time.time() - start_time

print('Total', total)
print('Avg', total/num_loops)
