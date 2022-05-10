#!/usr/bin/env python3
import logging
from functools import partial
from typing import List
from omegaconf import open_dict

from rdkit.Chem import PandasTools, CanonSmiles

import torch
import pandas as pd

from pytorch_lightning.trainer.trainer import Trainer

from nemo_chem.models.megamolbart import MegaMolBARTModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils.app_state import AppState


logger = logging.getLogger(__name__)


class MegaMolBART():

    def __init__(self, cfg) -> None:
        super().__init__()

        self.device = 'cuda'
        self.min_jitter_radius = 1.0
        self.model, self.version = self.load_model(cfg)
        self.tokenizer = self.model.tokenizer

    def _compute_radius(self, scaled_radius):
        if scaled_radius:
            return float(scaled_radius * self.min_jitter_radius)
        else:
            return self.min_jitter_radius

    def add_jitter(self, embedding, radius, cnt, shape=None):
        if shape is not None:
            embedding = torch.reshape(embedding, (1, shape[0], shape[1]))\
                             .to(embedding.device)

        permuted_emb = embedding.permute(1, 0, 2)
        distorteds = []
        for _ in range(cnt):
            noise = torch.normal(0, radius, permuted_emb.shape)\
                         .to(embedding.device)
            distorted = (noise + permuted_emb).permute(1, 0, 2)
            distorteds.append(distorted)

        return distorteds

    def load_model(self, cfg):
        """Load saved model checkpoint

        Params:
            checkpoint_path: path to nemo checkpoint

        Returns:
            MegaMolBART trained model
        """
        torch.set_grad_enabled(False)
        trainer = Trainer(plugins=NLPDDPPlugin(),
                          gpus=cfg.model.parallel_size,
                          precision=32)
        # app_state = AppState()
        # if cfg.model.parallel_size > 1:
        #     app_state.model_parallel_size = cfg.model.parallel_size
        #     app_state.model_parallel_rank = \
        #         compute_model_parallel_rank(trainer.local_rank,
        #                                     app_state.model_parallel_size)

        # Load config to set precision. WAR for dtype issue in NeMO. This issue
        # in NeMo is expected to be fixed in v1.8.x
        model_config = MegaMolBARTModel.restore_from(cfg.model.model_path,
                                                     trainer=trainer,
                                                     return_config=True)
        with open_dict(model_config):
            model_config.precision = 32

        # Load model for inference
        from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
        model = MegaMolBARTModel.restore_from(restore_path=cfg.model.model_path,
                                              trainer=trainer,
                                              override_config_path=model_config,
                                              save_restore_connector=NLPSaveRestoreConnector())

        model.freeze()

        # TODO: get version from model
        return model, '0.2.0'

    def _transform(self, smi):
        return self.model._transform(smi)
        # token_output = self.tokenizer.tokenize(smi, pad=True)
        # tokens = token_output["original_tokens"]
        # pad_masks = token_output["original_pad_masks"]

        # # import pdb; pdb.set_trace()
        # tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens)).cuda()
        # pad_masks = torch.tensor(pad_masks, device=tokens.device)

        # # resp = self.model.enc_dec_model(tokens,
        # #                                 pad_masks,
        # #                                 None,
        # #                                 None,
        # #                                 output_enc_hidden_only=True)
        # from operator import itemgetter
        # resp = itemgetter("enc_output")(
        #     self(
        #         encoder_input_ids=tokens,
        #         decoder_input_ids=None,
        #         encoder_attn_mask=pad,
        #         decoder_attn_mask=None,
        #         tokentype_ids=None,
        #         lm_labels=None,
        #         enc_hidden_states=None,
        #         output_enc_hidden_only=True,
        #     )
        # )

        # hidden_states, pad_masks = resp["enc_output"], resp["enc_output_mask"]
        # return tokens, hidden_states, pad_masks

    def _inverse_transform(self, tokens, hidden_states, pad_masks):
        return self.model._inverse_transform(tokens, hidden_states, pad_masks)
        # predicted_tokens_ids, _ =  self.model.decode(tokens,
        #                                              pad_masks,
        #                                              None,
        #                                              encoder_hidden_states=hidden_states)
        # predicted_tokens_ids = predicted_tokens_ids.cpu().detach().numpy().tolist()

        # # Prune tokens by eos / padding and convert to SMILES
        # for item, predicted_tokens_ in enumerate(predicted_tokens_ids):
        #     if self.tokenizer.eos_id in predicted_tokens_:
        #         idx = predicted_tokens_.index(self.tokenizer.eos_id)
        #         predicted_tokens_ids[item] = predicted_tokens_[:idx]
        #     else:
        #         # NB: this is slightly different from previous version in that pad tokens can be in the middle of sequence
        #         predicted_tokens_ids[item] = [id for id in predicted_tokens_ if id != self.tokenizer.pad_id]

        # predicted_tokens_text = self.tokenizer.ids_to_tokens(predicted_tokens_ids)
        # sampled_smiles = self.tokenizer.tokens_to_text(predicted_tokens_text)
        # return sampled_smiles

    def smiles2embedding(self, smi):
        """Calculate embedding and padding mask for smiles with optional extra padding

        Params
            smi: string, input SMILES molecule

        Returns
            embedding array and boolean mask
        """
        tokens, hidden_states, pad_masks = self._transform(smi)
        return tokens, hidden_states, pad_masks

    def embedding2smiles(self, tokens, hidden_states, pad_masks):
        pad_masks = pad_masks.clone()
        smi = self._inverse_transform(tokens, hidden_states, pad_masks)
        return smi

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
        tokens1, hidden_states1, pad_masks1 = self.smiles2embedding(smiles1)
        tokens2, hidden_states2, pad_masks2 = self.smiles2embedding(smiles2)

        # skip first and last because they're the selected molecules
        scale = torch.linspace(0.0, 1.0, num_interp + 2)[1:-1]
        scale = scale.unsqueeze(0).unsqueeze(-1).cuda()

        # dims: batch, tokens, embedding
        interpolated_emb = torch.lerp(hidden_states1, hidden_states2, scale).cuda()
        combined_mask = (pad_masks1 & pad_masks2).bool().cuda()

        embeddings = []
        dims = []
        for emb in interpolated_emb.permute(1, 0, 2):
            dims.append(tuple(emb.shape))
            embeddings.append(emb)

        generated_mols = self.embedding2smiles(tokens1,
                                               embeddings,
                                               combined_mask,
                                               k=k,
                                               sanitize=sanitize)
        generated_mols = [smiles1] + generated_mols + [smiles2]
        embeddings = [hidden_states1] + embeddings + [hidden_states2]
        dims = [tuple(hidden_states1.shape)] + dims + [tuple(hidden_states2.shape)]
        return generated_mols, embeddings, combined_mask, dims

    def find_similars_smiles_list(self,
                                  smiles: str,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  force_unique=False,
                                  sanitize=True,
                                  pad_length=None):
        distance = self._compute_radius(scaled_radius)
        logger.info(f'Sampling {num_requested} around {smiles} with distance {distance}...')

        token, hidden_state, pad_mask = self.smiles2embedding(smiles)

        neighboring_embeddings = self.add_jitter(hidden_state, distance, num_requested)

        generated_mols = self.embedding2smiles(token,
                                               neighboring_embeddings,
                                               pad_mask)
        if force_unique:
            generated_mols = list(set(generated_mols))

        generated_mols = [smiles] + generated_mols
        neighboring_embeddings = [hidden_state] + neighboring_embeddings
        return generated_mols, neighboring_embeddings, pad_mask

    def find_similars_smiles(self,
                             smiles: str,
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
        embeddings = []
        dims = []
        for neighboring_embedding in neighboring_embeddings:
            dims.append(tuple(neighboring_embedding.shape))
            embeddings.append(neighboring_embedding.flatten().tolist())
        generated_df = pd.DataFrame({'SMILES': generated_mols,
                                     'embeddings': embeddings,
                                     'embeddings_dim': dims,
                                     'Generated': [True for i in range(len(generated_mols))]})
        generated_df.iat[0, 3] = False

        if force_unique:
            inv_transform_funct = partial(self.embedding2smiles,
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

            inv_transform_funct = partial(self.embedding2smiles, mem_pad_mask=combined_mask)

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
