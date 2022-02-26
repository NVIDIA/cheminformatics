import logging
import os
import grpc
import pandas as pd

from typing import List

from generativesampler_pb2_grpc import GenerativeSamplerStub
from generativesampler_pb2 import GenerativeSpec, EmbeddingList, GenerativeModel

from cuchemcommon.data import GenerativeWfDao
from cuchemcommon.data.generative_wf import ChemblGenerativeWfDao
from cuchemcommon.utils.singleton import Singleton
from cuchemcommon.workflow import BaseGenerativeWorkflow

# Check if all these are needed:
from cuchemcommon.fingerprint import MorganFingerprint
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import cupy as cp
import pickle
from pathlib import Path
import numpy as np
from functools import partial
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, RDKFingerprint
from rdkit.DataStructs import FingerprintSimilarity
from cuml import Lasso, Ridge #LinearRegression
from cuml.metrics import mean_squared_error
from math import sqrt

logger = logging.getLogger(__name__)
PAD_TOKEN = 0 # TODO: use tokenizer.pad_token instead


class MegatronMolBART(BaseGenerativeWorkflow):
    __metaclass__ = Singleton

    def __init__(self, dao: GenerativeWfDao = ChemblGenerativeWfDao(None)) -> None:
        super().__init__(dao)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu' 
        self.min_jitter_radius = 1
        self.channel = grpc.insecure_channel('megamolbart:50051')
        self.stub = GenerativeSamplerStub(self.channel)

    def is_ready(self, timeout: int = 10) -> bool:
        try:
            grpc.channel_ready_future(self.channel).result(timeout=timeout)
            logger.info('Megatron MolBART is ready')
            return True
        except grpc.FutureTimeoutError:
            logger.warning('Megatron MolBART is not reachable.')
            return False

    def smiles_to_embedding(self,
                            smiles: str,
                            padding: int,
                            scaled_radius=None,
                            num_requested: int = 10,
                            sanitize=True):
        spec = GenerativeSpec(smiles=[smiles],
                              padding=padding,
                              radius=scaled_radius,
                              numRequested=num_requested,
                              sanitize=sanitize)

        result = self.stub.SmilesToEmbedding(spec)
        return result

    def embedding_to_smiles(self,
                            embedding,
                            dim: int,
                            pad_mask):
        spec = EmbeddingList(embedding=embedding,
                             dim=dim,
                             pad_mask=pad_mask)

        return self.stub.EmbeddingToSmiles(spec)

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             force_unique=False,
                             sanitize=True,
                             compound_id=None):
        if isinstance(compound_id, list):
            # Sometimes calling routine may send a list of length one containing the compound ID
            if len(compound_id) > 1:
                logger.info(f'find_similars_smiles received {compound_id}, generating neighbors only for first compound!')
            compound_id = compound_id[0]
        spec = GenerativeSpec(model=GenerativeModel.MegaMolBART,
                              smiles=smiles,
                              radius=scaled_radius,
                              numRequested=num_requested,
                              forceUnique=force_unique,
                              sanitize=sanitize)
        result = self.stub.FindSimilars(spec)
        generatedSmiles = result.generatedSmiles
        embeddings = []
        dims = []
        for embedding in result.embeddings:
            embeddings.append(list(embedding.embedding))
            dims.append(embedding.dim)
        if not compound_id:
            compound_id = 'source'
        generated_df = pd.DataFrame({
            'SMILES': generatedSmiles,
            'embeddings': embeddings,
            'embeddings_dim': dims,
            'Generated': [False] + [True] * (len(generatedSmiles) - 1),
            'id': [
                str(compound_id)] + [f'{compound_id}-g{i + 1}' 
                for i in range(len(generatedSmiles) - 1)
            ],
        })
        return generated_df

    def interpolate_smiles(
        self,
        smiles: List,
        num_points: int = 10,
        scaled_radius=None,
        force_unique=False,
        compound_ids=[],
        sanitize=True
    ):
        if len(compound_ids) == 0:
            compound_ids = [f'source{i}' for i in range(len(smiles))]

        spec = GenerativeSpec(model=GenerativeModel.MegaMolBART,
                              smiles=smiles,
                              radius=scaled_radius,
                              numRequested=num_points,
                              forceUnique=force_unique,
                              sanitize=sanitize)

        result = self.stub.Interpolate(spec)
        result = result.generatedSmiles
        n_pairs = len(compound_ids) - 1
        n_generated = num_points + 2
        n_generated_total = n_generated * n_pairs
        assert len(result) == n_generated_total, f"Expected generator to return {n_generated} compounds between each of the {n_pairs} compound-pairs but got {len(result)}"
        generated_df = pd.DataFrame({
            'SMILES': result,
            'Generated': [
                i % n_generated not in [0, n_generated - 1]
                for i in range(n_generated_total)
            ],
            'id': [
                str(compound_ids[i // n_generated]) if i % n_generated == 0 
                else str(compound_ids[1 + i // n_generated]) if i % n_generated == n_generated - 1  
                else f'{compound_ids[i // n_generated]}-{compound_ids[1 + i // n_generated]}_i{i % n_generated}' 
                for i in range(n_generated_total)
            ],
        })
        #generated_df = pd.DataFrame({'SMILES': result,
        #                             'Generated': [True for i in range(len(result))]})
        #generated_df.iat[0, 1] = False
        #generated_df.iat[-1, 1] = False
        return generated_df

   
    def extrapolate_from_cluster(self,
                                 compounds_df,
                                 compound_property: str,
                                 cluster_id: int = 0,
                                 n_compounds_to_transform=10,
                                 num_points: int = 10,
                                 step_size: float = 0.01,
                                 force_unique = False,
                                 scaled_radius: int = 1):
        """
        The embedding vector is calculated for the specified cluster_id and applied over it.
        TO DO: We should have a table of direction vectors in embedded space listed, just like the list of compound IDs.
        The user should choose one to be applied to the selected compounds, or to a cluster number.
        """
        smiles_list = None
        radius = self._compute_radius(scaled_radius)
        # TO DO: User must be able to extrapolate directly from smiles in the table;
        # these may themselves be generated compounds without any chemblid.
        df_cluster = compounds_df[ compounds_df['cluster'] == int(cluster_id) ].dropna().reset_index(drop=True).compute()
        if 'transformed_smiles' in df_cluster:
            smiles_col = 'transformed_smiles'
        elif 'SMILES' in df_cluster:
            smiles_col = 'SMILES'
        elif 'smiles' in df_cluster:
            smiles_col = 'smiles'
        else:
            logger.info(list(df_cluster.columns))
            logger.info(df_cluster.head())
            raise RuntimeError('No smiles column')
            smiles_col = None
        smiles_list = df_cluster[smiles_col].to_array()
        return self.extrapolate_from_smiles(smiles_list,
                                            compound_property_vals=df_cluster[compound_property].to_gpu_array(), #to_array(), #[:n_compounds_to_transform].to_array(),
                                            num_points=num_points,
                                            n_compounds_to_transform=n_compounds_to_transform,
                                            step_size=step_size,
                                            scaled_radius=radius,
                                            force_unique=force_unique,
                                            id_list=df_cluster['id'].to_array())

    def _get_embedding_direction(self,
                                 embedding_list,
                                 compound_property_vals,
                                 ):
        """
        Get the embedding of all compounds in the specified cluster.
        The performa a linear regression against the compound_property to find the direction in
        embedded space along which the compound_property tends to increase.
        Using the minimum and maximum values of the compound_property in the cluster to define the range,
        compute the step size along the direction that is expected to increase the compound_property value by step_percentage.
        """

        logger.info(f'_get_embedding_direction: emb:{embedding_list.shape}, {type(embedding_list)}, prop:{compound_property_vals.shape}, {type(compound_property_vals)}, prop: {min(compound_property_vals)} - {max(compound_property_vals)}')
        n_data = compound_property_vals.shape[0]
        n_dimensions = embedding_list[0].shape[0]
        reg = reg.fit(embedding_list, compound_property_vals)
        n_zero_coefs = len([x for x in reg.coef_ if x == 0.0])
        zero_coef_indices = [i for i, x in enumerate(reg.coef_) if x != 0.0]
        logger.info(f'coef: {n_zero_coefs} / {len(reg.coef_)} coefficients are zero (in some positions between {min(zero_coef_indices)} and {max(zero_coef_indices)});'\
                    f' range: {reg.coef_.argmin()}: {min(reg.coef_)} to {reg.coef_.argmax()}: {max(reg.coef_)}')
        
        y_pred = reg.predict(embedding_list)
        rmse = sqrt(mean_squared_error(compound_property_vals, y_pred.astype('float64')))
        pearson_rho = cp.corrcoef(compound_property_vals, y_pred)
        logger.info(f'_get_embedding_direction: n={len(compound_property_vals)}, rho={pearson_rho}, rmse={rmse}') #:.2f}')
        emb_std = np.std(embedding_list, axis=0)
        logger.info(f'embedding_list.std: {emb_std}')
     
        emb_max = embedding_list[ np.argmax(compound_property_vals) ]
        emb_min = embedding_list[ np.argmin(compound_property_vals) ]
        diff_size = np.linalg.norm(emb_max - emb_min) / sqrt(n_dimensions)
        # TODO: project on to embedding direction!!!
        logger.info(f'compound_property_vals: [{np.argmin(compound_property_vals)}]={np.amin(compound_property_vals)}, [{np.argmax(compound_property_vals)}]={np.amax(compound_property_vals)}, diff_size={diff_size}')
        return reg.coef_, emb_std, diff_size


    def extrapolate_from_smiles(self,
                                smiles_list,
                                compound_property_vals,
                                num_points: int,
                                step_size: float,
                                scaled_radius=None,
                                force_unique=False,
                                n_compounds_to_transform=10,
                                id_list=[],
                                debug=False):
        """
        Given a list of smiles strings, convert each to its embedding.
        Then taken num_points steps in the specified direction (in the embedded space) of size step_size.
        Convert these points on the embedded space back to smiles strings and return as a dataframe.
        Modify duplicates if force_unique is True by adding a jitter of magnitude radius to the embedding.
        """
        # TODO: generated compounds are the same no matter what the step-size is, check code!!!!
        # TODO: generated compounds are yielding different Tanimotos even though their are identical. Bug or jitter???
        step_size = float(step_size)
        n_compounds_to_transform = int(n_compounds_to_transform)
        if len(id_list) == 0:
            id_list = list(map(str, range(len(smiles_list))))
        logger.info(f'molbart: extrapolate_from_smiles: {len(smiles_list)} smiles ({type(smiles_list)}), {num_points} extrapolations each with step_size {step_size}')
        data = pd.DataFrame({'transformed_smiles': smiles_list})
        logger.info(data.head())
        #pad_length = max(map(len, smiles_list)) + 2 # add 2 for start / stop
        # TODO: check reversibility / recovery
        full_mask = None
        emb_shape = None
        n_recovered = 0
        avg_tani = 0
        embeddings = []
        for i, smiles in enumerate(smiles_list):
            spec = GenerativeSpec(
                model=GenerativeModel.MegaMolBART,
                smiles=smiles,
            )
            result = self.stub.SmilesToEmbedding(spec)
            emb = result.embedding
            mask = result.pad_mask
            emb_shape = result.dim
            if debug:
                spec = EmbeddingList(
                    embedding=emb,
                    dim=emb_shape,
                    pad_mask=mask
                )
                generated_mols = self.stub.EmbeddingToSmiles(spec).generatedSmiles
                if len(generated_mols) > 0:
                    n_recovered += 1
                    tani = FingerprintSimilarity(RDKFingerprint(MolFromSmiles(smiles)), RDKFingerprint(MolFromSmiles(generated_mols[0])))
                    logger.info(f'{n_recovered}/ {i+1}: {smiles} ({len(smiles)} chars)--> emb:{emb_shape}, mask:{mask.shape} --> {generated_mols} (tani={tani:.2f})')
                    avg_tani += tani
            logger.info(f'emb: {type(emb)}, dim={emb_shape}, mask={len(mask)}, emb={len(emb)}')
            embeddings.append(torch.tensor(emb)) #.detach().reshape(-1)) #torch tensor
            if full_mask is None:
                logger.info(f'First mask = {mask}')
                full_mask = mask
                emb_shape = emb_shape # n_tokens x 1 x 256
            else:
                full_mask = [a and b for a, b in zip(full_mask, mask)] # not used any more
        if debug:
            logger.info(f'{n_recovered} / {len(smiles_list)} compounds yielded something after embedding, with avg tani = {avg_tani / n_recovered if n_recovered > 0 else 0}')

        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=PAD_TOKEN) # n_smiles x embedding_length
        n_embedding_tokens = int(embeddings.shape[1] / (emb_shape[1] * emb_shape[2]))
        emb_shape = [n_embedding_tokens, emb_shape[1], emb_shape[2]]
        embeddings = cp.asarray(embeddings)
        full_mask = [False] * n_embedding_tokens
        logger.info(f'emb type: {type(embeddings)} of {type(embeddings[0])}')
        logger.info(f'embeddings.shape:{embeddings.shape}, emb_shape={emb_shape}, embeddings[0]={embeddings[0]}')

        # Use the entire cluster to infer the direction:
        direction, emb_std, diff_size = self._get_embedding_direction(embeddings, compound_property_vals)
        if diff_size == 0.0:
            logger.info(f'Increasing diff_size from 0.0 to 1e-6')
            diff_size = 1e-6

        # But apply the transform to no more than n_compounds_to_transform, chosen at random
        if n_compounds_to_transform < len(smiles_list):
            indices = np.random.choice(list(range(len(smiles_list))), size=n_compounds_to_transform, replace=False)
            smiles_list = [smiles_list[i] for i in indices]
            embeddings = cp.asarray([embeddings[i,:] for i in indices])
            id_list = [id_list[i] for i in indices]

        result_df_list = [ pd.DataFrame({'SMILES': smiles_list,  'Generated': False, 'id': id_list}) ]
        logger.info(f'direction: {type(direction)}, shape={direction.shape}, {direction}\n, embeddings: {type(embeddings)}, shape: {embeddings.shape}, embeddings[0]={embeddings[0]}')

        for step_num in range(1, 1 + num_points):
            direction_sampled = cp.random.normal(loc=direction, scale=emb_std, size=emb_std.shape) #direction + noise
            logger.info(f'step ({type(step_num)} * {type(diff_size)} * {type(step_size)} * {type(direction_sampled)}')
            step = float(step_num * diff_size * step_size) * direction_sampled
            logger.info(step)
            extrap_embeddings = embeddings + step # TODO: print and check output
            logger.info(f'step ({step_num} * {diff_size} * {step_size} * direction_sampled): {type(step)}, {step.shape}, {step}\n:extrap_embeddings: {type(extrap_embeddings)}, {extrap_embeddings.shape}, extrap_embeddings[0]={extrap_embeddings[0]}')
            smiles_gen_list = []
            ids_interp_list = []
            for i in range(len(extrap_embeddings)):
                extrap_embedding = list(extrap_embeddings[i,:])
                logger.info(f'embedding: {type(extrap_embedding)}, {len(extrap_embeddings)};'\
                            f' dim: {type(emb_shape)}, {len(emb_shape)}; pad_mask={type(full_mask)}, {len(full_mask)}')
                spec = EmbeddingList(
                    embedding=extrap_embedding,
                    dim=emb_shape,
                    pad_mask=full_mask
                )
                smiles_gen = self.stub.EmbeddingToSmiles(spec).generatedSmiles[0]
                logger.info(f'{i}: {smiles_gen}')
                smiles_gen_list.append(smiles_gen)
                ids_interp_list.append(f'{id_list[i]}-s{step_num}')
            extrap_df = pd.DataFrame({
                'SMILES': smiles_gen_list,
                'Generated': True,
                'id': ids_interp_list
            })
            logger.info(extrap_df.head())
            if force_unique:
                inv_transform_funct = partial(self.inverse_transform,
                                            mem_pad_mask=full_mask)
                extrap_df = self.compute_unique_smiles(extrap_df,
                                                   smiles_gen,
                                                   inv_transform_funct,
                                                   radius=radius)
            logger.info(f'step_num={step_num} yielded {len(extrap_df)} compounds:\n{extrap_df.head()}')
            result_df_list.append(extrap_df)
        results_df = pd.concat(result_df_list, ignore_index=True)
        results_df['id'] = results_df['id'].apply(str)
        results_df.sort_values('id', inplace=True)
        results_df.reset_index(drop=True, inplace=True)
        return results_df

    def fit_nn(
        self,
        compounds_df,
        compound_property,
        cluster_id_train,
        cluster_id_test,
        hidden_layer_sizes,
        activation_fn,
        final_activation_fn,
        max_epochs,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.0001,
        debug=False, #82 / 88 compounds yielded something after embedding, with avg tani = 0.8287583649661866
        #scaled_radius=None
        ):
        """
        Convert compound SMILES to embeddings, then train a neural network with n_layers hidden layers with the specified activation function (activation_fn)
        to predict the specified compound_property of cluster_id_train. Evaluate the model on cluster_id_test. Return actual and predicted values for both
        the train and test set.
        """
        logger.info(f'cluster_id_train={cluster_id_train}, cluster_id_test={cluster_id_test}, compound_property={compound_property}, compounds_df: {len(compounds_df)}, {type(compounds_df)}')
        df_train = compounds_df[ compounds_df['cluster'] == int(cluster_id_train) ].dropna().reset_index(drop=True)#.compute()
        if hasattr(df_train, 'compute'): 
            df_train = df_train.compute()
        df_test = compounds_df[ compounds_df['cluster'] == int(cluster_id_test) ].dropna().reset_index(drop=True)#.compute()
        if hasattr(df_test, 'compute'):
            df_test = df_test.compute()
        n_train = len(df_train)
        n_test = len(df_test)

        logger.info(f'df_train: {len(df_train)}\n{df_train.head()}')
        logger.info(f"type(df_train['transformed_smiles'])={type(df_train['transformed_smiles'])}")

        smiles_list = np.concatenate((df_train['transformed_smiles'].to_array(), df_test['transformed_smiles'].to_array()), axis=0)
        logger.info(f'smiles_list: {smiles_list.shape}')
        pad_length = max(map(len, smiles_list)) + 2 # add 2 for start / stop
        embeddings = []
        #full_mask = None
        emb_shape = None
        n_recovered = 0
        avg_tani = 0
        #radius = self._compute_radius(scaled_radius)

        for i, smiles in enumerate(smiles_list):
            spec = GenerativeSpec(
                model=GenerativeModel.MegaMolBART,
                smiles=smiles,
                #radius=radius
            )
            result = self.stub.SmilesToEmbedding(spec)
            emb = result.embedding
            mask = result.pad_mask
            dim = result.dim
            #logger.info(f'{i}: smiles={smiles}, emd: {len(emb)}, {emb[:5]}; dim={dim}, mask: {len(mask)}')
            emb_shape = result.dim
            if debug:
                spec = EmbeddingList(
                    embedding=emb,
                    dim=emb_shape,
                    pad_mask=mask
                )
                generated_mols = self.stub.EmbeddingToSmiles(spec).generatedSmiles                
                if len(generated_mols) > 0:
                    m = MolFromSmiles(generated_mols[0])
                    if m is not None:
                        n_recovered += 1
                        tani = FingerprintSimilarity(RDKFingerprint(MolFromSmiles(smiles)), RDKFingerprint(m))
                        logger.info(f'{n_recovered}/ {i+1}: {smiles} ({len(smiles)} chars)--> emb:{emb_shape}, mask:{len(mask)} --> {generated_mols} (tani={tani:.2f})')
                        avg_tani += tani
            embeddings.append(torch.tensor(emb, device=self.device)) #emb.detach().reshape(-1)) #torch tensor

        if debug:
            logger.info(f'{n_recovered} / {len(smiles_list)} compounds yielded something after embedding, with avg tani = {avg_tani / n_recovered if n_recovered > 0 else 0}')
        
        embeddings = torch.nn.utils.rnn.pad_sequence(
            embeddings, batch_first=True, padding_value=PAD_TOKEN)
        embeddings_train = embeddings[:n_train,:]
        embeddings_test = embeddings[n_train:,:]
        logger.info(f'emb train: {type(embeddings_train)} of {type(embeddings_train[0])}, {embeddings_train.shape}')
        compound_property_vals_train = torch.tensor(df_train[compound_property], device=self.device, dtype=torch.float32)#.to_gpu_array() # need to move to GPU array??
        compound_property_vals_test = torch.tensor(df_test[compound_property], device=self.device, dtype=torch.float32)#.to_gpu_array() # need to move to GPU array??
        logger.info(f'type(df_train[{compound_property}])={type(df_train[compound_property])}, type(compound_property_vals_train)={type(compound_property_vals_train)}')
        train_pred, test_pred = self._build_and_train_nn(
            embeddings_train,
            compound_property_vals_train,
            embeddings_test,
            compound_property_vals_test,
            hidden_layer_sizes = hidden_layer_sizes,
            activation_fn=activation_fn,
            final_activation_fn=final_activation_fn,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size
        )
        df = pd.DataFrame({
            'x': torch.cat((compound_property_vals_train, compound_property_vals_test), axis=0).to('cpu').numpy(),
            'y': torch.cat((train_pred.detach(), test_pred.detach()), axis=0).to('cpu').flatten().numpy(),
            'cluster': np.concatenate((df_train['cluster'].to_array(), df_test['cluster'].to_array()), axis=0),
            'id': np.concatenate((df_train['id'].to_array(), df_test['id'].to_array()), axis=0),
            'train_set': [True] * n_train + [False] * n_test
        })
        return df

    def _build_and_train_nn(self,
                            embedding_list_train,
                            compound_property_vals_train,
                            embedding_list_test,
                            compound_property_vals_test,
                            hidden_layer_sizes = [],
                            activation_fn='LeakyReLU',
                            final_activation_fn='LeakyReLU',
                            max_epochs=10,
                            batch_size=32,
                            learning_rate=0.001,
                            weight_decay=0.0001
                            ):
        """
        Construct a neural network with the specified number of layers, using the specified activation function.
        Then train it on the training set and evaluate on the test set. Return results.
        """

        logger.info(f'_build_and_train_nn: emb_train:{embedding_list_train.shape}, {type(embedding_list_train)}, embedding_list_train[0]:{len(embedding_list_train[0])},'\
                f' prop:{compound_property_vals_train.shape}, {type(compound_property_vals_train)},'\
                f' prop_train: {min(compound_property_vals_train)} - {max(compound_property_vals_train)}')
        n_data_train = compound_property_vals_train.shape[0]
        n_dimensions = embedding_list_train[0].shape[0]
        comp_net = CompoundNet(n_dimensions, hidden_layer_sizes, activation_fn, final_activation_fn).to(self.device)
        logger.info(comp_net)
        loss_fn = torch.nn.SmoothL1Loss()
        opt = torch.optim.Adam(comp_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_set = CompoundData(embedding_list_train, compound_property_vals_train)
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        i = 0
        for epoch in range(max_epochs):
            total_loss = 0.0
            for compounds, properties in loader:
                opt.zero_grad()
                predictions = comp_net(compounds)
                loss = loss_fn(predictions, properties)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                i += 1
            logger.info(f'epoch {epoch+1}, {i} batches: {total_loss / (n_data_train * (epoch+1))}')

        comp_net.eval()
        train_pred = comp_net(embedding_list_train)
        test_pred = comp_net(embedding_list_test)

        return train_pred, test_pred


class CompoundData(Dataset):

    def __init__(self, compounds, properties):
        self.compounds = compounds
        self.properties = properties

    def __len__(self):
        return len(self.compounds)

    def __getitem__(self, compound_index):
        return self.compounds[compound_index,:], self.properties[compound_index]


class CompoundNet(torch.nn.Module):

    def __init__(self, n_input_features, hidden_layer_sizes, activation_fn, last_activation_fn=None):
        super(CompoundNet, self).__init__()
        hidden_layer_sizes.append(1) # output layer size is appended to hidden layer sizes
        layers = [torch.nn.Linear(n_input_features, hidden_layer_sizes[0])]
        try:
            activation = getattr(torch.nn, activation_fn)
            if last_activation_fn:
                last_activation = getattr(torch.nn, last_activation_fn)
        except Exception as e:
            raise UserError(f'Activation function name {activation_fn} / {last_activation_fn} not recognized')
        for i, hidden_layer_size in enumerate(hidden_layer_sizes[:-1]):
            layers.append(activation())
            layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_sizes[i + 1]))
        if last_activation_fn:
            # Having a non-linear function right before the output may not be needed for some properties being predicted
            layers.append(last_activation())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
