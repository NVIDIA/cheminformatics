import logging
import torch

from generativesampler_pb2 import EmbeddingList, SmilesList, Version
import generativesampler_pb2_grpc
from megamolbart.inference import MegaMolBART

from cuchemcommon.utils import Singleton

logger = logging.getLogger(__name__)


class GenerativeSampler(generativesampler_pb2_grpc.GenerativeSampler, metaclass=Singleton):

    def __init__(self, *args, **kwargs):
        torch.set_grad_enabled(False)

        model_dir = kwargs['model_dir'] if 'model_dir' in kwargs else None
        self.megamolbart = MegaMolBART(model_dir=model_dir + '/megamolbart_checkpoint.nemo')

        logger.info(f'Loaded Version {self.megamolbart.version}')

    # TODO update to accept batched input if similes2embedding does
    # TODO how to handle length overrun for batch processing --> see also MegaMolBART.load_model in inference.py
    def SmilesToEmbedding(self, spec, context):

        smiles_str = ''.join(spec.smiles)

        embedding, pad_mask = self.megamolbart.smiles2embedding(smiles_str,
                                                                pad_length=spec.padding)
        dim = embedding.shape
        embedding = embedding.flatten().tolist()
        return EmbeddingList(embedding=embedding,
                             dim=dim,
                             pad_mask=pad_mask)

    def EmbeddingToSmiles(self, embedding_spec, context):
        '''
        Converts input embedding to SMILES.
        @param transform_spec: Input spec with embedding and mask.
        '''
        embedding = torch.FloatTensor(list(embedding_spec.embedding))
        pad_mask = torch.BoolTensor(list(embedding_spec.pad_mask))
        dim = tuple(embedding_spec.dim)

        embedding = torch.reshape(embedding, dim).cuda()
        pad_mask = torch.reshape(pad_mask, (dim[0], 1)).cuda()

        generated_mols = self.megamolbart.inverse_transform([embedding], pad_mask)
        return SmilesList(generatedSmiles=generated_mols)

    def FindSimilars(self, spec, context):

        smiles_str = ''.join(spec.smiles)

        generated_df = self.megamolbart.find_similars_smiles(
                smiles_str,
                num_requested=spec.numRequested,
                scaled_radius=spec.radius,
                sanitize=spec.sanitize,
                pad_length=spec.padding,
                force_unique=False)

        embeddings = []

        for _, row in generated_df.iterrows():
            embeddings.append(EmbeddingList(embedding=row.embeddings,
                                           dim=row.embeddings_dim))

        return SmilesList(generatedSmiles=generated_df['SMILES'],
                          embeddings=embeddings)

    def Interpolate(self, spec, context):

        _, generated_smiles = self.megamolbart.interpolate_smiles(
            spec.smiles,
            num_points=spec.numRequested,
            scaled_radius=spec.radius,
            sanitize=spec.sanitize,
            force_unique=False)
        return SmilesList(generatedSmiles=generated_smiles)

    def GetVersion(self, spec, context):
        return Version(version='0.1.0_' + self.megamolbart.version)
