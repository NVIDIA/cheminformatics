import logging
from typing import List

from cuchemcommon.data import GenerativeWfDao
from cuchemcommon.fingerprint import BaseTransformation

logger = logging.getLogger(__name__)


class BaseGenerativeWorkflow(BaseTransformation):

    def __init__(self, dao: GenerativeWfDao = None) -> None:
        self.dao = dao
        self.min_jitter_radius = None

    def is_ready(self, timeout: int = 10):
        return True

    def add_jitter(embedding, radius, cnt):
        NotImplemented

    def smiles_to_embedding(self,
                            smiles: str,
                            padding: int):
        NotImplemented

    def embedding_to_smiles(self,
                            embedding: float,
                            dim: int,
                            pad_mask):
        NotImplemented

    def interpolate_smiles(self,
                           smiles: List,
                           num_points: int = 10,
                           scaled_radius=None,
                           force_unique=False,
                           sanitize=True):
        NotImplemented

    def find_similars_smiles_list(self,
                                  smiles: str,
                                  num_requested: int = 10,
                                  scaled_radius=None,
                                  force_unique=False,
                                  sanitize=True):
        NotImplemented

    def find_similars_smiles(self,
                             smiles: str,
                             num_requested: int = 10,
                             scaled_radius=None,
                             force_unique=False,
                             sanitize=True):
        NotImplemented

    def _compute_radius(self, scaled_radius):
        if scaled_radius:
            return float(scaled_radius * self.min_jitter_radius)
        else:
            return self.min_jitter_radius

    def interpolate_by_id(self,
                          ids: List,
                          id_type: str = 'chembleid',
                          num_points=10,
                          force_unique=False,
                          scaled_radius: int = 1,
                          sanitize=True):
        smiles = None

        if not self.min_jitter_radius:
            raise Exception('Property `radius_scale` must be defined in model class.')

        if id_type.lower() == 'chembleid':
            smiles = [row[2] for row in self.dao.fetch_id_from_chembl(ids)]
            if len(smiles) != len(ids):
                raise Exception('One of the ids is invalid %s', ids)
        else:
            raise Exception('id type %s not supported' % id_type)

        return self.interpolate_smiles(smiles,
                                       num_points=num_points,
                                       scaled_radius=scaled_radius,
                                       force_unique=force_unique,
                                       sanitize=sanitize)

    def find_similars_smiles_by_id(self,
                                   chemble_id: str,
                                   id_type: str = 'chembleid',
                                   num_requested=10,
                                   force_unique=False,
                                   scaled_radius: int = 1,
                                   sanitize=True):
        smiles = None

        if not self.min_jitter_radius:
            raise Exception('Property `radius_scale` must be defined in model class.')

        if id_type.lower() == 'chembleid':
            smiles = [row[2] for row in self.dao.fetch_id_from_chembl(chemble_id)]
            if len(smiles) != len(chemble_id):
                raise Exception('One of the ids is invalid %s' + chemble_id)
        else:
            raise Exception('id type %s not supported' % id_type)

        return self.find_similars_smiles(smiles[0],
                                         num_requested=num_requested,
                                         scaled_radius=scaled_radius,
                                         force_unique=force_unique,
                                         sanitize=sanitize)
