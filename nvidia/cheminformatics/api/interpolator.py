import logging

from flask import jsonify
from flask import request
from flask_restplus import Resource

from nvidia.cheminformatics.api import api_rest
from nvidia.cheminformatics.wf.generative import Cddd, MolBART


logger = logging.getLogger(__name__)


@api_rest.route('/interpolator/<string:model>/<string:smiles>/<int:num_requested>')
class Interpolator(Resource):
    """
    Exposes all Request related operations thru a REST endpoint.
    """

    def get(self, model, smiles, num_requested=10):

        if model == 'CDDD':
            generated_smiles, neighboring_embeddings, pad_mask = \
                Cddd().find_similars_smiles_list(
                    smiles,
                    num_requested=num_requested,
                    force_unique=True)
        else:
            generated_smiles, neighboring_embeddings, pad_mask = \
                MolBART().find_similars_smiles_list(
                    smiles,
                    num_requested=num_requested,
                    force_unique=True)

        return jsonify(generated_smiles)
