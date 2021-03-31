import logging

from flask import jsonify
from flask import request
from flask_restplus import Resource

from nvidia.cheminformatics.api import api_rest
from nvidia.cheminformatics.wf.generative.cddd import Cddd


logger = logging.getLogger(__name__)


@api_rest.route('/interpolator/<string:model>/<string:request_ids>/<int:num_points>')
class Interpolator(Resource):
    """
    Exposes all Request related operations thru a REST endpoint.
    """

    def get(self, model, request_ids, num_points=10):
        request_ids = request_ids.split(',')
        genreated_df = Cddd().interpolate_from_smiles(
            request_ids,
            num_points=num_points)

        generated_smiles = []
        for idx in range(genreated_df.shape[0]):
            generated_smiles.append(genreated_df.iat[idx, 0])

        return jsonify(generated_smiles)
