import os
import logging

import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask import Blueprint, Flask, send_file, json
from flask.globals import current_app
from flask.helpers import safe_join
from flask_restplus import Api
from werkzeug.exceptions import HTTPException
from werkzeug.http import HTTP_STATUS_CODES

logger = logging.getLogger(__name__)


class CustomApi(Api):
    """This class overrides 'handle_error' method of 'Api' class ,
    to extend global exception handing functionality of 'flask-restful'.
    """

    def handle_error(self, err):
        """It helps preventing writing unnecessary
        try/except block though out the application
        """
        logger.exception(err)  # log every exception raised in the application
        # Handle HTTPExceptions
        if isinstance(err, HTTPException):
            return json.jsonify({
                'message': getattr(
                    err, 'description', HTTP_STATUS_CODES.get(err.code, '')
                )
            }), err.code
        # If msg attribute is not set,
        # consider it as Python core exception and
        # hide sensitive error info from end user
        if not getattr(err, 'message', None):
            return json.jsonify({
                'message': str(err)
            }), 500
        # Handle application specific custom exceptions
        return json.jsonify(**err.kwargs), err.http_status_code


api_bp = Blueprint('api_bp', __name__, url_prefix='/api')
api_rest = CustomApi(api_bp)

app = Flask(__name__, static_folder='../../public/')
app.register_blueprint(api_bp)


@app.route('/')
def index_client():
    return send_file('../../public/index.html')


def _send_static_file(dirname, path):
    filename = safe_join(dirname, path)

    if not os.path.isabs(filename):
        filename = os.path.join(current_app.root_path, filename)
    return send_file(filename)


@app.route('/js/<path:path>')
def send_js(path):
    return _send_static_file('../../public/js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return _send_static_file('../../public/css', path)


@app.route('/fonts/<path:path>')
def send_fonts(path):
    return _send_static_file('../../public/fonts', path)


@app.route('/imgs/<path:path>')
def send_imgs(path):
    return _send_static_file('../../public/imgs', path)


@api_bp.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Headers'] = \
        'Content-Type,Authorization'
    return response


from nvidia.cheminformatics.api.interpolator import *