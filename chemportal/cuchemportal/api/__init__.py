# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging

from flask import Blueprint, Flask, send_file, json
from flask.globals import current_app
from sqlalchemy.ext.declarative import DeclarativeMeta
from flask.helpers import safe_join

logger = logging.getLogger(__name__)


class AlchemyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj.__class__, DeclarativeMeta):
            # an SQLAlchemy class
            fields = {}
            for field in [x for x in dir(obj) if
                          not x.startswith('_') and x != 'metadata']:
                data = obj.__getattribute__(field)
                try:
                    json.dumps(data)
                    # will fail on non-encodable values, like other classes
                    fields[field] = data
                except TypeError:
                    fields[field] = None
            # a json-encodable dict
            return fields

        return json.JSONEncoder.default(self, obj)

    def encode(self, obj):
        return json.JSONEncoder.encode(self, obj)


api_bp = Blueprint('api_bp', __name__, url_prefix='/api')

app = Flask(__name__, static_folder='../../public/')
app.register_blueprint(api_bp)
app.json_encoder = AlchemyEncoder


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

from .workflow import *
