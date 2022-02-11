from flask import Blueprint, Flask

api_bp = Blueprint('api_bp', __name__, url_prefix='/api')

app = Flask(__name__, static_folder='../../public/')
app.register_blueprint(api_bp)


class Response(object):
    '''
    Encapsulates the response to be sent to the client.
    '''
    def __init__(self) -> None:
        self.is_successful = True
        self.error_code = 0
        self.error_msg = None
        self.data = None
        self.metadata = {}
