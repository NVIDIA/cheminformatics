from flask import jsonify
from . import app


@app.route("/api/workflow")
def hello():
    return jsonify({'name': 'Test'})
