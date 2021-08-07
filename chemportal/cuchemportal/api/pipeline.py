import logging
import sqlalchemy
import flask
import json
from datetime import datetime
from http import HTTPStatus
from flask import request


from cuchemportal import app, Response
from cuchemportal.pipeline import PipelineManager
from cuchemportal.pipeline.pipeline import Pipeline, pipelines_schema, pipeline_schema


logger = logging.getLogger(__name__)


@app.route("/api/pipeline", methods=["POST"])
def save():
    pipeline = Pipeline.from_dict(request.get_json(force=True))

    pipeline.created_by = 'app'
    pipeline.updated_by = 'app'

    ppln_mgr = PipelineManager()

    response = Response()
    http_status = HTTPStatus.BAD_REQUEST

    try:
        ppln_id = ppln_mgr.create(pipeline)
        response.is_successful = True
        response.data = ppln_id
        http_status = HTTPStatus.CREATED
    except sqlalchemy.exc.IntegrityError as e:
        response.is_successful = False
        response.error_code = 1
        response.error_msg = 'Please verify data. Pipeline name must be unique.'
        logger.exception(e)
    except Exception as e:
        response.is_successful = False
        response.error_code = 1
        response.error_msg = 'Unexpected error'
        logger.exception(e)

    return flask.Response(response=json.dumps(response.__dict__), status=http_status)


@app.route("/api/pipeline", methods=["PATCH"])
def update():
    pipeline = Pipeline.from_dict(request.get_json(force=True))

    pipeline.updated_by = 'app'

    ppln_mgr = PipelineManager()

    response = Response()
    http_status = HTTPStatus.BAD_REQUEST

    try:
        ppln = ppln_mgr.update(pipeline)
        response.is_successful = True
        ppln.created_at = datetime.strptime(ppln.created_at, "%Y-%m-%dT%H:%M:%S")

        response.data = pipeline_schema.dump(ppln)
        http_status = HTTPStatus.OK
    except sqlalchemy.exc.IntegrityError as e:
        response.is_successful = False
        response.error_code = 1
        response.error_msg = 'Please verify data. Pipeline name must be unique.'
        logger.exception(e)
    except Exception as e:
        response.is_successful = False
        response.error_code = 1
        response.error_msg = 'Unexpected error'
        logger.exception(e)

    return flask.Response(response=json.dumps(response.__dict__), status=http_status)


@app.route("/api/pipelines/<int:start>/<int:page_size>", methods=["GET"])
def fetch_all(start:int, page_size:int):

    ppln_mgr = PipelineManager()

    response = Response()
    http_status = HTTPStatus.BAD_REQUEST

    try:
        pplns = ppln_mgr.fetch_all(start_index=start, num_rows=page_size)
        response.is_successful = True
        response.data = pipelines_schema.dump(pplns)
        http_status = HTTPStatus.OK
    except Exception as e:
        response.is_successful = False
        response.error_code = 1
        response.error_msg = 'Unexpected error'
        logger.exception(e)

    return flask.Response(response=json.dumps(response.__dict__), status=http_status)


@app.route("/api/pipeline/<int:id>", methods=["GET"])
def fetch(id:int):

    ppln_mgr = PipelineManager()

    response = Response()
    http_status = HTTPStatus.BAD_REQUEST

    try:
        ppln = ppln_mgr.fetch_by_id(id)
        response.is_successful = True
        response.data = pipeline_schema.dump(ppln)
        http_status = HTTPStatus.OK
    except Exception as e:
        response.is_successful = False
        response.error_code = 1
        response.error_msg = 'Unexpected error'
        logger.exception(e)

    return flask.Response(response=json.dumps(response.__dict__), status=http_status)
