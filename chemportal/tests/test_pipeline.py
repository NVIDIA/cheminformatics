import json
import pytest
import logging
import uuid

from cuchemportal.pipeline.pipeline import Pipeline
from cuchemportal.context import Context
from cuchemportal.pipeline import PipelineManager

logger = logging.getLogger(__name__)


def create_pipeline(name=None):
    with open("tests/data/portal_config.json", 'r') as jfile:
        config = json.load(jfile)

    ppln = Pipeline()
    ppln.config = config
    ppln.name = name
    ppln.created_by = 'TESTER'
    ppln.updated_by = 'TESTER'
    ppln.definition = {"a": "b"}

    ppln_mgr = PipelineManager()
    ppln_id = ppln_mgr.create(ppln)
    assert ppln_id is not None

    return ppln_id

@pytest.mark.usefixtures("mysql_server")
def test_create_pipeline(mysql_server):

    context = Context()
    context.db_host = mysql_server.interface("eth0").addresses[0]

    ppln_name = str(uuid.uuid1())

    logger.info(f'Creating pipeline {ppln_name}...')
    ppln_id = create_pipeline(name=ppln_name)

    ppln_mgr = PipelineManager()
    ppln = ppln_mgr.fetch_by_id(ppln_id)

    assert ppln.name == ppln_name
