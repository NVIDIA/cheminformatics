import pytest
import json

from cuchemportal.context import Context
from cuchemportal.pipeline.pipeline import Pipeline
from cuchemportal.pipeline.pipeline_manager import PipelineManager

# Building a Pipeline and setting config
@pytest.mark.usefixtures("mysql_server")
def test_pipeline_insert(mysql_server):
    context = Context()
    context.db_host = mysql_server.interface("eth0").addresses[0]

    mgr = PipelineManager()
    with open("tests/data/portal_config.json", 'r') as jfile:
        config = json.load(jfile)

    ppln = Pipeline() # Todo: autoread into Task and Job Dataclasses
    ppln.config = config
    ppln.name = "cuchem_pipeline7"
    ppln.definition = {"a": "b"}
    ppln.id = 27
    record = mgr.create(ppln)
    assert record is not None


@pytest.mark.usefixtures("mysql_server")
def test_pipeline_update(mysql_server):
    context = Context()
    context.db_host = mysql_server.interface("eth0").addresses[0]

    mgr = PipelineManager()
    with open("tests/data/portal_second_config.json", 'r') as jfile:
        config = json.load(jfile)

    config["name"] = "cuchem_pipelines_8"
    config["id"] = 27
    config["definition"] = {"c":  {"a":"b"}}
    ppln = Pipeline() # Todo: autoread into Task and Job Dataclasses
    ppln.config = config
    record = mgr.update(27, ppln.config)
    assert record is not None


@pytest.mark.usefixtures("mysql_server")
def test_pipeline_fetch(mysql_server):
    context = Context()
    context.db_host = mysql_server.interface("eth0").addresses[0]

    mgr = PipelineManager()
    record = mgr.fetch_by_id(12)
    print(record)
    assert record is not None


@pytest.mark.usefixtures("mysql_server")
def test_pipeline_fetch_all(mysql_server):
    context = Context()
    context.db_host = mysql_server.interface("eth0").addresses[0]

    mgr = PipelineManager()
    record = mgr.fetch_all(0, 27)
    print(record)
    assert record is not None


@pytest.mark.usefixtures("mysql_server")
def test_pipeline_delete(mysql_server):
    context = Context()
    context.db_host = mysql_server.interface("eth0").addresses[0]

    mgr = PipelineManager()
    deleted = mgr.delete(27)
    assert deleted is not None