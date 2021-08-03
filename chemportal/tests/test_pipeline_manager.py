from cuchemportal.pipeline.pipeline import Pipeline
from cuchemportal.pipeline.pipeline_manager import PipelineManager
import json

# Building a Pipeline and setting config 
def test_pipeline_insert():
        mgr = PipelineManager()
        with open("cuchemportal/test/data/portal_config.json", 'r') as jfile:
            config = json.load(jfile)

        ppln = Pipeline() # Todo: autoread into Task and Job Dataclasses
        ppln.name = "cuchem_pipeline2"
        ppln.definition = {"a": "b"}
        ppln.config = config
        record = mgr.create(ppln)
        assert record is not None 

def test_pipeline_update():
    
    mgr = PipelineManager()
    with open("cuchemportal/tests/data/portal_second_config.json", 'r') as jfile:
        config = json.load(jfile)

    config["name"] = "cuchem_pipelines_3"
    config["id"] = 26
    config["definition"] = {"c":  {"a":"b"}}
    ppln = Pipeline() # Todo: autoread into Task and Job Dataclasses
    ppln.config = config
    record = mgr.update(26, ppln.config)
    assert record is not None 


def test_pipeline_fetch():
    mgr = PipelineManager()
    record = mgr.fetch_by_id(12)
    print(record)
    assert record is not None 

def test_pipeline_fetch_all():
    mgr = PipelineManager()
    record = mgr.fetch_all(0, 26)
    print(record)
    assert record is not None 

def test_pipeline_delete():
    mgr = PipelineManager()
    deleted = mgr.delete(26)
    assert deleted is not None