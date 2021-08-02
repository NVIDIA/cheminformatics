from cuchemportal.pipeline.pipeline import Pipeline
from cuchemportal.pipeline.pipeline_manager import PipelineManager
import json

# Building a Pipeline and setting config 
def test_pipeline_insert():
        mgr = PipelineManager()
        with open("cuchemportal/data/configurations/portal_config.json", 'r') as jfile:
            config = json.load(jfile)

        ppln = Pipeline() # Todo: autoread into Task and Job Dataclasses
        ppln.name = "cuchem_pipeline2"
        ppln.definition = {"a": "b"}
        ppln.config = config
        record = mgr.create(ppln)
        assert record is not None 

