from pipeline.pipeline import Pipeline
from pipeline.pipeline_manager import PipelineManager
import json

# Building a Pipeline and setting config 
def test_pipeline_insert():
        mgr = PipelineManager()
        with open("data/configurations/portal_config.json", 'r') as jfile:
            config = json.load(jfile)

        ppln = Pipeline() # Todo: autoread into Task and Job Dataclasses
        ppln.name = "cuchem_pipeline2"
        ppln.definition = {"a": "b"}
        ppln.config = config
        record = mgr.create(ppln)
        assert record is not None 

