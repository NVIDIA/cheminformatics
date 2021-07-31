from job import Job
from os import pipe
from job_manager import JobManager
from pipeline import Pipeline
from data.db_client import DBClient
from copy import deepcopy
import json


class PipelineManager:
    """A workhorse class which runs the Pipeline end to end"""
    def __init__(self):
        # Todo: build a context object
        self.connection_str: str = ("mysql+pymysql://" 
                                     "{0}:{1}@{2}/"
                                     "{3}".format("root", 
                                     "chemportal_database#321", 
                                     "gpasternak-dt.nvidia.com", 
                                     "cuchem_db"))
        self.db_client = DBClient(connection_string=self.connection_str) # TODO: add a valid database connection
        self.manager = JobManager()

    def create(self, ppln: Pipeline) -> bool:
        """Given Configuration, uses pipeline setter method to create pipeline"""

        # Adding Pipeline to dict of Pipelines
        with self.db_client.Session() as sess:
            self.db_client.insert(record=ppln, session=sess)
            sess.commit()

    def update(new_config: str) -> bool:
        """Receives an updated pipeline from the UI and precedes to reset config"""
        pass  # This is an identical API as create_pipeline, but here for CRUD consistency

    # Todo: add is_deleted column to Pipeline and mark pipelines as deleted
    def delete(self, pipeline_id: int):
        """Deletes Pipeline Object"""
        self.db_client.delete_pipeline(id = pipeline_id)

    #def clone_pipeline(self, pipeline_id: int):
        #"""Clones a Pipeline Object"""
        # Since overriden, will call the Pipeline class deepcopy implementation which we can modify
        #return deepcopy(self.pipelines[pipeline_id])

    # Todo: now access to database
    def publish(self, pipeline_id: int):
        """Publishes a pipeline"""
        self.pipelines[pipeline_id].publish()
        
    def get_task(self,pipeline_id: int, task_name: str):
        """Given pipeline id and Task name return"""
        # Obtaining Pipeline and calling its get task method
        return self.db_client.query_first("pipeline_id")

    def _jobify():
        """Creates jobs in DAG-like fashion using compute graph"""
        pass

    def execute(pipeline: Pipeline):
        """Executes all jobs within a pipeline using job manager"""
        pass

    def _run_job(job: Job):
        """Uses a Job Manager to run a job"""
        pass

    def log_to_db():
        """A call to Job Manager's API for logging to a database"""
        pass

if __name__ == "__main__":
    mgr = PipelineManager()
    with open("data/configurations/portal_config.json", 'r') as jfile:
        config = json.load(jfile)
    mgr.create("p1", config)