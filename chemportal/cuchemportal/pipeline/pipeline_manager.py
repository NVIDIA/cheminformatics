from cuchemportal.pipeline.job import Job
from os import pipe
from cuchemportal.pipeline.job_manager import JobManager
from cuchemportal.pipeline.pipeline import Pipeline
from cuchemportal.data.db_client import DBClient
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

    def create(self, ppln: Pipeline):
        """Given Configuration, uses pipeline setter method to create pipeline"""

        # Adding Pipeline to dict of Pipelines
        with self.db_client.Session() as sess:
            record = self.db_client.insert(record=ppln, session=sess)
            sess.commit()
        return record

    def update(self, previous_id: int, new_conf: dict) -> bool:
        """Receives an updated pipeline from the UI and precedes to reset config"""
        with self.db_client.Session() as sess:
            # Using DB Clients query id API - to be changed to more general query API is possible
            pipeline = self.db_client.update_record(db_table = Pipeline, id = previous_id,new_config=new_conf, session=sess)
            sess.commit()
        # Returning autoconverted pipeline
        return pipeline

    def fetch_by_id(self, pipeline_id: int) -> Pipeline:
        """Given a pipeline id, returns a pipeline object as configured in the database"""
        with self.db_client.Session() as sess:
            # Using DB Clients query id API - to be changed to more general query API is possible
            pipeline = self.db_client.query_id(db_table = Pipeline, id = pipeline_id, session=sess)
        # Returning autoconverted pipeline
        return pipeline

    def fetch_all(self, start_index:int, end_index: int):
        """Fetches all Pipelines in the interval [start,end)"""
        with self.db_client.Session() as sess:
            # Using DB Clients query id API - to be changed to more general query API is possible
            pipelines = self.db_client.query_range(db_table = Pipeline, start_idx = start_index, 
                                                end_idx = end_index,
                                                session=sess)
            sess.commit()
        # Returning autoconverted pipeline
        return pipelines

    # Todo: add is_deleted column to Pipeline and mark pipelines as deleted
    def delete(self, pipeline_id: int):
        """Deletes Pipeline Object"""
        with self.db_client.Session() as sess:
            deleted = self.db_client.delete(db_table = Pipeline, id = pipeline_id, session=sess)
            sess.commit()
        return deleted


    #def clone_pipeline(self, pipeline_id: int):
        #"""Clones a Pipeline Object"""
        # Since overriden, will call the Pipeline class deepcopy implementation which we can modify
        #return deepcopy(self.pipelines[pipeline_id])

    # Todo: now access to database
    def publish(self, pipeline_id: int):
        """Publishes a pipeline"""
        self.pipelines[pipeline_id].publish()

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