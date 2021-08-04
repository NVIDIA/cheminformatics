from sqlalchemy.sql.sqltypes import Date, DateTime
from cuchemportal.pipeline.job_task import JobTask
from dataclasses import dataclass
from datetime import date, datetime
from sqlalchemy.orm import registry
from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON

mapper_registry = registry()

@mapper_registry.mapped
@dataclass
class Job:
    """
    A Dataclass representation of a job - such a 
    representation should make for easy logging operations to a database
    """
    # Note: we could make job tasks a list of Job_Task type, but not logially coherent with parallelization
    __tablename__ = "jobs"
    job_id = Column(Integer, primary_key=True)
    pipeline_id =  Column(Integer)
    job_status = Column(Integer)  # Represents progression across job, whereas we can enumerate performance of task with simple enum
    job_tasks = Column(JSON) # Will be a dictionary mapping tasks to DAG execution information (mini-computation graph)
    time_started = Column(DateTime)
    time_finishe = Column(DateTime)
    job_configuration = Column(JSON)

    def run_tasks():
        """Executes tasks in sequence / in parallel as given by the computation graph"""
        # This will run Tasks in correct sequence
        pass