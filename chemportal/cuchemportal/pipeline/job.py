from job_task import Job_Task
from dataclasses import dataclass
from datetime import date, datetime

@dataclass
class Job:
    """
    A Dataclass representation of a job - such a 
    representation should make for easy logging operations to a database
    """
    # Note: we could make job tasks a list of Job_Task type, but not logially coherent with parallelization
    job_id: int
    pipeline_id: int
    job_status: int  # Represents progression across job, whereas we can enumerate performance of task with simple enum
    job_tasks: dict # Will be a dictionary mapping tasks to DAG execution information (mini-computation graph)
    time_started: datetime = datetime.now()
    time_finished: datetime = None
    job_artifacts: dict = None

    def run_tasks():
        """Executes tasks in sequence / in parallel as given by the computation graph"""
        # This will run Tasks in correct sequence
        pass