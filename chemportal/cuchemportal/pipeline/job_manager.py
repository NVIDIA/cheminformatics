from cuchemportal.pipeline.job_task import JobTask
from cuchemportal.data.db_client import DBClient
from cuchemportal.pipeline.job import Job


class JobManager:
    """An Execution Engine for Job Operations"""

    def __init__(self):
        """Class will function as job executor in composition with its role as DB logger"""
        pass

    def _log_to_db(self,job: Job) -> bool:
        """Logs a job and its task data to database"""
        pass

    def execute_jobs(self,job: Job) -> bool:
        """Runs jobs in given sequence"""
        pass

    def _construct_task_list(self,job: Job) -> list:
        """Given a job, constructs its task list"""
        pass

