from job_task import Job_Task 
from cuchemportal.data.db_client import DBClient
from cuchemportal.pipeline.job import Job


class Job_Manager:
    """An Execution Engine for Job Operations"""

    def __init__():
        """Class will function as job executor in composition with its role as DB logger"""
        db_client = DBClient()

    def _log_to_db(job: Job) -> bool:
        """Logs a job and its task data to database"""
        pass

    def execute_jobs(job: Job) -> bool:
        """Runs jobs in given sequence"""
        pass

    def _construct_task_list(job: Job) -> list[Job_Task]:
        """Given a job, constructs its task list"""
        pass

    