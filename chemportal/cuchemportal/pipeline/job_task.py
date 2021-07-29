from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class Task_Status(Enum):
    """Enum representing the task status"""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"


@dataclass()
class Job_Task:
    """"Class representing the frozen data for a task instance"""
    task_id: int
    job_id: int
    task_name: str
    implementation_config: dict
    task_status: Task_Status
    time_started: datetime = datetime.now()
    time_finished: datetime = None
    exit_code: int = 0 
    exit_message: str = ""