from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class TaskStatus(Enum):
    """Enum representing the task status"""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"


@dataclass()
class JobTask:
    """"Class representing the frozen data for a task instance"""
    task_id: int
    job_id: int
    task_name: str
    implementation_config: dict
    task_status: TaskStatus
    time_started: datetime = datetime.now()
    time_finished: datetime = None
    exit_code: int = 0
    exit_message: str = ""