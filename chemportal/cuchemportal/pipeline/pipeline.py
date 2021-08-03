from sqlalchemy.sql.sqltypes import Boolean, Date
from cuchemportal.pipeline.job_task import JobTask
from cuchemportal.data.configuration_tester import ConfigurationTester
import copy
from typing import Union, Optional
import json
from sqlalchemy.orm import registry
from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON, Boolean, DateTime

mapper_registry = registry()

@mapper_registry.mapped
class Pipeline:

    __tablename__ = "pipelines"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    definition = Column(JSON)
    is_published = Column(Boolean)
    ui_config = Column(JSON)
    user = Column(String)
    time_created = Column(DateTime)
    last_updated = Column(DateTime)

    
    """A State storing object representing a pipeline """
    def __init__(self, input_artifacts: dict = None):
        self._config: dict = None
        self.is_published: bool = False 
        self.input_artifacts: dict = input_artifacts
        self.attributes = (["id", "name", "definition", 
                        "is_published", "ui_config", "user", "time_created", "last_updated"])

    @property
    def config(self):
        """A builder method for the Pipeline which sets configuration attribute"""
        return self._config

    # Note: This is different then what was discussed, but seems to be an
    # intuitive way to interact with Pipeline manager, which will call this
    @config.setter
    def config(self, new_config: Optional[dict] = None, config_path: Optional[str] = None):
        """Validates input and sets configuration if it is valid"""
        # input validation pre config build, setting so long as not published
        assert not self.is_published, "Cannot edit a published pipeline"
        if new_config is None:
            if config_path is not None:
                with open(config_path, "r") as jsonfile:
                    new_config = json.load(jsonfile)
            else:
                raise ValueError("config_path and new_config cannot both be None")

        # Running test suite to validate input so that config not used unless is valid
        ConfigurationTester.run_pipeline_creation_tests(new_config)
        self._initialize_attrs(new_config)
        self._config = new_config

    def _initialize_attrs(self, new_config: dict):
        """A setter for all attributes which are present in the config"""
        for attribute in self.attributes:
            # Setting all attributes which are present in the config
            if attribute in new_config:
                setattr(self, attribute, new_config[attribute])
        
    @config.getter
    def config(self) -> dict:
        return self._config

    def publish(self):
        self.is_published = True

    def __str__(self):
        """Returns Pipeline id if initialized"""
        if hasattr(self, "config"):
            return "Pipeline #" + str(self._config["id"])
        elif hasattr(self, "id"):
            return "Pipeline #" + str(self.id)
        else:
            return "uninitialized pipeline"

    @property
    def tasks(self):
        return self._config["tasks"]

    def get_input_artifacts(self) -> dict:
        return self.input_artifacts
    