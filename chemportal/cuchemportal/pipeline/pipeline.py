from cuchemportal.pipeline.job_task import Job_Task
from cuchemportal.data.configuration_tester import ConfigurationTester
import copy
from typing import Union, Optional
import json

# Todo: add a way to order pipelines/ sort them by id?

class Pipeline:
    """A State storing object representing a pipeline """
    def __init__(self, input_artifacts: dict = None):
        self._config: dict = None
        self._compute_graph = None # Todo: what is the type of this? How will this be structured?
        self.is_published: bool = False 
        self.input_artifacts: dict = input_artifacts

    @property
    def config(self):
        """A builder method for the Pipeline which sets configuration attribute"""
        return self._config

    # Note: This is different then what was discussed, but seems to be an
    # intuitive way to interact with Pipeline manager, which will call this
    @config.setter
    def config_setter(self, config_path: Optional[str] = None, new_config: Optional[dict] = None):
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
        self._config = new_config

    @config.getter
    def config_getter(self) -> dict:
        return self._config

    def publish(self):
        self.is_published = True

    @property
    def tasks(self):
        return self._config["tasks"]

    def get_input_artifacts(self) -> dict:
        return self.input_artifacts
    