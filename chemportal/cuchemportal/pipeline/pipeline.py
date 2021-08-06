import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.sql.sqltypes import Boolean
from sqlalchemy.orm import registry
from sqlalchemy import Column, Integer, String, JSON, Boolean, DateTime

from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from cuchemportal.data.configuration_tester import ConfigurationTester
from cuchemportal.data import BaseModel

mapper_registry = registry()

logger = logging.getLogger(__name__)


@mapper_registry.mapped
class Pipeline(BaseModel):
    __tablename__ = "pipelines"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    definition = Column(JSON)
    ui_config = Column(JSON)
    is_published = Column(Boolean)
    is_deleted = Column(Boolean)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    create_by = Column(String)
    updated_by = Column(String)

    """A State storing object representing a pipeline """
    def __init__(self, input_artifacts: dict = None):
        self._config: dict = None
        self.is_published: bool = False
        self.is_deleted: bool = False
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.input_artifacts: dict = input_artifacts

        # TODO: Do we really need this?
        self.attributes = (["id", "name", "description", "definition",
                            "ui_config", "is_published", "is_deleted",
                            "created_at", "updated_at" "create_by", "updated_by"])

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

    # def __str__(self):
    #     """Returns Pipeline id if initialized"""
    #     if hasattr(self, "config"):
    #         return "Pipeline #" + str(self._config["id"])
    #     elif hasattr(self, "id"):
    #         return "Pipeline #" + str(self.id)
    #     else:
    #         return "uninitialized pipeline"

    @property
    def tasks(self):
        return self._config["tasks"]

    def get_input_artifacts(self) -> dict:
        return self.input_artifacts


class PipelineSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Pipeline

pipeline_schema = PipelineSchema()
pipelines_schema = PipelineSchema(many=True)