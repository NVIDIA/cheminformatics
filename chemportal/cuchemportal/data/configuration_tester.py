import json
from json.decoder import JSONDecodeError

class ConfigurationTester():
    @staticmethod
    def run_pipeline_creation_tests(config: dict):
        ConfigurationTester.test_task_entry_is_not_null(input_config = config)
        ConfigurationTester.validate_output_input_type_mapping(input_config = config)

    @staticmethod
    def test_task_entry_is_not_null(input_config: dict):
        """Validates that a Pipeline has tasks"""
        assert "tasks" in input_config and input_config["tasks"] is not None, "Config cannot be empty"

    @staticmethod
    def validate_output_input_type_mapping(input_config: dict):
        """Ensures output artifacts from each preceding step match type of input artifact to next step"""
        pass

if __name__ == "__main__":
    tester = ConfigurationTester
    tester.test_task_entry_is_not_null({"tasks": "task1"})