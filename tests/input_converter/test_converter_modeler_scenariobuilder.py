from pathlib import Path

import pytest

from gems.input_converter.src.converter import AntaresStudyConverter
from gems.input_converter.src.logger import Logger


class TestAntaresStudyConverterReal:
    @pytest.fixture
    def study_path(self):
        return Path("tests/input_converter/resources/mini_test_BP_conversion")

    @pytest.fixture
    def scenario_builder_file(self, study_path):
        scenario_file = study_path / "input" / "modeler-scenariobuilder.dat"
        scenario_file.parent.mkdir(parents=True, exist_ok=True)
        return scenario_file

    @pytest.fixture
    def library_file(self, study_path):
        return study_path / "antares_legacy_models.yml"

    @pytest.fixture
    def output_folder(self, tmp_path):
        output_dir = tmp_path / "converted_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def test_conversion_with_scenario_builder(
        self, study_path, scenario_builder_file, library_file, output_folder
    ):
        converter = AntaresStudyConverter(
            study_input=study_path,
            logger=Logger("test-scenario-builder", str(study_path)),
            output_folder=output_folder,
            lib_paths=[library_file],
            mode="full",
            models_to_convert=["wind"],
            modeler_scenario_builder_file=scenario_builder_file,
        )

        input_system = converter.convert_study_to_input_system()

        copied_files = list(output_folder.glob("**/modeler-scenariobuilder.dat"))
        assert copied_files, "Scenario builder file was not copied to output directory"

        scenario_found = any(
            getattr(comp, "scenario_group", None) is not None
            for comp in input_system.components
        )
        assert scenario_found, "No component has scenario_group"
