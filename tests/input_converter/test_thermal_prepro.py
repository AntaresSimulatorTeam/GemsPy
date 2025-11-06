# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.
import os
from pathlib import Path

import pandas as pd
import pytest
from antares.craft.model.area import Area
from antares.craft.model.study import Study
from antares.craft.model.thermal import ThermalCluster

from gems.input_converter.src.converter import AntaresStudyConverter
from gems.input_converter.src.data_preprocessing.thermal import (
    Direction,
    ThermalDataPreprocessing,
)
from gems.input_converter.src.logger import Logger
from gems.study.parsing import InputComponentParameter
from tests.input_converter.conftest import create_dataframe_from_constant

DATAFRAME_PREPRO_THERMAL_CONFIG = (
    create_dataframe_from_constant(lines=8760, columns=4, value=3),  # modulation
    create_dataframe_from_constant(lines=8760, columns=1, value=6),  # series
)
LIB_PATHS = [
    "src/gems/libs/antares_historic/antares_historic.yml",
    "src/gems/libs/reference_models/andromede_v1_models.yml",
]
LIB_PATHS_WITH_BASE = [str(Path(os.getcwd()) / suffix) for suffix in LIB_PATHS]


class TestThermalPreprocessing:
    @staticmethod
    def setup_preprocessing_thermal(
        local_study_w_thermal: Study,
    ) -> AntaresStudyConverter:
        """
        Initializes test parameters and returns the instance and expected file path.
        """

        logger = Logger(__name__, local_study_w_thermal.path)
        converter: AntaresStudyConverter = AntaresStudyConverter(
            study_input=local_study_w_thermal,
            logger=logger,
            mode="full",
            lib_paths=LIB_PATHS,
            output_folder=local_study_w_thermal.path.parent / "converter_output",
        )

        return converter

    @staticmethod
    def get_first_thermal_cluster_from_study(
        converter: AntaresStudyConverter, area_id: str = "fr"
    ) -> ThermalCluster:
        areas: dict[Area] = converter.study.get_areas().values()

        thermal: ThermalCluster = next(
            (
                thermal
                for area in areas
                for thermal in area.get_thermals().values()
                if thermal.area_id == area_id
            ),
            None,
        )
        return thermal

    def _init_tdp(self, local_study_w_thermal: Study) -> ThermalDataPreprocessing:
        converter = self.setup_preprocessing_thermal(local_study_w_thermal)
        thermal: ThermalCluster = self.get_first_thermal_cluster_from_study(converter)
        return ThermalDataPreprocessing(thermal, converter.output_folder, suffix=".txt")

    def _validate_component_parameter(
        self,
        timeserie_file_path: Path,
        component_parameter: InputComponentParameter,
        component_id: str,
        expected_values: list,
    ):
        """
        Executes the given processing method, validates the component, and compares the output dataframe.
        """

        expected_component = InputComponentParameter(
            id=component_id,
            time_dependent=True,
            scenario_dependent=True,
            value=component_parameter.value,
        )
        # current_path = Path(component_parameter.value).with_suffix(".txt")
        current_df = pd.read_csv(timeserie_file_path, header=None)
        expected_df = pd.DataFrame(expected_values)
        assert current_df.equals(expected_df)

        assert component_parameter == expected_component

    @pytest.mark.parametrize(
        "local_study_w_thermal",
        [DATAFRAME_PREPRO_THERMAL_CONFIG],
        indirect=True,
    )
    def test_p_min_cluster(self, local_study_w_thermal):
        """Tests the p_min_cluster parameter processing."""
        tdp: ThermalDataPreprocessing = self._init_tdp(local_study_w_thermal)

        expected_values = create_dataframe_from_constant(
            lines=8760, columns=1, value=6.0
        )  # min(min_gen_modulation * unit_count * nominal_capacity, p_max_cluster)
        expected_values = expected_values.squeeze()
        expected_values.name = None
        component_parameter = tdp.generate_component_parameter("p_min_cluster")
        filepath, _ = tdp._build_csv_path_and_name("p_min_cluster")
        self._validate_component_parameter(
            filepath, component_parameter, "p_min_cluster", expected_values
        )

    @pytest.mark.parametrize(
        "local_study_w_thermal",
        [DATAFRAME_PREPRO_THERMAL_CONFIG],
        indirect=True,
    )
    def test_nb_units_min(self, local_study_w_thermal: Study):
        """Tests the nb_units_min parameter processing."""
        tdp: ThermalDataPreprocessing = self._init_tdp(local_study_w_thermal)

        expected_values = create_dataframe_from_constant(
            lines=8760, columns=1, value=3.0
        )  # ceil(p_min_cluster / p_max_unit) en float
        expected_values = expected_values.squeeze()
        expected_values.name = None
        tdp.generate_component_parameter("p_min_cluster")
        component_parameter = tdp.generate_component_parameter("nb_units_min")
        filepath, _ = tdp._build_csv_path_and_name("nb_units_min")
        self._validate_component_parameter(
            filepath, component_parameter, "nb_units_min", expected_values
        )

    @pytest.mark.parametrize(
        "local_study_w_thermal",
        [DATAFRAME_PREPRO_THERMAL_CONFIG],
        indirect=True,
    )
    def test_nb_units_max(self, local_study_w_thermal: Study):
        """Tests the nb_units_max parameter processing."""
        tdp: ThermalDataPreprocessing = self._init_tdp(local_study_w_thermal)

        expected_values = create_dataframe_from_constant(
            lines=8760, columns=1, value=3.0
        )  # ceil(p_max_cluster / p_max_unit) en float
        expected_values = expected_values.squeeze()
        expected_values.name = None
        tdp.generate_component_parameter("p_min_cluster")
        component_parameter = tdp.generate_component_parameter("nb_units_max")
        filepath, _ = tdp._build_csv_path_and_name("nb_units_max")
        self._validate_component_parameter(
            filepath, component_parameter, "nb_units_max", expected_values
        )

    def nb_units_max_variation(
        self,
        local_study_w_thermal: Study,
        direction: Direction,
    ):
        """
        Tests nb_units_max_variation_forward and nb_units_max_variation_backward processing.
        """

        tdp: ThermalDataPreprocessing = self._init_tdp(local_study_w_thermal)

        expected_path = (
            tdp.study_path
            / "input"
            / "data-series"
            / f"fr_gaz_nb_units_max_variation_{direction.value}.txt"
        )
        tdp.generate_component_parameter("nb_units_max")
        nb_unit_max_path, _ = tdp._build_csv_path_and_name("nb_units_max")
        variation_component = tdp.generate_component_parameter(
            f"nb_units_max_variation_{direction.value}"
        )

        series_path, _ = tdp._build_csv_path_and_name(
            f"nb_units_max_variation_{direction.value}"
        )
        current_df = pd.read_csv(series_path, header=None)

        nb_units_max_output = pd.read_csv(nb_unit_max_path, header=None)

        assert current_df[0][0] == max(
            0, nb_units_max_output[0][167] - nb_units_max_output[0][0]
        )
        assert current_df[0][3] == max(
            0, nb_units_max_output[0][2] - nb_units_max_output[0][3]
        )
        assert current_df[0][168] == max(
            0, nb_units_max_output[0][335] - nb_units_max_output[0][168]
        )

        assert str(series_path) == str(expected_path)

    @pytest.mark.parametrize(
        "direction, local_study_w_thermal",
        [
            (Direction.FORWARD, DATAFRAME_PREPRO_THERMAL_CONFIG),
            (Direction.BACKWARD, DATAFRAME_PREPRO_THERMAL_CONFIG),
        ],
        indirect=["local_study_w_thermal"],
    )
    def test_nb_units_max_variation(
        self, local_study_w_thermal: Study, direction: Direction
    ):
        self.nb_units_max_variation(local_study_w_thermal, direction)
