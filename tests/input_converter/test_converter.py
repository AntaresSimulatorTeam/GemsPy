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
from antares.craft.model.study import Study

from gems.input_converter.src.converter import AntaresStudyConverter
from gems.input_converter.src.data_preprocessing.dataclasses import Operation
from gems.input_converter.src.logger import Logger
from gems.input_converter.src.utils import read_yaml_file, transform_to_yaml
from gems.study.parsing import (
    InputComponent,
    InputComponentParameter,
    InputPortConnections,
    InputSystem,
    parse_yaml_components,
)
from tests.input_converter.conftest import create_dataframe_from_constant

RESOURCES_FOLDER = (
    Path(__file__).parents[2]
    / "src"
    / "gems"
    / "input_converter"
    / "data"
    / "model_configuration"
)
DATAFRAME_PREPRO_SERIES = (
    create_dataframe_from_constant(lines=8760),  # series
)

DATAFRAME_PREPRO_THERMAL_CONFIG = (
    create_dataframe_from_constant(lines=8760, columns=4),  # modulation
    create_dataframe_from_constant(lines=8760),  # series
)

DATAFRAME_PREPRO_BC_CONFIG = (
    create_dataframe_from_constant(lines=8760, columns=6),  # modulation
    create_dataframe_from_constant(lines=8760, columns=4),  # series
)


class TestConverter:
    def _init_converter_from_study(self, local_study):
        logger = Logger(__name__, local_study.service.config.study_path)
        converter: AntaresStudyConverter = AntaresStudyConverter(
            study_input=local_study, logger=logger
        )
        return converter

    def _init_converter_from_path(self, local_path: Path):
        logger = Logger(__name__, str(local_path))
        converter: AntaresStudyConverter = AntaresStudyConverter(
            study_input=local_path, logger=logger
        )
        return converter

    def test_convert_study_to_input_study(self, local_study_w_areas: Study):
        converter = self._init_converter_from_study(local_study_w_areas)
        input_study = converter.convert_study_to_input_study()

        expected_input_study = InputSystem(
            nodes=[
                InputComponent(
                    id="fr",
                    model="antares-historic.area",
                    scenario_group=None,
                    parameters=[
                        InputComponentParameter(
                            id="ens_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=0.5,
                        ),
                        InputComponentParameter(
                            id="spillage_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=1.0,
                        ),
                    ],
                ),
                InputComponent(
                    id="it",
                    model="antares-historic.area",
                    scenario_group=None,
                    parameters=[
                        InputComponentParameter(
                            id="ens_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=0.5,
                        ),
                        InputComponentParameter(
                            id="spillage_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=1.0,
                        ),
                    ],
                ),
            ],
            components=[],
            connections=[],
        )

        assert input_study.nodes == expected_input_study.nodes

    def test_convert_area_to_component(self, local_study_w_areas: Study, lib_id: str):
        converter = self._init_converter_from_study(local_study_w_areas)
        area_components = converter._convert_area_to_component_list(
            lib_id, ["fr", "it"]
        )

        expected_area_components = [
            InputComponent(
                id="fr",
                model="antares-historic.area",
                parameters=[
                    InputComponentParameter(
                        id="ens_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.5,
                    ),
                    InputComponentParameter(
                        id="spillage_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                ],
            ),
            InputComponent(
                id="it",
                model="antares-historic.area",
                parameters=[
                    InputComponentParameter(
                        id="ens_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.5,
                    ),
                    InputComponentParameter(
                        id="spillage_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                ],
            ),
        ]

        assert area_components == expected_area_components

    def test_convert_area_to_yaml(self, local_study_w_areas: Study, lib_id: str):
        converter = self._init_converter_from_study(local_study_w_areas)
        area_components = converter._convert_area_to_component_list(
            lib_id, ["fr", "it"]
        )
        input_study = InputSystem(nodes=area_components)

        # Dump model into yaml file
        yaml_path = converter.study_path / "study_path.yaml"
        transform_to_yaml(model=input_study, output_path=yaml_path)

        # Open yaml file to validate
        with open(yaml_path, "r", encoding="utf-8") as yaml_file:
            validated_data = parse_yaml_components(yaml_file)

        expected_validated_data = InputSystem(
            nodes=[
                InputComponent(
                    id="it",
                    model="antares-historic.area",
                    scenario_group=None,
                    parameters=[
                        InputComponentParameter(
                            id="ens_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=0.5,
                        ),
                        InputComponentParameter(
                            id="spillage_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=1.0,
                        ),
                    ],
                ),
                InputComponent(
                    id="fr",
                    model="antares-historic.area",
                    scenario_group=None,
                    parameters=[
                        InputComponentParameter(
                            id="ens_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=0.5,
                        ),
                        InputComponentParameter(
                            id="spillage_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=1.0,
                        ),
                    ],
                ),
            ],
            components=[],
            connections=[],
        )
        expected_validated_data.nodes.sort(key=lambda x: x.id)
        validated_data.nodes.sort(key=lambda x: x.id)
        assert validated_data == expected_validated_data

    def test_convert_st_storages_to_component(
        self, local_study_with_st_storage, lib_id: str
    ):
        converter = self._init_converter_from_study(local_study_with_st_storage)
        path_load = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "st-storage.yaml"
        )

        resource_content = read_yaml_file(path_load).get("template", {})

        valid_areas: dict = converter._validate_resources_not_excluded(
            resource_content, "area"
        )

        (
            storage_components,
            storage_connections,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, resource_content)
        study_path = converter.study_path

        default_path = (
            study_path / "input" / "st-storage" / "series" / "fr" / "storage_1"
        )
        inflows_path = default_path / "inflows"
        lower_rule_curve_path = default_path / "lower-rule-curve"
        pmax_injection_path = default_path / "PMAX-injection"
        pmax_withdrawal_path = default_path / "PMAX-withdrawal"
        upper_rule_curve_path = default_path / "upper-rule-curve"
        expected_storage_connections = [
            InputPortConnections(
                component1="fr_storage_1",
                port1="injection_port",
                component2="fr",
                port2="balance_port",
            )
        ]
        expected_storage_component = [
            InputComponent(
                id="fr_storage_1",
                model=f"{lib_id}.short-term-storage",
                scenario_group=None,
                parameters=[
                    InputComponentParameter(
                        id="reservoir_capacity",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    InputComponentParameter(
                        id="injection_nominal_capacity",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=10.0,
                    ),
                    InputComponentParameter(
                        id="withdrawal_nominal_capacity",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=10.0,
                    ),
                    InputComponentParameter(
                        id="efficiency_injection",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1,
                    ),
                    InputComponentParameter(
                        id="efficiency_withdrawal",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1,
                    ),
                    InputComponentParameter(
                        id="lower_rule_curve",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{lower_rule_curve_path}",
                    ),
                    InputComponentParameter(
                        id="upper_rule_curve",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{upper_rule_curve_path}",
                    ),
                    InputComponentParameter(
                        id="p_max_injection_modulation",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{pmax_injection_path}",
                    ),
                    InputComponentParameter(
                        id="p_max_withdrawal_modulation",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{pmax_withdrawal_path}",
                    ),
                    InputComponentParameter(
                        id="inflows",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{inflows_path}",
                    ),
                    InputComponentParameter(
                        id="initial_level",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.5,
                    ),
                ],
            )
        ]

        assert storage_components == expected_storage_component
        assert storage_connections == expected_storage_connections

    # This parametrize allows to pass the parameter "DATAFRAME_PREPRO_THERMAL_CONFIG" inside the fixture
    # To specify the modulation and series dataframes
    @pytest.mark.parametrize(
        "local_study_w_thermal",
        [DATAFRAME_PREPRO_THERMAL_CONFIG],
        indirect=True,
    )
    def test_convert_thermals_to_component(
        self,
        local_study_w_thermal: Study,
    ):
        converter = self._init_converter_from_study(local_study_w_thermal)
        path_load = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "thermal.yaml"
        )

        resource_content = read_yaml_file(path_load).get("template", {})

        valid_areas: dict = converter._validate_resources_not_excluded(
            resource_content, "area"
        )

        (
            thermals_components,
            thermals_connections,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, resource_content)

        study_path = converter.study_path
        series_path = study_path / "input" / "thermal" / "series" / "fr" / "gaz"
        expected_thermals_connections = [
            InputPortConnections(
                component1="gaz",
                port1="balance_port",
                component2="fr",
                port2="balance_port",
            )
        ]
        expected_thermals_components = [
            InputComponent(
                id="gaz",
                model="antares-historic.thermal",
                scenario_group=None,
                parameters=[
                    InputComponentParameter(
                        id="p_min_cluster",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=str(series_path / "p_min_cluster"),
                    ),
                    InputComponentParameter(
                        id="nb_units_min",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=str(series_path / "nb_units_min"),
                    ),
                    InputComponentParameter(
                        id="nb_units_max",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=str(series_path / "nb_units_max"),
                    ),
                    InputComponentParameter(
                        id="nb_units_max_variation_forward",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=str(series_path / "nb_units_max_variation_forward"),
                    ),
                    InputComponentParameter(
                        id="nb_units_max_variation_backward",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=str(series_path / "nb_units_max_variation_backward"),
                    ),
                    InputComponentParameter(
                        id="unit_count",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                    InputComponentParameter(
                        id="p_min_unit",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    InputComponentParameter(
                        id="efficiency",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=100.0,
                    ),
                    InputComponentParameter(
                        id="p_max_unit",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=2.0,
                    ),
                    InputComponentParameter(
                        id="generation_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    InputComponentParameter(
                        id="fixed_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    InputComponentParameter(
                        id="startup_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    InputComponentParameter(
                        id="d_min_up",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                    InputComponentParameter(
                        id="d_min_down",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                    InputComponentParameter(
                        id="p_max_cluster",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=str(series_path / "series"),
                    ),
                ],
            )
        ]
        # TODO preprocessing + nouveaux parametres liées a la nouvelle version antarescraft
        assert thermals_components == expected_thermals_components
        assert thermals_connections == expected_thermals_connections

    def test_convert_load_to_component_from_path(self):
        path = Path(__file__).parent / "resources" / "mini_test_batterie_BP23"

        output_path = path / "reference.yaml"
        expected_data = read_yaml_file(output_path)["system"]
        expected_components = expected_data["components"]
        expected_connections = expected_data["connections"]

        converter = self._init_converter_from_path(path)
        path_load = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "load.yaml"
        )

        resource_content = read_yaml_file(path_load).get("template", {})

        valid_areas: dict = converter._validate_resources_not_excluded(
            resource_content, "area"
        )

        (
            load_components,
            load_connections,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, resource_content)

        ### Compare connections
        connection = load_connections[0]
        expected_connection: InputPortConnections = InputPortConnections(
            **next(
                (
                    connection
                    for connection in expected_connections
                    if connection["component1"] == "load_fr"
                ),
                None,
            )
        )

        assert connection == expected_connection
        ### Compare components
        expected_component = next(
            (
                component
                for component in expected_components
                if component["id"] == "load_fr"
            ),
            None,
        )

        # A little formatting of expected parameters:
        # Convert tiret fields with snake_case version
        # Add scenario group to None, if not present
        for item in expected_component["parameters"]:
            item["scenario_dependent"] = item.pop("scenario-dependent")
            item["time_dependent"] = item.pop("time-dependent")
            if not item.get("scenario_group"):
                item["scenario_group"] = None

        # A little formatting of obtained parameters:
        # Convert list of objects to list of dictionaries
        # Replace absolute path with relative path
        obtained_parameters_to_dict = [
            component.model_dump()
            for component in dict(load_components[0])["parameters"]
        ]
        obtained_parameters = TestConverter._match_area_pattern(
            obtained_parameters_to_dict, "", str(path) + "/"
        )
        assert obtained_parameters == expected_component["parameters"]

    @pytest.mark.parametrize(
        "fr_solar",
        [DATAFRAME_PREPRO_SERIES],
        indirect=True,
    )
    def test_convert_solar_to_component_from_study(self, fr_solar: None):
        converter = self._init_converter_from_study(fr_solar)

        path_load = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "solar.yaml"
        )

        resource_content = read_yaml_file(path_load).get("template", {})

        valid_areas: dict = converter._validate_resources_not_excluded(
            resource_content, "area"
        )

        (
            solar_components,
            solar_connections,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, resource_content)
        solar_fr_component = next(
            (comp for comp in solar_components if comp.id == "solar_fr"), None
        )
        solar_fr_connection = next(
            (conn for conn in solar_connections if conn.component1 == "solar_fr"), None
        )
        solar_timeseries = str(
            converter.study_path / "input" / "solar" / "series" / "generation_fr"
        )
        expected_solar_connection = InputPortConnections(
            component1="solar_fr",
            port1="balance_port",
            component2="fr",
            port2="balance_port",
        )

        expected_solar_components = InputComponent(
            id="solar_fr",
            model="antares-historic.renewable",
            scenario_group=None,
            parameters=[
                InputComponentParameter(
                    id="nominal_capacity",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                    scenario_group=None,
                ),
                InputComponentParameter(
                    id="unit_count",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                    scenario_group=None,
                ),
                InputComponentParameter(
                    id="generation",
                    time_dependent=True,
                    scenario_dependent=True,
                    value=f"{solar_timeseries}",
                    scenario_group=None,
                ),
            ],
        )
        assert solar_fr_connection == expected_solar_connection
        assert solar_fr_component.model_dump() == expected_solar_components.model_dump()

    @pytest.mark.parametrize(
        "fr_load",
        [DATAFRAME_PREPRO_SERIES],
        indirect=True,
    )
    def test_convert_load_to_component_from_study(self, fr_load: None):
        converter = self._init_converter_from_study(fr_load)
        path_load = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "load.yaml"
        )

        resource_content = read_yaml_file(path_load).get("template", {})

        valid_areas: dict = converter._validate_resources_not_excluded(
            resource_content, "area"
        )

        (
            load_components,
            load_connections,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, resource_content)
        load_fr_component = next(
            (comp for comp in load_components if comp.id == "load_fr"), None
        )
        load_fr_connection = next(
            (conn for conn in load_connections if conn.component1 == "load_fr"), None
        )

        load_timeseries = str(
            converter.study_path / "input" / "load" / "series" / "load_fr"
        )
        expected_load_connection = InputPortConnections(
            component1="load_fr",
            port1="balance_port",
            component2="fr",
            port2="balance_port",
        )
        expected_load_components = InputComponent(
            id="load_fr",
            model="antares-historic.load",
            scenario_group=None,
            parameters=[
                InputComponentParameter(
                    id="load",
                    time_dependent=True,
                    scenario_dependent=True,
                    value=f"{load_timeseries}",
                    scenario_group=None,
                ),
            ],
        )
        assert load_fr_connection == expected_load_connection
        assert load_fr_component.model_dump() == expected_load_components.model_dump()

    @pytest.mark.parametrize(
        "fr_wind",
        [DATAFRAME_PREPRO_SERIES],
        indirect=True,
    )
    def test_convert_wind_to_component_from_study(self, fr_wind: Study):
        converter = self._init_converter_from_study(fr_wind)

        path_load = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "wind.yaml"
        )

        resource_content = read_yaml_file(path_load).get("template", {})

        valid_areas: dict = converter._validate_resources_not_excluded(
            resource_content, "area"
        )

        (
            wind_components,
            wind_connections,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, resource_content)
        wind_fr_component = next(
            (comp for comp in wind_components if comp.id == "wind_fr"), None
        )
        wind_fr_connection = next(
            (conn for conn in wind_connections if conn.component1 == "wind_fr"), None
        )

        wind_timeseries = str(
            converter.study_path / "input" / "wind" / "series" / "generation_fr"
        )
        expected_wind_connection = InputPortConnections(
            component1="wind_fr",
            port1="balance_port",
            component2="fr",
            port2="balance_port",
        )
        # ON touche plus
        expected_wind_components = InputComponent(
            id="wind_fr",
            model="antares-historic.renewable",
            scenario_group=None,
            parameters=[
                InputComponentParameter(
                    id="nominal_capacity",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                    scenario_group=None,
                ),
                InputComponentParameter(
                    id="unit_count",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                    scenario_group=None,
                ),
                InputComponentParameter(
                    id="generation",
                    time_dependent=True,
                    scenario_dependent=True,
                    scenario_group=None,
                    value=f"{wind_timeseries}",
                ),
            ],
        )
        assert wind_fr_connection == expected_wind_connection
        assert wind_fr_component.model_dump() == expected_wind_components.model_dump()

    @pytest.mark.parametrize(
        "fr_wind",
        [
            pd.DataFrame(),  # DataFrame empty
        ],
        indirect=True,
    )
    def test_convert_wind_to_component_empty_file(
        self,
        fr_wind: object,
    ):
        converter = self._init_converter_from_study(fr_wind)

        path_load = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "wind.yaml"
        )

        resource_content = read_yaml_file(path_load).get("template", {})

        valid_areas: dict = converter._validate_resources_not_excluded(
            resource_content, "area"
        )

        (
            wind_components,
            _,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, resource_content)
        assert wind_components == []

    @pytest.mark.parametrize(
        "fr_wind",
        [
            pd.DataFrame([0, 0, 0]),  # DataFrame full of 0
        ],
        indirect=True,
    )
    def test_convert_wind_to_component_zero_values(self, fr_wind: int):
        converter = self._init_converter_from_study(fr_wind)

        path_load = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "wind.yaml"
        )

        resource_content = read_yaml_file(path_load).get("template", {})

        valid_areas: dict = converter._validate_resources_not_excluded(
            resource_content, "area"
        )

        (
            wind_components,
            _,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, resource_content)
        assert wind_components == []

    def test_convert_links_to_component(self, local_study_w_links: Study, lib_id: str):
        converter = self._init_converter_from_study(local_study_w_links)
        path_load = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "link.yaml"
        )
        resource_content = read_yaml_file(path_load).get("template", {})

        valid_areas: dict = converter._validate_resources_not_excluded(
            resource_content, "area"
        )

        (
            links_components,
            links_connections,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, resource_content)
        study_path = converter.study_path

        fr_prefix_path = study_path / "input" / "links" / "fr" / "capacities"
        at_prefix_path = study_path / "input" / "links" / "at" / "capacities"
        fr_it_direct_links_timeseries = str(fr_prefix_path / "it_direct")
        fr_it_indirect_links_timeseries = str(fr_prefix_path / "it_indirect")
        at_fr_direct_links_timeseries = str(at_prefix_path / "fr_direct")
        at_fr_indirect_links_timeseries = str(at_prefix_path / "fr_indirect")
        at_it_direct_links_timeseries = str(at_prefix_path / "it_direct")
        at_it_indirect_links_timeseries = str(at_prefix_path / "it_indirect")
        expected_link_component = [
            InputComponent(
                id="fr / it",
                model="antares-historic.link",
                scenario_group=None,
                parameters=[
                    InputComponentParameter(
                        id="capacity_direct",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{fr_it_direct_links_timeseries}",
                    ),
                    InputComponentParameter(
                        id="capacity_indirect",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{fr_it_indirect_links_timeseries}",
                    ),
                ],
            ),
            InputComponent(
                id="at / fr",
                model="antares-historic.link",
                scenario_group=None,
                parameters=[
                    InputComponentParameter(
                        id="capacity_direct",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{at_fr_direct_links_timeseries}",
                    ),
                    InputComponentParameter(
                        id="capacity_indirect",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{at_fr_indirect_links_timeseries}",
                    ),
                ],
            ),
            InputComponent(
                id="at / it",
                model="antares-historic.link",
                scenario_group=None,
                parameters=[
                    InputComponentParameter(
                        id="capacity_direct",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{at_it_direct_links_timeseries}",
                    ),
                    InputComponentParameter(
                        id="capacity_indirect",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{at_it_indirect_links_timeseries}",
                    ),
                ],
            ),
        ]
        
        expected_link_connections = [
            InputPortConnections(
                component1="at / fr",
                port1="in_port",
                component2="at",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="at / fr",
                port1="out_port",
                component2="fr",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="at / it",
                port1="in_port",
                component2="at",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="at / it",
                port1="out_port",
                component2="it",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="fr / it",
                port1="in_port",
                component2="fr",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="fr / it",
                port1="out_port",
                component2="it",
                port2="balance_port",
            ),
        ]
        assert sorted(links_components, key=lambda x: x.id) == sorted(expected_link_component, key=lambda x: x.id)
        assert links_connections == expected_link_connections

    @staticmethod
    def _match_area_pattern(object, param_values: dict[str, str], pattern: str) -> any:
        if isinstance(object, dict):
            return {
                TestConverter._match_area_pattern(
                    k, param_values, pattern
                ): TestConverter._match_area_pattern(v, param_values, pattern)
                for k, v in object.items()
            }
        elif isinstance(object, list):
            return [
                TestConverter._match_area_pattern(elem, param_values, pattern)
                for elem in object
            ]
        elif isinstance(object, str):
            return object.replace(pattern, param_values)
        else:
            return object

    def test_convert_bc_to_component_from_path(self):
        path = Path(__file__).parent / "resources" / "mini_test_batterie_BP23"

        output_path = path / "reference.yaml"
        expected_data = read_yaml_file(output_path)["system"]

        expected_components = expected_data["components"]
        expected_connections = expected_data["connections"]
        converter = self._init_converter_from_path(path)
        path_cc = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "battery_new.yaml"
        )

        bc_data = read_yaml_file(path_cc).get("template", {})
        valid_areas: dict = converter._validate_resources_not_excluded(bc_data, "area")
        # TODO pour les preprocessng, il faut revenir a letat initial. du coup ca implique
        # De copier le dossier entier, le tester, le supprimer apres
        (
            binding_components,
            binding_connections,
            _,
        ) = converter._convert_model_to_component_list(valid_areas, bc_data)
        connection = binding_connections[0]

        # Compare connections
        expected_connection: InputPortConnections = InputPortConnections(
            **next(
                (
                    connection
                    for connection in expected_connections
                    if connection["component1"] == "battery_fr"
                ),
                None,
            )
        )

        assert connection == expected_connection
        # Compare components
        expected_component = next(
            (
                component
                for component in expected_components
                if component["id"] == "battery_fr"
            ),
            None,
        )

        # A little formatting of expected parameters:
        # Convert tiret fields with snake_case version
        # Add scenario group to None, if not present
        for item in expected_component["parameters"]:
            item["scenario_dependent"] = item.pop("scenario-dependent")
            item["time_dependent"] = item.pop("time-dependent")
            if not item.get("scenario_group"):
                item["scenario_group"] = None

        # A little formatting of obtained parameters:
        # Convert list of objects to list of dictionaries
        # Replace absolute path with relative path
        obtained_parameters_to_dict = [
            component.model_dump()
            for component in dict(binding_components[0])["parameters"]
        ]
        obtained_parameters = TestConverter._match_area_pattern(
            obtained_parameters_to_dict, "", str(path) + "/"
        )

        assert obtained_parameters == expected_component["parameters"]

    def test_convert_study_path_to_input_study(self):
        path = Path(__file__).parent / "resources" / "mini_test_batterie_BP23"
        output_path = path / "reference.yaml"
        expected_data = read_yaml_file(output_path)["system"]
        converter = self._init_converter_from_path(path)
        obtained_data = converter.convert_study_to_input_study()

        # A little formatting of expected parameters:
        # Convert tiret fields with snake_case version
        # Add scenario group to None, if not present
        for component in expected_data["components"]:
            if not component.get("scenario_group"):
                component["scenario_group"] = None
            for item in component["parameters"]:
                item["scenario_dependent"] = item.pop("scenario-dependent")
                item["time_dependent"] = item.pop("time-dependent")
                if not item.get("scenario_group"):
                    item["scenario_group"] = None
        # A little formatting of obtained parameters:
        # Convert list of objects to list of dictionaries
        # Replace absolute path with relative path
        obtained_components_to_dict = [
            component.model_dump() for component in dict(obtained_data)["components"]
        ]
        obtained_components = TestConverter._match_area_pattern(
            obtained_components_to_dict, "", str(path) + "/"
        )

        def normalize_components(components):
            return [
                {
                    **c,
                    "parameters": [
                        {
                            k: p[k]
                            for k in (
                                "id",
                                "value",
                                "scenario_dependent",
                                "time_dependent",
                                "scenario_group",
                            )
                        }
                        for p in c["parameters"]
                    ],
                }
                for c in components
            ]

        assert sorted(
            normalize_components(obtained_components), key=lambda x: x["id"]
        ) == sorted(expected_data["components"], key=lambda x: x["id"])

    def test_multiply_operation(self):
        operation = Operation(multiply_by=2)
        assert operation.execute(10) == 20

        operation = Operation(multiply_by="factor")
        preprocessed_values = {"factor": 5}
        assert operation.execute(10, preprocessed_values) == 50

        operation = Operation(multiply_by=2)
        df = pd.Series([1, 2, 3, 4, 5, 6])
        assert operation.execute(df).all() == pd.Series([2, 4, 6, 8, 10, 12]).all()

    def test_divide_operation(self):
        operation = Operation(divide_by=2)
        assert operation.execute(10) == 5

        operation = Operation(divide_by="divisor")
        preprocessed_values = {"divisor": 2}
        assert operation.execute(10, preprocessed_values) == 5

        operation = Operation(divide_by=2)
        df = pd.Series([1, 2, 3, 4, 5, 6])
        assert operation.execute(df).all() == pd.Series([0.5, 1, 1.5, 2, 2.5, 3]).all()

    def test_max_operation(self):
        operation = Operation(type="max")
        assert operation.execute([1, 2, 3, 4, 5]) == 5.0

        df = pd.Series([1, 2, 3, 4, 5, 6])
        assert operation.execute(df) == 6.0

    def test_missing_preprocessed_value(self):
        operation = Operation(multiply_by="missing_key")
        with pytest.raises(ValueError):
            operation.execute(10, {})

    def test_missing_operation(self):
        operation = Operation()
        with pytest.raises(ValueError):
            operation.execute(10)
