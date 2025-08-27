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

DATAFRAME_PREPRO_THERMAL_CONFIG = (
    create_dataframe_from_constant(lines=840, columns=4),  # modulation
    create_dataframe_from_constant(lines=840),  # series
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

        # To ensure that the comparison between the actual and expected results is not affected by the order of the nodes,
        # both the area_components.nodes and expected_area_components.nodes lists are sorted by the id attribute of each node.
        # This sorting step ensures that the test checks only the presence and validity of the nodes, not their order.
        input_study.nodes.sort(key=lambda x: x.id)
        expected_input_study.nodes.sort(key=lambda x: x.id)

        assert input_study == expected_input_study

    def test_convert_area_to_component(self, local_study_w_areas: Study, lib_id: str):
        converter = self._init_converter_from_study(local_study_w_areas)
        area_components = converter._convert_area_to_component_list(lib_id, {})

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

        # To ensure that the comparison between the actual and expected results is not affected by the order of the nodes,
        # both the area_components.nodes and expected_area_components.nodes lists are sorted by the id attribute of each node.
        # This sorting step ensures that the test checks only the presence and validity of the nodes, not their order.
        expected_area_components.sort(key=lambda x: x.id)
        area_components.sort(key=lambda x: x.id)
        assert area_components == expected_area_components

    def test_convert_area_to_yaml(self, local_study_w_areas: Study, lib_id: str):
        converter = self._init_converter_from_study(local_study_w_areas)
        area_components = converter._convert_area_to_component_list(lib_id, {})
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

    def test_convert_renewables_to_component(
        self, local_study_with_renewable: Study, lib_id: str
    ):
        converter = self._init_converter_from_study(local_study_with_renewable)
        study_path = converter.study_path
        (
            renewables_components,
            renewable_connections,
        ) = converter._convert_renewable_to_component_list(lib_id, {}, {})

        timeseries_path = str(
            study_path
            / "input"
            / "renewables"
            / "series"
            / "fr"
            / "generation"
            / "series"
        )
        expected_renewable_connections = [
            InputPortConnections(
                component1="generation",
                port1="balance_port",
                component2="fr",
                port2="balance_port",
            )
        ]
        expected_renewable_component = [
            InputComponent(
                id="generation",
                model="antares-historic.renewable",
                scenario_group=None,
                parameters=[
                    InputComponentParameter(
                        id="unit_count",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                    InputComponentParameter(
                        id="p_max_unit",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    InputComponentParameter(
                        id="generation",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{timeseries_path}",
                    ),
                ],
            )
        ]
        assert renewables_components == expected_renewable_component
        assert renewable_connections == expected_renewable_connections

    def test_convert_st_storages_to_component(
        self, local_study_with_st_storage, lib_id: str
    ):
        converter = self._init_converter_from_study(local_study_with_st_storage)
        study_path = converter.study_path
        (
            storage_components,
            storage_connections,
        ) = converter._convert_st_storage_to_component_list(lib_id, {}, {})

        default_path = study_path / "input" / "st-storage" / "series" / "fr" / "battery"
        inflows_path = default_path / "inflows"
        lower_rule_curve_path = default_path / "lower-rule-curve"
        pmax_injection_path = default_path / "PMAX-injection"
        pmax_withdrawal_path = default_path / "PMAX-withdrawal"
        upper_rule_curve_path = default_path / "upper-rule-curve"
        expected_storage_connections = [
            InputPortConnections(
                component1="battery",
                port1="injection_port",
                component2="fr",
                port2="balance_port",
            )
        ]
        expected_storage_component = [
            InputComponent(
                id="battery",
                model=f"{lib_id}.short-term-storage",
                scenario_group=None,
                parameters=[
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
                        id="initial_level",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.5,
                    ),
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
                        id="inflows",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{inflows_path}",
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
        lib_id: str,
    ):
        converter = self._init_converter_from_study(local_study_w_thermal)
        study_path = converter.study_path

        (
            thermals_components,
            thermals_connections,
        ) = converter._convert_thermal_to_component_list(lib_id, {}, {})

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
        assert thermals_components == expected_thermals_components
        assert thermals_connections == expected_thermals_connections

    def test_convert_solar_to_component(
        self, local_study_w_areas: Study, fr_solar: None, lib_id: str
    ):
        converter = self._init_converter_from_study(local_study_w_areas)

        solar_components, solar_connection = converter._convert_solar_to_component_list(
            lib_id, {}, {}
        )

        solar_timeseries = str(
            converter.study_path / "input" / "solar" / "series" / "solar_fr"
        )
        expected_solar_connection = [
            InputPortConnections(
                component1="solar",
                port1="balance_port",
                component2="fr",
                port2="balance_port",
            )
        ]
        expected_solar_components = InputComponent(
            id="fr",
            model="antares-historic.solar",
            scenario_group=None,
            parameters=[
                InputComponentParameter(
                    id="solar",
                    time_dependent=True,
                    scenario_dependent=True,
                    scenario_group=None,
                    value=f"{solar_timeseries}",
                ),
            ],
        )

        assert solar_components[0] == expected_solar_components
        assert solar_connection == expected_solar_connection

    def test_convert_load_to_component(
        self, local_study_w_areas: Study, fr_load: None, lib_id: str
    ):
        converter = self._init_converter_from_study(local_study_w_areas)

        load_components, load_connection = converter._convert_load_to_component_list(
            lib_id, {}, {}
        )

        load_timeseries = str(
            converter.study_path / "input" / "load" / "series" / "load_fr"
        )
        expected_load_connection = [
            InputPortConnections(
                component1="load",
                port1="balance_port",
                component2="fr",
                port2="balance_port",
            )
        ]
        expected_load_components = InputComponent(
            id="load",
            model="antares-historic.load",
            scenario_group=None,
            parameters=[
                InputComponentParameter(
                    id="load",
                    time_dependent=True,
                    scenario_dependent=True,
                    scenario_group=None,
                    value=f"{load_timeseries}",
                ),
            ],
        )

        assert load_components[0] == expected_load_components
        assert load_connection == expected_load_connection

    @pytest.mark.parametrize(
        "fr_wind",
        [
            [1, 1, 1],  # Dataframe filled with 1
        ],
        indirect=True,
    )
    def test_convert_wind_to_component_not_empty_file(
        self, local_study_w_areas: Study, fr_wind: int, lib_id: str
    ):
        converter = self._init_converter_from_study(local_study_w_areas)

        wind_components, wind_connection = converter._convert_wind_to_component_list(
            lib_id, {}, {}
        )

        wind_timeseries = str(
            converter.study_path / "input" / "wind" / "series" / "wind_fr"
        )
        expected_wind_connection = [
            InputPortConnections(
                component1="wind",
                port1="balance_port",
                component2="fr",
                port2="balance_port",
            )
        ]
        expected_wind_components = InputComponent(
            id="fr",
            model="antares-historic.wind",
            scenario_group=None,
            parameters=[
                InputComponentParameter(
                    id="wind",
                    time_dependent=True,
                    scenario_dependent=True,
                    scenario_group=None,
                    value=f"{wind_timeseries}",
                ),
            ],
        )

        assert wind_components[0] == expected_wind_components
        assert wind_connection == expected_wind_connection

    @pytest.mark.parametrize(
        "fr_wind",
        [
            [],  # DataFrame empty
        ],
        indirect=True,
    )
    def test_convert_wind_to_component_empty_file(
        self, local_study_w_areas: Study, fr_wind: object, lib_id: str
    ):
        converter = self._init_converter_from_study(local_study_w_areas)

        wind_components, _ = converter._convert_wind_to_component_list(lib_id, {}, {})

        assert wind_components == []

    @pytest.mark.parametrize(
        "fr_wind",
        [
            [0, 0, 0],  # DataFrame full of 0
        ],
        indirect=True,
    )
    def test_convert_wind_to_component_zero_values(
        self, local_study_w_areas: Study, fr_wind: int, lib_id: str
    ):
        converter = self._init_converter_from_study(local_study_w_areas)

        wind_components, _ = converter._convert_wind_to_component_list(lib_id, {}, {})

        assert wind_components == []

    def test_convert_links_to_component(self, local_study_w_links: Study, lib_id: str):
        converter = self._init_converter_from_study(local_study_w_links)
        study_path = converter.study_path
        (
            links_components,
            links_connections,
        ) = converter._convert_link_to_component_list(lib_id, {}, {})

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

        assert sorted(links_components, key=lambda x: x.id) == sorted(
            expected_link_component, key=lambda x: x.id
        )
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

    def test_convert_binding_constraints_to_component(self, lib_id: str):
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
            / "battery.yaml"
        )

        bc_data = read_yaml_file(path_cc).get("template", {})
        legacy_objects_for_bc: dict = (
            converter._extract_legacy_objects_from_model_config(bc_data)
        )
        valid_areas: dict = converter._extract_valid_areas_from_model_config(bc_data)

        (
            binding_components,
            binding_connections,
        ) = converter._convert_cc_to_component_list(
            lib_id, legacy_objects_for_bc, valid_areas
        )

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
        assert sorted(expected_data["components"], key=lambda x: x["id"]) == sorted(
            obtained_components, key=lambda x: x["id"]
        )

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
