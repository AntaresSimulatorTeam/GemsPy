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
import shutil
from pathlib import Path

import pandas as pd
import pytest
from antares.craft.model.study import Study

from gems.input_converter.src.config import MODEL_NAME_TO_FILE_NAME
from gems.input_converter.src.converter import AntaresStudyConverter
from gems.input_converter.src.logger import Logger
from gems.input_converter.src.parsing import Operation, parse_conversion_template
from gems.input_converter.src.utils import (
    check_file_exists,
    dump_to_yaml,
    read_yaml_file,
)
from gems.model.resolve_library import resolve_library
from gems.study.parsing import (
    InputAreaConnections,
    InputComponent,
    InputComponentParameter,
    InputPortConnections,
    InputSystem,
    parse_yaml_components,
)
from gems.study.resolve_components import resolve_system
from tests.input_converter.conftest import create_dataframe_from_constant

RESOURCES_FOLDER = (
    Path(__file__).parents[2]
    / "src"
    / "gems"
    / "input_converter"
    / "data"
    / "model_configuration"
)
LOCAL_PATH = "mini_test_batterie_BP23"
DATAFRAME_PREPRO_SERIES = (create_dataframe_from_constant(lines=8760),)  # series

DATAFRAME_PREPRO_THERMAL_CONFIG = (
    create_dataframe_from_constant(lines=8760, columns=4),  # modulation
    create_dataframe_from_constant(lines=8760),  # series
)

DATAFRAME_PREPRO_BC_CONFIG = (
    create_dataframe_from_constant(lines=8760, columns=6),  # modulation
    create_dataframe_from_constant(lines=8760, columns=4),  # series
)
LIB_PATHS = [
    "src/gems/libs/antares_historic/antares_historic.yml",
    "src/gems/libs/reference_models/andromede_v1_models.yml",
]
MODEL_LIST_WITH_BASE = [str(Path(os.getcwd()) / suffix) for suffix in LIB_PATHS]


class TestConverter:
    def _init_converter_from_study(
        self,
        local_study,
        model_list: list[str] = list(MODEL_NAME_TO_FILE_NAME.keys()),
        mode: str = "full",
    ):
        logger = Logger(__name__, local_study.path)
        converter: AntaresStudyConverter = AntaresStudyConverter(
            study_input=local_study,
            logger=logger,
            mode=mode,
            lib_paths=LIB_PATHS,
            models_to_convert=model_list,
            output_folder=local_study.path.parent / "converter_output",
        )
        return converter

    def _init_converter_from_path(
        self,
        input_path: Path,
        output_path: Path,
        mode: str = "full",
        lib_paths: list = None,
        model_list: list = list(MODEL_NAME_TO_FILE_NAME.keys()),
    ):
        logger = Logger(__name__, str(input_path))
        converter: AntaresStudyConverter = AntaresStudyConverter(
            study_input=input_path,
            logger=logger,
            mode=mode,
            output_folder=output_path,
            lib_paths=lib_paths,
            models_to_convert=model_list,
        )
        return converter

    def test_convert_study_to_input_study(self, local_study_w_areas: Study):
        converter = self._init_converter_from_study(local_study_w_areas, model_list=[])
        input_study = converter.convert_study_to_input_system()

        expected_input_study = InputSystem(
            id="studyTest",
            components=[
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
        )
        assert input_study == expected_input_study

    def test_convert_area_to_component(self, local_study_w_areas: Study, lib_id: str):
        converter = self._init_converter_from_study(local_study_w_areas, model_list=[])
        area_components = converter._convert_area_to_component_list(lib_id)

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
        converter = self._init_converter_from_study(local_study_w_areas, model_list=[])
        area_components = converter._convert_area_to_component_list(lib_id)
        input_study = InputSystem(id=converter.study.name, components=area_components)

        # Dump model into yaml file
        yaml_path = converter.output_folder / "study_path.yaml"
        dump_to_yaml(model=input_study, output_path=yaml_path)
        # Open yaml file to validate
        with open(yaml_path, "r", encoding="utf-8") as yaml_file:
            validated_data = parse_yaml_components(yaml_file)

        expected_validated_data = InputSystem(
            id="studyTest",
            components=[
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
        )

        expected_validated_data.components.sort(key=lambda x: x.id)
        validated_data.components.sort(key=lambda x: x.id)
        assert validated_data == expected_validated_data

    def test_convert_st_storages_to_component(
        self, local_study_with_st_storage: Study, lib_id: str
    ):
        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_study(
            local_study_with_st_storage, model_list=[]
        )
        path_load = RESOURCES_FOLDER / "st-storage.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            storage_components,
            storage_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)

        inflows_path = "inflows_fr_storage_1"
        lower_rule_curve_path = "lower_rule_curve_fr_storage_1"
        pmax_injection_path = "p_max_injection_modulation_fr_storage_1"
        pmax_withdrawal_path = "p_max_withdrawal_modulation_fr_storage_1"
        upper_rule_curve_path = "upper_rule_curve_fr_storage_1"
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
        converter = self._init_converter_from_study(
            local_study_w_thermal, model_list=[]
        )
        path_load = RESOURCES_FOLDER / "thermal.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            thermals_components,
            thermals_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        # study_path = converter.output_folder
        # series_path = study_path / "input" / "thermal" / "series" / "fr" / "gaz"
        print(thermals_components)
        expected_thermals_connections = [
            InputPortConnections(
                component1="fr_gaz",
                port1="balance_port",
                component2="fr",
                port2="balance_port",
            )
        ]
        expected_thermals_components = [
            InputComponent(
                id="fr_gaz",
                model="antares-historic.thermal",
                scenario_group=None,
                parameters=[
                    InputComponentParameter(
                        id="p_min_cluster",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value="fr_gaz_p_min_cluster",
                    ),
                    InputComponentParameter(
                        id="nb_units_min",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value="fr_gaz_nb_units_min",
                    ),
                    InputComponentParameter(
                        id="nb_units_max",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value="fr_gaz_nb_units_max",
                    ),
                    InputComponentParameter(
                        id="nb_units_max_variation_forward",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value="fr_gaz_nb_units_max_variation_forward",
                    ),
                    InputComponentParameter(
                        id="nb_units_max_variation_backward",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value="fr_gaz_nb_units_max_variation_backward",
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
                        value="fr_gaz_p_max_cluster",
                    ),
                ],
            )
        ]
        # TODO preprocessing + nouveaux parametres liées a la nouvelle version antarescraft
        assert thermals_components == expected_thermals_components
        assert thermals_connections == expected_thermals_connections

    def test_convert_load_to_component_from_path(self, tmp_path: Path):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH

        output_path = local_path / "reference.yaml"
        expected_data = read_yaml_file(output_path)["system"]
        expected_components = expected_data["components"]
        expected_connections = expected_data["connections"]

        input_path = tmp_path / "input" / LOCAL_PATH
        output_path = tmp_path / "output" / LOCAL_PATH
        shutil.copytree(local_path, input_path)

        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_path(
            input_path, output_path, "full", model_list=[]
        )
        path_load = RESOURCES_FOLDER / "load.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            load_components,
            load_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)

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
        path_to_remove = converter.output_folder / "input" / "data-series"
        obtained_parameters = TestConverter._match_area_pattern(
            obtained_parameters_to_dict, "", str(path_to_remove) + "/"
        )

        assert obtained_parameters == expected_component["parameters"]
        # TODO enrich

    @pytest.mark.parametrize(
        "fr_solar",
        [DATAFRAME_PREPRO_SERIES],
        indirect=True,
    )
    def test_convert_solar_to_component_from_study(self, fr_solar: None):
        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_study(fr_solar, model_list=[])

        path_load = RESOURCES_FOLDER / "solar.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            solar_components,
            solar_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        solar_fr_component = next(
            (comp for comp in solar_components if comp.id == "solar_fr"), None
        )
        solar_fr_connection = next(
            (conn for conn in solar_connections if conn.component1 == "solar_fr"), None
        )
        solar_timeseries = "generation_fr"
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

    def test_convert_load_to_component_from_study(self, fr_load: None):
        converter = self._init_converter_from_study(fr_load)
        path_load = RESOURCES_FOLDER / "load.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            load_components,
            load_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        load_fr_component = next(
            (comp for comp in load_components if comp.id == "load_fr"), None
        )
        load_fr_connection = next(
            (conn for conn in load_connections if conn.component1 == "load_fr"), None
        )

        load_timeseries = "load_fr"
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
        converter = self._init_converter_from_study(fr_wind, model_list=[])

        path_load = RESOURCES_FOLDER / "wind.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            wind_components,
            wind_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        wind_fr_component = next(
            (comp for comp in wind_components if comp.id == "wind_fr"), None
        )
        wind_fr_connection = next(
            (conn for conn in wind_connections if conn.component1 == "wind_fr"), None
        )

        expected_wind_connection = InputPortConnections(
            component1="wind_fr",
            port1="balance_port",
            component2="fr",
            port2="balance_port",
        )
        expected_wind_components = InputComponent(
            id="wind_fr",
            model="antares-historic.renewable",
            scenario_group="wind_group",
            parameters=[
                InputComponentParameter(
                    id="nominal_capacity",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                ),
                InputComponentParameter(
                    id="unit_count",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                ),
                InputComponentParameter(
                    id="generation",
                    time_dependent=True,
                    scenario_dependent=True,
                    value="generation_fr",
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
        converter = self._init_converter_from_study(fr_wind, model_list=[])

        path_load = RESOURCES_FOLDER / "wind.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            wind_components,
            _,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        assert wind_components == []

    @pytest.mark.parametrize(
        "fr_wind",
        [
            pd.DataFrame([0, 0, 0]),  # DataFrame full of 0
        ],
        indirect=True,
    )
    def test_convert_wind_to_component_zero_values(self, fr_wind: int):
        converter = self._init_converter_from_study(fr_wind, model_list=[])

        path_load = RESOURCES_FOLDER / "wind.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            wind_components,
            _,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        assert wind_components == []

    def test_convert_links_to_component(self, local_study_w_links: Study, lib_id: str):
        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_study(local_study_w_links, model_list=[])
        path_load = RESOURCES_FOLDER / "link.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            links_components,
            links_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)

        fr_it_direct_links_timeseries = "capacity_direct_fr_it"
        fr_it_indirect_links_timeseries = "capacity_indirect_fr_it"
        fr_it_direct_costs_timeseries = "hurdle_cost_direct_fr_it"
        fr_it_indirect_costs_timeseries = "hurdle_cost_indirect_fr_it"
        at_fr_direct_links_timeseries = "capacity_direct_at_fr"
        at_fr_indirect_links_timeseries = "capacity_indirect_at_fr"
        at_it_direct_links_timeseries = "capacity_direct_at_it"
        at_it_indirect_links_timeseries = "capacity_indirect_at_it"
        at_fr_direct_costs_timeseries = "hurdle_cost_direct_at_fr"
        at_fr_indirect_costs_timeseries = "hurdle_cost_indirect_at_fr"
        at_it_direct_costs_timeseries = "hurdle_cost_direct_at_it"
        at_it_indirect_costs_timeseries = "hurdle_cost_indirect_at_it"
        expected_link_component = [
            InputComponent(
                id="fr_/_it",
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
                    InputComponentParameter(
                        id="hurdle_cost_direct",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{fr_it_direct_costs_timeseries}",
                    ),
                    InputComponentParameter(
                        id="hurdle_cost_indirect",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{fr_it_indirect_costs_timeseries}",
                    ),
                ],
            ),
            InputComponent(
                id="at_/_fr",
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
                    InputComponentParameter(
                        id="hurdle_cost_direct",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_fr_direct_costs_timeseries}",
                    ),
                    InputComponentParameter(
                        id="hurdle_cost_indirect",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_fr_indirect_costs_timeseries}",
                    ),
                ],
            ),
            InputComponent(
                id="at_/_it",
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
                    InputComponentParameter(
                        id="hurdle_cost_direct",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_it_direct_costs_timeseries}",
                    ),
                    InputComponentParameter(
                        id="hurdle_cost_indirect",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_it_indirect_costs_timeseries}",
                    ),
                ],
            ),
        ]

        expected_link_connections = [
            InputPortConnections(
                component1="at_/_fr",
                port1="in_port",
                component2="at",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="at_/_fr",
                port1="out_port",
                component2="fr",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="at_/_it",
                port1="in_port",
                component2="at",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="at_/_it",
                port1="out_port",
                component2="it",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="fr_/_it",
                port1="in_port",
                component2="fr",
                port2="balance_port",
            ),
            InputPortConnections(
                component1="fr_/_it",
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

    def test_convert_binding_constraints_to_component(
        self, lib_id: str, tmp_path: Path
    ):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH

        output_path = local_path / "reference.yaml"
        expected_data = read_yaml_file(output_path)["system"]

        expected_components = expected_data["components"]
        expected_connections = expected_data["connections"]

        input_path = tmp_path / "input" / LOCAL_PATH
        output_path = tmp_path / "output" / LOCAL_PATH
        shutil.copytree(local_path, input_path)

        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_path(
            input_path, output_path, "full", model_list=[]
        )
        path_cc = RESOURCES_FOLDER / "battery.yaml"

        with path_cc.open() as template:
            bc_data = parse_conversion_template(template)
        (
            binding_components,
            binding_connections,
            area_connections,
        ) = converter._convert_model_to_component_list(
            bc_data, bc_data.get_excluded_objects_ids()
        )  # Bad design, either the test should call a higher level function, or virtual objects should be deduced from single model

        connection = binding_connections[0]
        ### Compare area connections
        assert area_connections == []
        # Compare connections

        expected_connection: InputPortConnections = InputPortConnections(
            **next(
                (
                    connection
                    for connection in expected_connections
                    if connection["component2"] == "fr"
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
            obtained_parameters_to_dict, "", str(converter.output_folder) + "/"
        )
        assert obtained_parameters == expected_component["parameters"]
        # TODO enrich

    def test_hybrid_data_series_presence(self, tmp_path: Path):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH
        lib_paths: list = LIB_PATHS
        model_list: list = "battery"

        input_path = tmp_path / "input" / LOCAL_PATH
        output_path = tmp_path / "output" / LOCAL_PATH
        shutil.copytree(local_path, input_path)

        converter = self._init_converter_from_path(
            input_path, output_path, "hybrid", lib_paths, model_list
        )
        path_cc = (
            Path(__file__).parent.parent.parent
            / "src"
            / "gems"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "battery.yaml"
        )

        with path_cc.open() as template:
            bc_data = parse_conversion_template(template)

        (
            _,
            _,
            area_connections,
        ) = converter._convert_model_to_component_list(bc_data)

        output_path = converter.output_folder
        path1 = (
            output_path / "input" / "data-series" / "marginal_cost_fr_z_batteries.tsv"
        )
        path2 = (
            output_path
            / "input"
            / "data-series"
            / "p_max_injection_modulation_fr_z_batteries.tsv"
        )
        path3 = (
            output_path
            / "input"
            / "data-series"
            / "p_max_withdrawal_modulation_fr_fr_batteries_inj.tsv"
        )
        path4 = (
            output_path
            / "input"
            / "data-series"
            / "upper_rule_curve_z_batteries_z_batteries_batteries_fr_1.tsv"
        )
        assert check_file_exists(path1)
        assert check_file_exists(path2)
        assert check_file_exists(path3)
        assert check_file_exists(path4)
        ### Compare area connections
        expected_area_connections = [
            InputAreaConnections(
                component="battery_fr", port="injection_port", area="fr"
            )
        ]
        assert area_connections == expected_area_connections

    def test_hybrid_convert_study_path_to_input_study(self, tmp_path: Path):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH

        output_path = local_path / "reference_hybrid.yaml"
        expected_data = read_yaml_file(output_path)["system"]

        model_list: list = ["battery"]

        input_path = tmp_path / "input" / LOCAL_PATH
        output_path = tmp_path / "output" / LOCAL_PATH
        shutil.copytree(local_path, input_path)

        converter = self._init_converter_from_path(
            input_path, output_path, "hybrid", MODEL_LIST_WITH_BASE, model_list
        )
        thermal_cluster_filepath = (
            converter.output_folder
            / "input"
            / "thermal"
            / "clusters"
            / "z_batteries"
            / "list.ini"
        )
        bc_filepath = (
            converter.output_folder
            / "input"
            / "bindingconstraints"
            / "bindingconstraints.ini"
        )
        links_filepath = (
            converter.output_folder / "input" / "links" / "fr" / "properties.ini"
        )
        assert thermal_cluster_filepath.stat().st_size > 0
        assert bc_filepath.stat().st_size > 0
        assert links_filepath.stat().st_size > 0
        obtained_data = converter.convert_study_to_input_system()

        # Check files have been correctly deleted
        thermal_cluster_filepath = (
            converter.output_folder
            / "input"
            / "thermal"
            / "clusters"
            / "z_batteries"
            / "list.ini"
        )
        bc_filepath = (
            converter.output_folder
            / "input"
            / "bindingconstraints"
            / "bindingconstraints.ini"
        )
        links_filepath = (
            converter.output_folder / "input" / "links" / "fr" / "properties.ini"
        )
        assert thermal_cluster_filepath.stat().st_size == 0
        assert bc_filepath.stat().st_size == 0
        assert links_filepath.stat().st_size == 0
        # TODO check folder data-models is present

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
            obtained_components_to_dict, "", str(converter.output_folder) + "/"
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

    def test_convert_study_path_to_input_study(self, tmp_path: Path):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH
        ref_path = local_path / "reference.yaml"
        input_path = tmp_path / "input"
        output_path = tmp_path / "output"
        shutil.copytree(local_path, input_path)
        converter = self._init_converter_from_path(
            input_path, output_path, "full", MODEL_LIST_WITH_BASE
        )
        obtained_sys = converter.convert_study_to_input_system()
        with open(ref_path) as system_file:
            expected_sys = parse_yaml_components(system_file)
        assert obtained_sys.components == expected_sys.components

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
