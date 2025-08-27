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
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, Union

from antares.craft.model.renewable import RenewableCluster
from antares.craft.model.st_storage import STStorage
from antares.craft.model.study import Study, read_study_local
from antares.craft.model.thermal import ThermalCluster

from gems.input_converter.src.data_preprocessing.binding_constraints import (
    BindingConstraintsPreprocessing,
)
from gems.input_converter.src.data_preprocessing.thermal import ThermalDataPreprocessing
from gems.input_converter.src.utils import (
    check_dataframe_validity,
    read_yaml_file,
    resolve_path,
    transform_to_yaml,
)
from gems.study.parsing import (
    InputComponent,
    InputComponentParameter,
    InputPortConnections,
    InputSystem,
)

BC_FILENAME = "battery.yaml"
BC_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "model_configuration"
    / BC_FILENAME
)


class AntaresStudyConverter:
    def __init__(
        self,
        study_input: Union[Path, Study],
        logger: logging.Logger,
        output_path: Optional[Path] = None,
        period: Optional[int] = None,
    ):
        """
        Initialize processor
        """
        self.logger = logger
        self.period: int = period if period else 168

        if isinstance(study_input, Study):
            self.study = study_input
            self.study_path = study_input.service.config.study_path  # type: ignore
        elif isinstance(study_input, Path):
            self.study_path = resolve_path(study_input)
            self.study = read_study_local(self.study_path)
        else:
            raise TypeError("Invalid input type")
        self.output_path = (
            Path(output_path) if output_path else self.study_path / Path("output.yaml")
        )
        self.areas: MappingProxyType = self.study.get_areas()
        self.bc_area_pattern: str = "${area}"

    def _match_area_pattern(self, object: Any, param_values: str) -> Any:
        if isinstance(object, dict):
            return {
                self._match_area_pattern(k, param_values): self._match_area_pattern(
                    v, param_values
                )
                for k, v in object.items()
            }
        elif isinstance(object, list):
            return [self._match_area_pattern(elem, param_values) for elem in object]
        elif isinstance(object, str):
            return object.replace(self.bc_area_pattern, param_values)
        else:
            return object

    def _legacy_component_to_exclude(
        self, legacy_objects_for_bc: dict, component_type: str
    ) -> list:
        """This function aim at finding components that are only present for binding constraint model purpose
        and should be removed from other conversions"""

        components = legacy_objects_for_bc.get(component_type, [])
        return [
            item
            for area in self.areas.values()
            for item in self._match_area_pattern(components, area.id)  # type: ignore
        ]

    def _extract_legacy_objects_from_model_config(self, bc_data: dict) -> dict:
        """This function aim at extracting components that are only present for binding constraint model."""
        legacy = bc_data.get("legacy-objects-to-delete", {})
        return {
            "binding_constraints": legacy.get("binding_constraints", []),
            "links": legacy.get("links", []),
            "nodes": legacy.get("nodes", []),
            "thermals": legacy.get("thermal_clusters", []),
        }

    def _convert_area_to_component_list(
        self, lib_id: str, legacy_objects_for_bc: dict
    ) -> list[InputComponent]:
        components = []
        self.logger.info("Converting areas to component list...")

        for area in self.areas.values():
            if area.id in legacy_objects_for_bc.get("nodes", []):
                continue
            components.append(
                InputComponent(
                    id=area.id,
                    model=f"{lib_id}.area",
                    parameters=[
                        InputComponentParameter(
                            id="ens_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            value=area.properties.energy_cost_unsupplied,
                        ),
                        InputComponentParameter(
                            id="spillage_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            value=area.properties.energy_cost_spilled,
                        ),
                    ],
                )
            )
        return components

    def _convert_renewable_to_component_list(
        self, lib_id: str, legacy_objects_for_bc: dict, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting renewables to component list...")
        for area in self.areas.values():
            renewables: dict[str, RenewableCluster] = area.get_renewables()
            for renewable in renewables.values():
                series_path = (
                    self.study_path
                    / "input"
                    / "renewables"
                    / "series"
                    / Path(renewable.area_id)
                    / Path(renewable.id)
                    / "series.txt"
                )
                components.append(
                    InputComponent(
                        id=renewable.id,
                        model=f"{lib_id}.renewable",
                        parameters=[
                            InputComponentParameter(
                                id="unit_count",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=renewable.properties.unit_count,
                            ),
                            InputComponentParameter(
                                id="p_max_unit",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=renewable.properties.nominal_capacity,
                            ),
                            InputComponentParameter(
                                id="generation",
                                time_dependent=True,
                                scenario_dependent=True,
                                value=str(series_path).removesuffix(".txt"),
                            ),
                        ],
                    )
                )
                connections.append(
                    InputPortConnections(
                        component1=renewable.id,
                        port1="balance_port",
                        component2=renewable.area_id,
                        port2="balance_port",
                    )
                )

        return components, connections

    def _convert_thermal_to_component_list(
        self, lib_id: str, legacy_objects_for_bc: dict, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting thermals to component list...")

        thermals_to_exclude: list = self._legacy_component_to_exclude(
            legacy_objects_for_bc, component_type="thermals"
        )

        # Add thermal components for each area
        for area in self.areas.values():
            thermals: dict[str, ThermalCluster] = area.get_thermals()
            for thermal in thermals.values():
                if f"{area.id}.{thermal.id}" in thermals_to_exclude:
                    continue

                series_path = (
                    self.study_path
                    / "input"
                    / "thermal"
                    / "series"
                    / Path(thermal.area_id)
                    / Path(thermal.id)
                    / "series.txt"
                )
                tdp = ThermalDataPreprocessing(thermal, self.study_path)
                components.append(
                    InputComponent(
                        id=thermal.id,
                        model=f"{lib_id}.thermal",
                        parameters=[
                            tdp.generate_component_parameter("p_min_cluster"),
                            tdp.generate_component_parameter("nb_units_min"),
                            tdp.generate_component_parameter("nb_units_max"),
                            tdp.generate_component_parameter(
                                "nb_units_max_variation_forward", self.period
                            ),
                            tdp.generate_component_parameter(
                                "nb_units_max_variation_backward", self.period
                            ),
                            InputComponentParameter(
                                id="unit_count",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=thermal.properties.unit_count,
                            ),
                            InputComponentParameter(
                                id="p_min_unit",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=thermal.properties.min_stable_power,
                            ),
                            InputComponentParameter(
                                id="efficiency",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=thermal.properties.efficiency,
                            ),
                            InputComponentParameter(
                                id="p_max_unit",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=thermal.properties.nominal_capacity,
                            ),
                            InputComponentParameter(
                                id="generation_cost",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=thermal.properties.marginal_cost,
                            ),
                            InputComponentParameter(
                                id="fixed_cost",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=thermal.properties.fixed_cost,
                            ),
                            InputComponentParameter(
                                id="startup_cost",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=thermal.properties.startup_cost,
                            ),
                            InputComponentParameter(
                                id="d_min_up",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=thermal.properties.min_up_time,
                            ),
                            InputComponentParameter(
                                id="d_min_down",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=thermal.properties.min_down_time,
                            ),
                            InputComponentParameter(
                                id="p_max_cluster",
                                time_dependent=True,
                                scenario_dependent=True,
                                value=str(series_path).removesuffix(".txt"),
                            ),
                        ],
                    )
                )

                connections.append(
                    InputPortConnections(
                        component1=thermal.id,
                        port1="balance_port",
                        component2=area.id,
                        port2="balance_port",
                    )
                )
        return components, connections

    def _convert_st_storage_to_component_list(
        self, lib_id: str, legacy_objects_for_bc: dict, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting short-term storages to component list...")
        # Add thermal components for each area
        for area in self.areas.values():
            storages: dict[str, STStorage] = area.get_st_storages()
            for storage in storages.values():
                series_path = (
                    self.study_path
                    / "input"
                    / "st-storage"
                    / "series"
                    / Path(storage.area_id)
                    / Path(storage.id)
                )
                inflows_path = series_path / "inflows"
                lower_rule_curve_path = series_path / "lower-rule-curve"
                pmax_injection_path = series_path / "PMAX-injection"
                pmax_withdrawal_path = series_path / "PMAX-withdrawal"
                upper_rule_curve_path = series_path / "upper-rule-curve"
                components.append(
                    InputComponent(
                        id=storage.id,
                        model=f"{lib_id}.short-term-storage",
                        parameters=[
                            InputComponentParameter(
                                id="efficiency_injection",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=storage.properties.efficiency,
                            ),
                            # TODO wait for update of antares craft that support the 9.2 version of Antares
                            InputComponentParameter(
                                id="efficiency_withdrawal",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=1,
                                # value=storage.properties.efficiencywithdrawal,
                            ),
                            InputComponentParameter(
                                id="initial_level",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=storage.properties.initial_level,
                            ),
                            InputComponentParameter(
                                id="reservoir_capacity",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=storage.properties.reservoir_capacity,
                            ),
                            InputComponentParameter(
                                id="injection_nominal_capacity",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=storage.properties.injection_nominal_capacity,
                            ),
                            InputComponentParameter(
                                id="withdrawal_nominal_capacity",
                                time_dependent=False,
                                scenario_dependent=False,
                                value=storage.properties.withdrawal_nominal_capacity,
                            ),
                            InputComponentParameter(
                                id="inflows",
                                time_dependent=True,
                                scenario_dependent=True,
                                value=str(inflows_path),
                            ),
                            InputComponentParameter(
                                id="lower_rule_curve",
                                time_dependent=True,
                                scenario_dependent=True,
                                value=str(lower_rule_curve_path),
                            ),
                            InputComponentParameter(
                                id="upper_rule_curve",
                                time_dependent=True,
                                scenario_dependent=True,
                                value=str(upper_rule_curve_path),
                            ),
                            InputComponentParameter(
                                id="p_max_injection_modulation",
                                time_dependent=True,
                                scenario_dependent=True,
                                value=str(pmax_injection_path),
                            ),
                            InputComponentParameter(
                                id="p_max_withdrawal_modulation",
                                time_dependent=True,
                                scenario_dependent=True,
                                value=str(pmax_withdrawal_path),
                            ),
                        ],
                    )
                )

                connections.append(
                    InputPortConnections(
                        component1=storage.id,
                        port1="injection_port",
                        component2=area.id,
                        port2="balance_port",
                    )
                )
        return components, connections

    def _convert_link_to_component_list(
        self, lib_id: str, legacy_objects_for_bc: dict, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting links to component list...")

        links_to_exclude: list = self._legacy_component_to_exclude(
            legacy_objects_for_bc, component_type="links"
        )

        # Add links components for each area
        links = self.study.get_links()
        for link in links.values():
            if f"{link.area_from_id}%{link.area_to_id}" in links_to_exclude:
                continue
            capacity_direct_path = (
                self.study_path
                / "input"
                / "links"
                / Path(link.area_from_id)
                / "capacities"
                / f"{link.area_to_id}_direct"
            )
            capacity_indirect_path = (
                self.study_path
                / "input"
                / "links"
                / Path(link.area_from_id)
                / "capacities"
                / f"{link.area_to_id}_indirect"
            )
            components.append(
                InputComponent(
                    id=link.id,
                    model=f"{lib_id}.link",
                    parameters=[
                        InputComponentParameter(
                            id="capacity_direct",
                            time_dependent=True,
                            scenario_dependent=True,
                            value=str(capacity_direct_path),
                        ),
                        InputComponentParameter(
                            id="capacity_indirect",
                            time_dependent=True,
                            scenario_dependent=True,
                            value=str(capacity_indirect_path),
                        ),
                    ],
                )
            )
            connections.append(
                InputPortConnections(
                    component1=link.id,
                    port1="in_port",
                    component2=link.area_from_id,
                    port2="balance_port",
                )
            )
            connections.append(
                InputPortConnections(
                    component1=link.id,
                    port1="out_port",
                    component2=link.area_to_id,
                    port2="balance_port",
                ),
            )
        return components, connections

    def _convert_wind_to_component_list(
        self, lib_id: str, legacy_objects_for_bc: dict, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting wind to component list...")
        for area in self.areas.values():
            series_path = (
                self.study_path / "input" / "wind" / "series" / f"wind_{area.id}.txt"
            )
            if series_path.exists():
                if check_dataframe_validity(area.get_wind_matrix()):
                    components.append(
                        InputComponent(
                            id=area.id,
                            model=f"{lib_id}.wind",
                            parameters=[
                                InputComponentParameter(
                                    id="wind",
                                    time_dependent=True,
                                    scenario_dependent=True,
                                    value=str(series_path).removesuffix(".txt"),
                                )
                            ],
                        )
                    )
                    connections.append(
                        InputPortConnections(
                            component1="wind",
                            port1="balance_port",
                            component2=area.id,
                            port2="balance_port",
                        )
                    )

        return components, connections

    def _convert_solar_to_component_list(
        self, lib_id: str, legacy_objects_for_bc: dict, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting solar to component list...")
        for area in self.areas.values():
            series_path = (
                self.study_path / "input" / "solar" / "series" / f"solar_{area.id}.txt"
            )

            if series_path.exists():
                if check_dataframe_validity(area.get_solar_matrix()):
                    components.append(
                        InputComponent(
                            id=area.id,
                            model=f"{lib_id}.solar",
                            parameters=[
                                InputComponentParameter(
                                    id="solar",
                                    time_dependent=True,
                                    scenario_dependent=True,
                                    value=str(series_path).removesuffix(".txt"),
                                )
                            ],
                        )
                    )
                    connections.append(
                        InputPortConnections(
                            component1="solar",
                            port1="balance_port",
                            component2=area.id,
                            port2="balance_port",
                        )
                    )

        return components, connections

    def _convert_load_to_component_list(
        self, lib_id: str, legacy_objects_for_bc: dict, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting load to component list...")
        for area in self.areas.values():
            series_path = (
                self.study_path / "input" / "load" / "series" / f"load_{area.id}.txt"
            )
            if series_path.exists():
                if check_dataframe_validity(area.get_load_matrix()):
                    components.append(
                        InputComponent(
                            id="load",
                            model=f"{lib_id}.load",
                            parameters=[
                                InputComponentParameter(
                                    id="load",
                                    time_dependent=True,
                                    scenario_dependent=True,
                                    value=str(series_path).removesuffix(".txt"),
                                )
                            ],
                        )
                    )
                    connections.append(
                        InputPortConnections(
                            component1="load",
                            port1="balance_port",
                            component2=area.id,
                            port2="balance_port",
                        )
                    )

        return components, connections

    def _convert_cc_to_component_list(
        self, lib_id: str, legacy_objects_for_bc: dict, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting binding constraints to component list...")

        bc_data = read_yaml_file(BC_CONFIG_PATH).get("template")
        try:
            for area in valid_areas.values():
                data_with_area: dict = self._match_area_pattern(bc_data, area.id)
                bcp = BindingConstraintsPreprocessing(self.study)

                components.append(
                    InputComponent(
                        id=data_with_area["component"]["id"],
                        model=data_with_area["model"],
                        parameters=[
                            InputComponentParameter(
                                id=str(param.get("id")),
                                time_dependent=bool(param.get("time-dependent")),
                                scenario_dependent=bool(
                                    param.get("scenario-dependent")
                                ),
                                value=bcp.convert_param_value(
                                    param.get("id"), param.get("value")
                                ),
                            )
                            for param in data_with_area["component"]["parameters"]
                        ],
                    )
                )
                connections.append(
                    InputPortConnections(
                        component1=data_with_area["component"]["id"],
                        port1="injection_port",
                        component2=area.id,
                        port2="balance_port",
                    )
                )
        except (KeyError, FileNotFoundError) as e:
            return components, connections

        return components, connections

    def _extract_valid_areas_from_model_config(self, bc_data: dict) -> dict:
        for template_param in bc_data["template-parameters"]:
            if template_param.get("exclude"):
                return {
                    k: v
                    for k, v in self.areas.items()
                    if k not in template_param["exclude"]
                }
        return {}

    def convert_study_to_input_study(self) -> InputSystem:
        antares_historic_lib_id = "antares-historic"
        bc_data = read_yaml_file(BC_CONFIG_PATH).get("template", {})
        # Get area pattern for binding constraint from model config
        self.bc_area_pattern = f"${{{bc_data['template-parameters'][0]['name']}}}"

        legacy_objects_for_bc: dict = self._extract_legacy_objects_from_model_config(
            bc_data
        )
        valid_areas = self._extract_valid_areas_from_model_config(bc_data)
        area_components = self._convert_area_to_component_list(
            antares_historic_lib_id, legacy_objects_for_bc
        )

        list_components: list[InputComponent] = []
        list_connections: list[InputPortConnections] = []

        conversion_methods = [
            self._convert_renewable_to_component_list,
            self._convert_thermal_to_component_list,
            self._convert_st_storage_to_component_list,
            self._convert_load_to_component_list,
            self._convert_wind_to_component_list,
            self._convert_solar_to_component_list,
            self._convert_link_to_component_list,
            self._convert_cc_to_component_list,
        ]

        for method in conversion_methods:
            components, connections = method(
                antares_historic_lib_id, legacy_objects_for_bc, valid_areas
            )
            list_components.extend(components)
            list_connections.extend(connections)

        self.logger.info(
            "Converting node, components and connections into Input study..."
        )
        return InputSystem(
            nodes=area_components,
            components=list_components,
            connections=list_connections,
        )

    def process_all(self) -> None:
        study = self.convert_study_to_input_study()
        self.logger.info("Converting input study into yaml file...")
        transform_to_yaml(model=study, output_path=self.output_path)
