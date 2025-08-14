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
from typing import Any, Optional, Union, Callable

from antares.craft.model.renewable import RenewableCluster
from antares.craft.model.st_storage import STStorage
from antares.craft.model.study import Study, read_study_local
from antares.craft.model.thermal import ThermalCluster

from gems.input_converter.src.data_preprocessing.preprocessing import (
    ModelsConfigurationPreprocessing,
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

RESOURCES_FOLDER = (
    Path(__file__).parents[1]
    / "data"
    / "model_configuration"
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
        self.legacy_objects_for_bc = {}

    def _match_area_pattern(self, object: Any, param_value: str, model_area_pattern: str = "${area}") -> Any:
        if isinstance(object, dict):
            return {
                self._match_area_pattern(k, param_value, model_area_pattern): self._match_area_pattern(
                    v, param_value, model_area_pattern
                )
                for k, v in object.items()
            }
        elif isinstance(object, list):
            return [self._match_area_pattern(elem, param_value, model_area_pattern) for elem in object]
        elif isinstance(object, str):
            return object.replace(model_area_pattern, param_value)
        else:
            return object

    def _legacy_component_to_exclude(
        self, legacy_objects_for_bc: dict, component_type: str, model_area_pattern: str = "${area}"
    ) -> list:
        """This function aim at finding components that are only present for binding constraint model purpose
        and should be removed from other conversions"""

        components = legacy_objects_for_bc.get(component_type, [])
        return [
            item
            for area in self.areas.values()
            for item in self._match_area_pattern(components, area.id, model_area_pattern)  # type: ignore
        ]

    def _safe_extract_objects_to_delete(self, new_model_data: dict) -> dict:
        """This function aim at extracting legacy components."""
        legacy = new_model_data.get("legacy-objects-to-delete", {})
        # TODO on doit rajouter tous les objets legacy dans le dictionnaire (wind, solar, etc.)
        return {
            "binding_constraints": legacy.get("binding_constraints", []),
            "links": legacy.get("links", []),
            "nodes": legacy.get("nodes", []),
            "thermals": legacy.get("thermal_clusters", []),
        } if legacy else {}

    def _convert_area_to_component_list(
        self, lib_id: str, list_valid_areas: list[MappingProxyType[str, Any]] = []
    ) -> list[InputComponent]:
        components = []
        self.logger.info("Converting areas to component list...")
        for area in self.areas.values():
            if area.id not in list_valid_areas:
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
        self, lib_id: str, valid_areas: dict
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
        self, lib_id: str, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting thermals to component list...")

        thermals_to_exclude: list = self._legacy_component_to_exclude(self.legacy_objects_for_bc, component_type="thermals"
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
        self, lib_id: str, valid_areas: dict
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
        self, lib_id: str, valid_areas: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting links to component list...")

        links_to_exclude: list = self._legacy_component_to_exclude(
            self.legacy_objects_for_bc, component_type="links"
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
        self, lib_id: str, valid_areas: dict
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
        self, lib_id: str, valid_areas: dict
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
        self, lib_id: str, valid_areas: dict
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

    def _iterate_through_model(self, valid_resources: dict, components: list, connections: list, mp: ModelsConfigurationPreprocessing):
        components.append(
            InputComponent(
                id=valid_resources["component"]["id"],
                model=valid_resources["model"],
                parameters=[
                    InputComponentParameter(
                        id=str(param.get("id")),
                        time_dependent=bool(param.get("time-dependent")),
                        scenario_dependent=bool(
                            param.get("scenario-dependent")
                        ),
                        value=mp.convert_param_value(
                            param["id"], param["value"]
                        ),
                    )
                    for param in valid_resources["component"]["parameters"]
                ],
            )
        )
        connections.append(
            InputPortConnections(
                component1=valid_resources["connections"][0]["component1"],
                port1=valid_resources["connections"][0]["port1"],
                component2=valid_resources["connections"][0]["component2"],
                port2=valid_resources["connections"][0]["port2"],
            )
        )

        objects_to_delete: dict = self._safe_extract_objects_to_delete(
        valid_resources
        )
        if objects_to_delete:
            pass # TODO We need to delete those objects
    
    def _convert_model_to_component_list(
        self, valid_areas: dict, resource_content: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting models to component list...")
        
        model_area_pattern = f"${{{resource_content['template-parameters'][0]['name']}}}"
        mp = ModelsConfigurationPreprocessing(self.study)
        try:
            if resource_content["name"] in ["link"]:
                valid_resources: dict = self._check_resources_not_excluded(resource_content, "link")
                for link in valid_resources.values():
                    data_with_link: dict = self._match_area_pattern(resource_content, link.id, model_area_pattern)

                    self._iterate_through_model(data_with_link, components, connections, mp, link.id)

            else:
                for area in valid_areas.values():
                    data_with_area: dict = self._match_area_pattern(resource_content, area.id, model_area_pattern)
                    
                    if resource_content["name"] in ["wind", "solar", "load", "renewables"]:
                        if any(not mp.check_timeseries_validity(param["value"])
                        for param in data_with_area["component"]["parameters"]):
                            continue

                    self._iterate_through_model(data_with_area, components, connections, mp)
                    # Sortir le convert param value, ou bien executer une mthode préalable pour detecter que la TS
                    # Est vide mp.detect_ts_validity_and_existence et si ce nest pas le cas, on append pas le
                    # composant => pour ["wind", "solar", "load", "renewables"]

        except (KeyError, FileNotFoundError) as e:
            self.logger.error(
                f"Error while converting model to component list: {e}. "
                "Please check the model configuration file."
            )
            return components, connections

        return components, connections

    def _check_resources_not_excluded(self, resource_content: dict, parameter: str) -> dict:
        """
        Args:
            resource_content: Dictionary with template parameters.
            parameter:  ("area", "link", "thermal", "renewable").

        """
        RESOURCE_MAP: dict[str, Callable[[], dict]] = {
            "area": lambda self: self.areas,
            "link": lambda self: self.study.get_links(),
            "thermal": lambda self: {
                thermal_id: thermal
                for _, area in self.areas.items()
                for thermal_id, thermal in area.get_thermals().items()
            },
            "renewable": lambda self: {
                renewable_id: renewable
                for _, area in self.areas.items()
                for renewable_id, renewable in area.get_renewables().items()
            }
        }
        if parameter not in RESOURCE_MAP:
            raise ValueError(f"Unsupported parameter: {parameter}")

        excluded = []
        for param in resource_content.get("template-parameters", []):
            if param.get("name") == parameter:
                excluded = param.get("exclude", [])
                break

        items = RESOURCE_MAP[parameter](self)

        return {
            k: v
            for k, v in items.items()
            if k not in excluded
        }

    def convert_study_to_input_study(self) -> InputSystem:
        antares_historic_lib_id = "antares-historic"

        list_components: list[InputComponent] = []
        list_connections: list[InputPortConnections] = []
        list_valid_areas = set()
        for file in RESOURCES_FOLDER.iterdir():
            if file.is_file() and file.name.endswith(".yaml"):
                resource_content = read_yaml_file(file).get("template", {})

                valid_areas: dict = self._check_resources_not_excluded(resource_content, "area")
                
                components, connections = self._convert_model_to_component_list(
                valid_areas, resource_content
                )
                list_components.extend(components)
                list_connections.extend(connections)

                list_valid_areas.update(valid_areas.keys())

        area_components = self._convert_area_to_component_list(
            antares_historic_lib_id, list_valid_areas
        )
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
