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
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, Union

import pandas as pd
from antares.craft.exceptions.exceptions import ReferencedObjectDeletionNotAllowed
from antares.craft.model.study import Study, read_study_local
from antares.craft.model.thermal import ThermalCluster

from gems.input_converter.src.config import (
    MATRIX_TYPES_TO_SET_METHOD,
    STUDY_LEVEL_DELETION,
    STUDY_LEVEL_GET,
    TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD,
    TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD,
)
from gems.input_converter.src.data_preprocessing.dataclasses import ConversionMode
from gems.input_converter.src.data_preprocessing.preprocessing import (
    ModelsConfigurationProcessing,
)
from gems.input_converter.src.data_preprocessing.thermal import ThermalDataPreprocessing
from gems.input_converter.src.utils import (
    read_yaml_file,
    resolve_path,
    transform_to_yaml,
)
from gems.study.parsing import (
    InputAreaConnections,
    InputComponent,
    InputComponentParameter,
    InputPortConnections,
    InputSystem,
)

from .logger import Logger

RESOURCES_FOLDER = Path(__file__).parents[1] / "data" / "model_configuration"


class AntaresStudyConverter:
    def __init__(
        self,
        study_input: Union[Path, Study],
        logger: Logger,
        mode: ConversionMode = "full",
        output_path: Optional[Path] = None,
        period: Optional[int] = None,
    ):
        """
        Initialize processor
        """
        self.logger = logger
        self.mode = mode
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
        self.legacy_objects = []

    def _convert_thermal_to_component_list(
        self, valid_areas: dict, components: list, connections: list
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        self.logger.info("Converting thermals to component list...")
        lib_id = "antares-historic"

        # Add thermal components for each area
        for area in self.areas.values():
            if area.id not in valid_areas:
                continue
            thermals: dict[str, ThermalCluster] = area.get_thermals()
            for thermal in thermals.values():
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

    def _match_area_pattern(
        self, object: Any, param_value: str, model_area_pattern: str = "${area}"
    ) -> Any:
        if isinstance(object, dict):
            return {
                self._match_area_pattern(
                    k, param_value, model_area_pattern
                ): self._match_area_pattern(v, param_value, model_area_pattern)
                for k, v in object.items()
            }
        elif isinstance(object, list):
            return [
                self._match_area_pattern(elem, param_value, model_area_pattern)
                for elem in object
            ]
        elif isinstance(object, str):
            return object.replace(model_area_pattern, param_value)
        else:
            return object

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

    def _delete_legacy_objects(self) -> dict:
        for item in self.legacy_objects:
            item_type = item.get("type")
            try:
                if item_type in STUDY_LEVEL_DELETION:
                    id = (
                        item["binding-constraint-id"]
                        if item_type == "binding_constraint"
                        else item[item_type]
                    )
                    getattr(self.study, STUDY_LEVEL_DELETION[item_type])(
                        getattr(self.study, STUDY_LEVEL_GET[item_type])()[id]
                    )
                elif item_type in TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD:
                    print(
                        "deleting",
                        TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD[item_type],
                        TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[item_type],
                        [item.get("cluster")],
                    )
                    getattr(
                        self.areas[item.get("area")],
                        TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD[item_type],
                    )(
                        getattr(
                            self.areas[item.get("area")],
                            TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[item_type],
                        )()[item.get("cluster")]
                    )
                elif item_type in MATRIX_TYPES_TO_SET_METHOD:
                    getattr(
                        self.areas[item.get("area")],
                        MATRIX_TYPES_TO_SET_METHOD[item_type],
                    )(pd.DataFrame())
            except ReferencedObjectDeletionNotAllowed:
                self.logger.warning(
                    f"Item {item} will not be deleted because it reference an object from binding constraints"
                )
            except NotImplementedError as e:
                self.logger.warning(
                    f"Failure to delete {item} because the method is not implemented yet on antares craft"
                )

        self.legacy_objects[:] = []

    def _iterate_through_model(
        self,
        valid_resources: dict,
        components: list,
        connections: list,
        mp: ModelsConfigurationProcessing,
    ):
        components.append(
            InputComponent(
                id=valid_resources["component"]["id"],
                model=valid_resources["model"],
                parameters=[
                    InputComponentParameter(
                        id=str(param.get("id")),
                        time_dependent=bool(param.get("time-dependent")),
                        scenario_dependent=bool(param.get("scenario-dependent")),
                        value=mp.convert_param_value(param["id"], param["value"]),
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
        if self.mode == ConversionMode.HYBRID:
            # TODO area-connections
            area_connections = []
            area_connections.append(InputAreaConnections())

        if self.mode == ConversionMode.HYBRID:
            for item in valid_resources.get("legacy-objects-to-delete", []):
                item = item.get("object-properties")
                if not isinstance(item, dict):
                    continue
                self.legacy_objects.append(item)


    def _convert_model_to_component_list(
        self, valid_areas: dict, resource_content: dict
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        components = []
        connections = []
        self.logger.info("Converting models to component list...")

        model_area_pattern = (
            f"${{{resource_content['template-parameters'][0]['name']}}}"
        )
        resource_name = resource_content["name"]
        mp = ModelsConfigurationProcessing(self.study)
        try:
            if resource_name in ["link"]:
                valid_resources: dict = self._validate_resources_not_excluded(
                    resource_content, "link"
                )
                for link in valid_resources.values():
                    data_with_link: dict = self._match_area_pattern(
                        resource_content, link.id, model_area_pattern
                    )
                    self._iterate_through_model(
                        data_with_link, components, connections, mp
                    )

            else:
                if resource_name == "thermal":
                    # Legacy conversion for thermal cluster
                    self._convert_thermal_to_component_list(
                        valid_areas, components, connections
                    )
                    return components, connections
                for area in valid_areas.values():
                    data_consolidated: dict = self._match_area_pattern(
                        resource_content, area.id, model_area_pattern
                    )
                    if resource_name in ["wind", "solar", "load"]:
                        if any(
                            not mp.check_timeseries_validity(param["value"])
                            for param in data_consolidated["component"]["parameters"]
                        ):
                            continue
                    cluster_type = next(
                        (
                            template.get("cluster-type")
                            for template in resource_content.get(
                                "template-parameters", []
                            )
                        ),
                        None,
                    )
                    if cluster_type:
                        for cluster_id in getattr(
                            area, TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[cluster_type]
                        )():
                            data_consolidated: dict = self._match_area_pattern(
                                data_consolidated, cluster_id, f"${{{cluster_type}}}"
                            )
                            self._iterate_through_model(
                                data_consolidated, components, connections, mp
                            )
                    else:
                        self._iterate_through_model(
                            data_consolidated, components, connections, mp
                        )

        except (KeyError, FileNotFoundError) as e:
            self.logger.error(
                f"Error while converting model to component list: {e}. "
                "Please check the model configuration file."
            )
            return components, connections

        return components, connections

    def _validate_resources_not_excluded(
        self, resource_content: dict, parameter: str
    ) -> dict:
        excluded_ids = set()
        for param in resource_content.get("template-parameters", []):
            if param.get("name") == parameter:
                excluded_ids.update(item["id"] for item in param.get("exclude", []))

        if parameter == "area":
            resources = self.areas
        elif parameter == "link":
            resources = self.study.get_links()
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")

        return {
            key: value for key, value in resources.items() if key not in excluded_ids
        }

    def convert_study_to_input_study(self) -> InputSystem:
        antares_historic_lib_id = "antares-historic"

        list_components: list[InputComponent] = []
        list_connections: list[InputPortConnections] = []
        list_valid_areas = set(self.areas.keys())
        all_excluded_areas = set()
        for file in RESOURCES_FOLDER.iterdir():
            if file.is_file() and file.name.endswith(".yaml"):
                resource_content = read_yaml_file(file).get("template", {})
                valid_areas: dict = self._validate_resources_not_excluded(
                    resource_content, "area"
                )

                components, connections = self._convert_model_to_component_list(
                    valid_areas, resource_content
                )
                list_components.extend(components)
                list_connections.extend(connections)

                for param in resource_content.get("template-parameters", []):
                    if param.get("name") == "area":
                        all_excluded_areas.update(
                            item["id"] for item in param.get("exclude", [])
                        )

                list_valid_areas.difference_update(all_excluded_areas)

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


@staticmethod
def count_objects_in_yaml_file(objet: Study):
    pass
    # TODO faire un compteur delements
