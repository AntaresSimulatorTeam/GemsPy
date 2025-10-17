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
import shutil
from pathlib import Path
from types import MappingProxyType
from typing import Any, List, Optional, Union

import pandas as pd
from antares.craft.exceptions.exceptions import ReferencedObjectDeletionNotAllowed
from antares.craft.model.study import Study, read_study_local
from antares.craft.model.thermal import ThermalCluster

from gems.input_converter.src.config import (
    MATRIX_TYPES_TO_SET_METHOD,
    MODEL_NAME_TO_FILE_NAME,
    STUDY_LEVEL_DELETION,
    STUDY_LEVEL_GET,
    TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD,
    TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD,
)
from gems.input_converter.src.data_preprocessing.data_classes import ConversionMode
from gems.input_converter.src.data_preprocessing.preprocessing import (
    ModelsConfigurationProcessing,
)
from gems.input_converter.src.data_preprocessing.thermal import ThermalDataPreprocessing
from gems.input_converter.src.utils import (
    dump_to_yaml,
    match_area_pattern,
    read_yaml_file,
    resolve_path,
)
from gems.study.parsing import (
    InputAreaConnections,
    InputComponent,
    InputComponentParameter,
    InputPortConnections,
    InputSystem,
)

RESOURCES_FOLDER = Path(__file__).parents[1] / "data" / "model_configuration"
LIBS_FOLDER = "model-libraries"


class AntaresStudyConverter:
    def __init__(
        self,
        study_input: Union[Path, Study],
        logger: logging.Logger,
        mode: str = "full",
        output_folder: Path = Path("/tmp/"),
        period: Optional[int] = None,
        lib_paths: Optional[list[str]] = None,
        model_list: Optional[list[str]] = None,
    ):
        """
        Initialize processor
        """
        self.logger = logger
        self.period: int = period if period else 168
        self.lib_paths: list[str] = lib_paths if lib_paths else []
        self.model_list: list[str] = model_list if model_list else []

        try:
            self.mode = ConversionMode(mode)
        except ValueError:
            raise ValueError(
                f"Invalid conversionmode: {mode}, possible values are {[conv_mode.value for conv_mode in ConversionMode]}"
            )

        # TODO: The logic is still too complicated, needs more refacto / to understand why thermal preprocessing sometimes needs different paths
        study_input_path = (
            study_input.path.stem
            if isinstance(study_input, Study)
            else study_input.stem
        )
        self.output_folder = output_folder / study_input_path

        if self.mode == ConversionMode.HYBRID:
            # In hybrid mode, the output is the input study from which we replace converted components by Gems ones, hence we copy the original study
            shutil.copytree(
                study_input.path if isinstance(study_input, Study) else study_input,
                self.output_folder,
                dirs_exist_ok=True,
            )
            if isinstance(study_input, Path):
                study_input = self.output_folder
        else:
            # In full mode, the output is a full Gems study so no need to copy the original study, we start "from scratch"
            self.output_folder.mkdir(parents=True, exist_ok=True)

        if isinstance(study_input, Study):
            # We have a different way of managing thermal preprocessing files, because in this case we want to modify the study_path.
            # But in the same moment we dont want the preprocessing files in the modified study_path
            self.thermal_input_path = Path(study_input.path)
            study_input.path = self.output_folder
            self.study = study_input
        else:
            self.thermal_input_path = Path(study_input)
            # TODO: Check whether this dinstinction is needed
            if mode == ConversionMode.HYBRID:
                self.study = read_study_local(resolve_path(self.output_folder))
            else:
                self.study = read_study_local(resolve_path(study_input))

        self.output_path = self.output_folder / "input" / "system.yml"

        self.areas: MappingProxyType = self.study.get_areas()
        self.legacy_objects: list[dict] = []

    def _convert_thermal_to_component_list(
        self,
        lib_id: str,
        valid_areas: dict,
        components: list,
        connections: list,
        area_connections: list,
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        self.logger.info("Converting thermals to component list...")
        # Add thermal components for each area
        for area in (area for area in self.areas.values() if area.id in valid_areas):
            thermals: dict[str, ThermalCluster] = area.get_thermals()
            for thermal in thermals.values():
                # TODO Do  we move preprocessing files in data series folder ?
                series_path = (
                    self.thermal_input_path
                    / "input"
                    / "thermal"
                    / "series"
                    / Path(thermal.area_id)
                    / Path(thermal.id)
                    / "series.txt"
                )
                tdp = ThermalDataPreprocessing(thermal, self.thermal_input_path)
                components.append(
                    InputComponent(
                        id=f"{thermal.area_id}_{thermal.id}",
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
                if self.mode == ConversionMode.FULL:
                    connections.append(
                        InputPortConnections(
                            component1=f"{thermal.area_id}_{thermal.id}",
                            port1="balance_port",
                            component2=f"{thermal.area_id}",
                            port2="balance_port",
                        )
                    )
                else:
                    area_connections.append(
                        InputAreaConnections(
                            component=f"{thermal.area_id}_{thermal.id}",
                            port="balance_port",
                            area=f"{thermal.area_id}",
                        )
                    )
        return components, connections

    def _convert_area_to_component_list(
        self, lib_id: str, list_valid_areas: Optional[list[str]] = None
    ) -> list[InputComponent]:
        components = []
        self.logger.info("Converting areas to component list...")
        for area in self.areas.values():
            if not list_valid_areas or area.id in list_valid_areas:
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

    def _delete_legacy_objects(self) -> None:
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
                    # To "delete" legacy wind, solar or load object, we simply set an empty timeseries
                    getattr(
                        self.areas[item.get("area")],
                        MATRIX_TYPES_TO_SET_METHOD[item_type],
                    )(pd.DataFrame())

            except ReferencedObjectDeletionNotAllowed:
                self.logger.warning(
                    f"Item {item} will not be deleted because it is referenced in a binding constraint"
                )
            except NotImplementedError:
                self.logger.warning(
                    f"Failure to delete {item} because the method is not implemented yet on antares craft"
                )

        self.legacy_objects = []

    def _iterate_through_model(
        self,
        valid_resources: dict,
        components: list,
        connections: list,
        area_connections: list,
        mp: ModelsConfigurationProcessing,
    ) -> None:
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

        if self.mode == ConversionMode.HYBRID:
            for resource_connection in valid_resources["area-connections"]:
                if "." in resource_connection["component"]:
                    component_parts = resource_connection["component"].split(".")
                    component_value = getattr(
                        self.study.get_links()[component_parts[0]], component_parts[1]
                    )
                else:
                    component_value = resource_connection["component"]
                area_connections.append(
                    InputAreaConnections(
                        component=component_value,
                        port=resource_connection["port"],
                        area=resource_connection["area"],
                    )
                )
            for item in valid_resources.get("legacy-objects-to-delete", []):
                self.legacy_objects.append(item["object-properties"])
        else:
            for resource_connection in valid_resources["connections"]:
                if "." in resource_connection["component2"]:
                    component2_parts = resource_connection["component2"].split(".")
                    component2_value = getattr(
                        self.study.get_links()[component2_parts[0]], component2_parts[1]
                    )
                else:
                    component2_value = resource_connection["component2"]
                connections.append(
                    InputPortConnections(
                        component1=resource_connection["component1"],
                        port1=resource_connection["port1"],
                        component2=component2_value,
                        port2=resource_connection["port2"],
                    )
                )

    def _convert_model_to_component_list(
        self, valid_areas: dict, resource_content: dict
    ) -> tuple[
        list[InputComponent], list[InputPortConnections], list[InputAreaConnections]
    ]:
        components: list[InputComponent] = []
        connections: list[InputPortConnections] = []
        area_connections: list[InputAreaConnections] = []
        self.logger.info("Converting models to component list...")

        model_area_pattern = (
            f"${{{resource_content['template-parameters'][0]['name']}}}"
        )
        resource_name = resource_content["name"]
        mp = ModelsConfigurationProcessing(self.study, self.mode, self.output_folder)
        try:
            if resource_name in ["link"]:
                valid_resources: dict = self._validate_resources_not_excluded(
                    resource_content, "link"
                )
                for link in valid_resources.values():
                    data_with_link: dict = match_area_pattern(
                        resource_content, link.id, model_area_pattern
                    )
                    self._iterate_through_model(
                        data_with_link, components, connections, area_connections, mp
                    )
            else:
                if resource_name == "thermal":
                    # Legacy conversion for thermal cluster
                    self._convert_thermal_to_component_list(
                        self.get_model_name_among_libs("thermal"),
                        valid_areas,
                        components,
                        connections,
                        area_connections,
                    )
                    return components, connections, area_connections
                for area in valid_areas.values():
                    data_consolidated: dict = match_area_pattern(
                        resource_content, area.id, model_area_pattern
                    )
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
                            data_consolidated = match_area_pattern(
                                data_consolidated, cluster_id, f"${{{cluster_type}}}"
                            )
                            self._iterate_through_model(
                                data_consolidated,
                                components,
                                connections,
                                area_connections,
                                mp,
                            )

                    elif resource_name in ["wind", "solar", "load"]:
                        if all(
                            mp.check_timeseries_validity(param["value"])
                            for param in data_consolidated["component"]["parameters"]
                        ):
                            self._iterate_through_model(
                                data_consolidated,
                                components,
                                connections,
                                area_connections,
                                mp,
                            )
                    else:
                        self._iterate_through_model(
                            data_consolidated,
                            components,
                            connections,
                            area_connections,
                            mp,
                        )
        except (KeyError, FileNotFoundError) as e:
            self.logger.error(
                f"Error while converting model to component list: {e}. "
                "Please check the model configuration file."
            )
            return components, connections, area_connections

        return components, connections, area_connections

    def _validate_resources_not_excluded(
        self, resource_content: dict, parameter: str
    ) -> dict:
        excluded_ids: set[Any] = set()
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

    @staticmethod
    def _extract_lib_and_model_ids(path: str) -> tuple[str, list]:
        lib_data = read_yaml_file(Path(path))["library"]
        models = lib_data.get("models", [])
        return lib_data["id"], [model["id"] for model in models]

    def _check_converted_models_are_in_libs(self) -> None:
        lib_to_model_ids = {}
        for lib_path in self.lib_paths:
            lib_id, model_ids = self._extract_lib_and_model_ids(lib_path)
            lib_to_model_ids[lib_id] = model_ids

        for model in self.model_list:
            model_conversion_config_file = (
                RESOURCES_FOLDER / MODEL_NAME_TO_FILE_NAME[model]
            )
            if not model_conversion_config_file.exists():
                raise FileNotFoundError(
                    f"The model configuration file for {model} has not been found at the location {model_conversion_config_file}"
                )
            lib_and_model_id = read_yaml_file(model_conversion_config_file)["template"][
                "model"
            ]
            lib_id, model_id = lib_and_model_id.split(".")
            if lib_id not in lib_to_model_ids:
                raise ValueError(
                    "Library {lib_id} has not been found in provided libraries"
                )
            if model_id not in lib_to_model_ids[lib_id]:
                raise ValueError(
                    f"Model {model_id} has not been found in library {lib_id}"
                )

    def _copy_libs_to_model_librairies(self) -> None:
        # Retrieve library files and put it in the output study (as fro now libs must be contained in modeler studies)
        dest_dir = self.output_folder / "input" / LIBS_FOLDER
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in self.lib_paths:
            shutil.copy2(path, dest_dir)

    def get_model_name_among_libs(self, model_name: str) -> str:
        for lib_path in self.lib_paths:
            lib_name, file_model = self._extract_lib_and_model_ids(lib_path)
            if model_name in file_model:
                return lib_name
        return "antares-historic"

    def convert_study_to_input_system(self) -> InputSystem:
        self._copy_libs_to_model_librairies()
        if self.mode == ConversionMode.HYBRID:
            self._check_converted_models_are_in_libs()
        # TODO : Needs to add a check that all legacy models are in provided libs in full mode

        list_components: list[InputComponent] = []
        list_connections: list[InputPortConnections] = []
        list_area_connections: list[InputAreaConnections] = []

        list_valid_areas: set[str] = set(self.areas.keys())
        all_excluded_areas: set[Any] = set()

        def _conversion_loop(model: str) -> None:
            file_path = RESOURCES_FOLDER / MODEL_NAME_TO_FILE_NAME[model]
            if not file_path.exists():
                return
            resource_content = read_yaml_file(file_path).get("template", {})
            valid_areas = self._validate_resources_not_excluded(
                resource_content, "area"
            )

            (
                components,
                connections,
                area_connections,
            ) = self._convert_model_to_component_list(valid_areas, resource_content)
            list_components.extend(components)
            list_connections.extend(connections)
            list_area_connections.extend(area_connections)

            for param in resource_content.get("template-parameters", []):
                if param.get("name") == "area":
                    all_excluded_areas.update(
                        item["id"] for item in param.get("exclude", [])
                    )

            list_valid_areas.difference_update(all_excluded_areas)

        if self.mode == ConversionMode.HYBRID:
            for model in self.model_list:
                _conversion_loop(model)
                self._delete_legacy_objects()
        else:
            for model in MODEL_NAME_TO_FILE_NAME:
                _conversion_loop(model)

            list_components.extend(
                self._convert_area_to_component_list(
                    self.get_model_name_among_libs("area"), list(list_valid_areas)
                )
            )

        self.logger.info(
            "Converting node, components and connections into Input study..."
        )
        system = InputSystem(
            id=self.study.name,
            components=list_components,
            connections=list_connections or None,
            area_connections=list_area_connections or None,
        )
        data = system.model_dump(exclude_none=True)
        return InputSystem(**data)

    def process_all(self) -> None:
        system = self.convert_study_to_input_system()
        self.logger.info("Dumping input system into yaml file...")
        dump_to_yaml(model=system, output_path=self.output_path)
