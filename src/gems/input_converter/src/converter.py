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
from typing import Optional, Union

import pandas as pd
from antares.craft.exceptions.exceptions import ReferencedObjectDeletionNotAllowed
from antares.craft.model.link import Link
from antares.craft.model.study import Study, read_study_local

from gems.input_converter.src.config import (
    LINK_TYPES,
    MATRIX_TYPES,
    MATRIX_TYPES_TO_SET_METHOD,
    MODEL_NAME_TO_FILE_NAME,
    STUDY_LEVEL_DELETION,
    STUDY_LEVEL_GET,
    TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD,
    TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD,
)
from gems.input_converter.src.data_preprocessing.data_classes import ConversionMode
from gems.input_converter.src.data_preprocessing.preprocessing import (
    ModelConversionPreprocessor,
)
from gems.input_converter.src.data_preprocessing.thermal import ThermalDataPreprocessing
from gems.input_converter.src.parsing import (
    ConversionTemplate,
    ObjectProperties,
    VirtualObjectsRepository,
    parse_conversion_template,
)
from gems.input_converter.src.utils import dump_to_yaml, read_yaml_file, resolve_path
from gems.study.parsing import (
    InputAreaConnections,
    InputComponent,
    InputComponentParameter,
    InputPortConnections,
    InputSystem,
)

ANTARES_HISTORIC_LIB_ID = "antares-historic"
MODEL_TEMPLATE_FOLDER = Path(__file__).parents[1] / "data" / "model_configuration"
LIBS_FOLDER = "model-libraries"
SERIES_FOLDER = "data-series"

# TODO: Move all global variables in a config class, that is used in AntaresStudyConverter constructor


class AntaresStudyConverter:
    def __init__(
        self,
        study_input: Union[Path, Study],
        logger: logging.Logger,
        mode: str = "full",
        output_folder: Path = Path("/tmp/"),
        period: Optional[int] = None,
        lib_paths: Optional[list[str]] = None,
        models_to_convert: list[str] = list(MODEL_NAME_TO_FILE_NAME.keys()),
        modeler_scenario_builder_file: Optional[Path] = None,
    ):
        """
        Initialize processor
        """
        self.logger = logger
        self.period: int = period if period else 168
        self.lib_paths: list[str] = lib_paths if lib_paths else []
        self.models_to_convert = models_to_convert
        self.modeler_scenario_builder_file = modeler_scenario_builder_file
        try:
            self.mode = ConversionMode(mode)
        except ValueError:
            raise ValueError(
                f"Invalid conversion mode: {mode}, possible values are {[conv_mode.value for conv_mode in ConversionMode]}"
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
            study_input.path = self.output_folder
            self.study = study_input
        else:
            # TODO: Check whether this dinstinction is needed
            if mode == ConversionMode.HYBRID:
                self.study = read_study_local(resolve_path(self.output_folder))
            else:
                self.study = read_study_local(resolve_path(study_input))
        self.output_system_path = self.output_folder / "input" / "system.yml"
        self.areas = self.study.get_areas()
        self.legacy_objects: list[ObjectProperties] = []

    def _convert_thermal_to_component_list(
        self,
        lib_id: str,
        virtual_objects: VirtualObjectsRepository,
        components: list,
        connections: list,
        area_connections: list,
        scenario_group: Optional[str] = None,
    ) -> tuple[list[InputComponent], list[InputPortConnections]]:
        self.logger.info("Converting thermals to component list...")
        # Add thermal components for each area
        for area in self.areas.values():
            if area.id not in virtual_objects.areas:
                resolved_virtual_objects = virtual_objects.resolve_template(
                    "${area}", area.id
                )
                thermals = area.get_thermals()
                for thermal in thermals.values():
                    if thermal.id not in resolved_virtual_objects.thermals:
                        tdp = ThermalDataPreprocessing(thermal, self.output_folder)
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
                                    tdp.generate_component_parameter("p_max_cluster"),
                                ],
                                scenario_group=scenario_group,
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
                            self.legacy_objects.append(
                                ObjectProperties(
                                    type="thermal",
                                    area=thermal.area_id,
                                    link=None,
                                    cluster=thermal.id,
                                    binding_constraint_id=None,
                                    field=None,
                                )
                            )
        return components, connections

    def _convert_area_to_component_list(
        self, lib_id: str, excluded_areas: Optional[list[str]] = None
    ) -> list[InputComponent]:
        components = []
        self.logger.info("Converting areas to component list...")
        for area in self.areas.values():
            if not excluded_areas or area.id not in excluded_areas:
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
        for legacy_component in self.legacy_objects:
            try:
                if legacy_component.type in STUDY_LEVEL_DELETION:
                    if legacy_component.type == "area":
                        id = legacy_component.area
                    elif legacy_component.type == "link":
                        id = legacy_component.link
                    elif legacy_component.type == "binding_constraint":
                        id = legacy_component.binding_constraint_id
                    else:
                        # Should not happen
                        pass
                    getattr(self.study, STUDY_LEVEL_DELETION[legacy_component.type])(
                        getattr(self.study, STUDY_LEVEL_GET[legacy_component.type])()[
                            id
                        ]
                    )
                elif (
                    legacy_component.type in TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD
                    and legacy_component.area is not None
                ):
                    getattr(
                        self.areas[legacy_component.area],
                        TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD[legacy_component.type],
                    )(
                        getattr(
                            self.areas[legacy_component.area],
                            TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[legacy_component.type],
                        )()[legacy_component.cluster]
                    )
                elif (
                    legacy_component.type in MATRIX_TYPES_TO_SET_METHOD
                    and legacy_component.area is not None
                ):
                    # To "delete" legacy wind, solar or load object, we simply set an empty timeseries
                    getattr(
                        self.areas[legacy_component.area],
                        MATRIX_TYPES_TO_SET_METHOD[legacy_component.type],
                    )(pd.DataFrame())

            except ReferencedObjectDeletionNotAllowed:
                self.logger.warning(
                    f"Item {legacy_component} will not be deleted because it is referenced in a binding constraint"
                )
            except NotImplementedError:
                self.logger.warning(
                    f"Failure to delete {legacy_component} because the method is not implemented yet on antares craft"
                )

        self.legacy_objects = []

    def _iterate_through_model(
        self,
        resolved_conversion_template: ConversionTemplate,
        components: list,
        connections: list,
        area_connections: list,
        mp: ModelConversionPreprocessor,
    ) -> None:
        parameters = [
            InputComponentParameter(
                id=param.id,
                time_dependent=bool(param.time_dependent),
                scenario_dependent=bool(param.scenario_dependent),
                value=mp.convert_param_value(param.id, param.value),
            )
            for param in resolved_conversion_template.component.parameters
        ]
        scenario_group = getattr(resolved_conversion_template, "scenario_group", None)
        kwargs = {}
        if scenario_group is not None:
            kwargs["scenario_group"] = scenario_group

        components.append(
            InputComponent(
                id=(resolved_conversion_template.component.id).replace(" ", "_"),
                model=resolved_conversion_template.model,
                parameters=parameters,
                **kwargs,
            )
        )

        if self.mode == ConversionMode.HYBRID:
            for area_connection in resolved_conversion_template.area_connections:
                # TODO: Improve logic
                if "." in area_connection.component:
                    component_parts = area_connection.component.split(".")
                    component_value = getattr(
                        self.study.get_links()[component_parts[0]], component_parts[1]
                    )
                else:
                    component_value = area_connection.component
                component_value = component_value.replace(" ", "_")
                if "." in area_connection.area:
                    area_parts = area_connection.area.split(".")
                    area_value = getattr(
                        self.study.get_links()[area_parts[0]], area_parts[1]
                    )
                else:
                    area_value = area_connection.area
                area_value = area_value.replace(" ", "_")
                area_connections.append(
                    InputAreaConnections(
                        component=component_value,
                        port=area_connection.port,
                        area=area_value,
                    )
                )
            # TODO : Simplify usage, use directly legacy_objectis_to_delete
            for item in resolved_conversion_template.legacy_objects_to_delete:
                self.legacy_objects.append(item.object_properties)
        else:
            for connection in resolved_conversion_template.connections:
                # TODO: Factorize logic with previous connections
                treated_components = []
                for component in [connection.component1, connection.component2]:
                    if "." in component:
                        component_parts = component.split(".")
                        component_value = getattr(
                            self.study.get_links()[component_parts[0]],
                            component_parts[1],
                        )
                    else:
                        component_value = component
                    treated_components.append(component_value.replace(" ", "_"))

                connections.append(
                    InputPortConnections(
                        component1=treated_components[0],
                        port1=connection.port1,
                        component2=treated_components[1],
                        port2=connection.port2,
                    )
                )

    def _convert_model_to_component_list(
        self,
        conversion_template: ConversionTemplate,
        virtual_objects: VirtualObjectsRepository = VirtualObjectsRepository(),
    ) -> tuple[
        list[InputComponent], list[InputPortConnections], list[InputAreaConnections]
    ]:
        components: list[InputComponent] = []
        connections: list[InputPortConnections] = []
        area_connections: list[InputAreaConnections] = []

        model_area_pattern = f"${{{conversion_template.template_parameters[0].name}}}"

        model_preprocessor = ModelConversionPreprocessor(
            self.study, self.mode, self.output_folder
        )

        try:
            if conversion_template.name in LINK_TYPES:
                for link in self.study.get_links().values():
                    if not self.is_virtual_link(link, virtual_objects):
                        resolved_template = conversion_template.resolve_template(
                            model_area_pattern, link.id
                        )
                        self._iterate_through_model(
                            resolved_template,
                            components,
                            connections,
                            area_connections,
                            model_preprocessor,
                        )
            else:
                if conversion_template.name == "thermal":
                    # Legacy conversion for thermal cluster
                    self._convert_thermal_to_component_list(
                        self.get_model_name_among_libs("thermal"),
                        virtual_objects,
                        components,
                        connections,
                        area_connections,
                        scenario_group=getattr(
                            conversion_template, "scenario_group", None
                        ),
                    )
                    return components, connections, area_connections
                for area in self.areas.values():
                    if area.id not in virtual_objects.areas:
                        resolved_template = conversion_template.resolve_template(
                            model_area_pattern, area.id
                        )
                        cluster_type = next(
                            (
                                template.cluster_type
                                for template in conversion_template.template_parameters
                            ),
                            None,
                        )
                        if cluster_type:
                            for cluster_id in getattr(
                                area, TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[cluster_type]
                            )():
                                # We have already resolved areas, now need to resolve cluster ids
                                resolved_template = resolved_template.resolve_template(
                                    f"${{{cluster_type}}}", cluster_id
                                )
                                self._iterate_through_model(
                                    resolved_template,
                                    components,
                                    connections,
                                    area_connections,
                                    model_preprocessor,
                                )

                        elif conversion_template.name in MATRIX_TYPES:
                            if all(
                                model_preprocessor.check_timeseries_validity(
                                    param.value
                                )
                                for param in resolved_template.component.parameters
                            ):
                                self._iterate_through_model(
                                    resolved_template,
                                    components,
                                    connections,
                                    area_connections,
                                    model_preprocessor,
                                )
                        else:
                            self._iterate_through_model(
                                resolved_template,
                                components,
                                connections,
                                area_connections,
                                model_preprocessor,
                            )
        except (KeyError, FileNotFoundError) as e:
            self.logger.error(
                f"Error while converting model to component list: {e}. "
                "Please check the model configuration file."
            )
            return components, connections, area_connections

        return components, connections, area_connections

    @staticmethod
    def is_virtual_link(link: Link, virtual_objects: VirtualObjectsRepository) -> bool:
        return (
            link.id in virtual_objects.links
            or link.area_from_id in virtual_objects.areas
            or link.area_to_id in virtual_objects.areas
        )

    @staticmethod
    def _extract_lib_and_model_ids(path: str) -> tuple[str, list]:
        lib_data = read_yaml_file(Path(path))["library"]
        models = lib_data.get("models", [])
        return lib_data["id"], [model["id"] for model in models]

    def _check_converted_models_are_in_libs(
        self, model_conversion_templates: dict[str, ConversionTemplate]
    ) -> None:
        lib_to_model_ids = {}
        for lib_path in self.lib_paths:
            lib_id, model_ids = self._extract_lib_and_model_ids(lib_path)
            lib_to_model_ids[lib_id] = model_ids

        for model in self.models_to_convert:
            lib_id, model_id = model_conversion_templates[model].model.split(".")
            if lib_id not in lib_to_model_ids:
                raise ValueError(
                    f"Library {lib_id} has not been found in provided libraries"
                )
            if model_id not in lib_to_model_ids[lib_id]:
                raise ValueError(
                    f"Model {model_id} has not been found in library {lib_id}"
                )

    # TODO: Does not depend on self for now, but will be once the config is a class attribute
    def _get_model_conversion_template(self, model: str) -> ConversionTemplate:
        model_conversion_template_file = (
            MODEL_TEMPLATE_FOLDER / MODEL_NAME_TO_FILE_NAME[model]
        )
        if not model_conversion_template_file.exists():
            raise FileNotFoundError(
                f"The model configuration file for {model} has not been found at the location {model_conversion_template_file}"
            )
        # TODO: Parse yaml file with Pydantic to return a proper Python object rather than a generic dict
        with model_conversion_template_file.open() as template:
            return parse_conversion_template(template)

    def _copy_libs_to_model_librairies(self) -> None:
        # Retrieve library files and put it in the output study (as fro now libs must be contained in modeler studies)
        dest_dir = self.output_folder / "input" / LIBS_FOLDER
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in self.lib_paths:
            shutil.copy2(path, dest_dir)

    def _create_dataseries_dir(self) -> None:
        dest_dir = self.output_folder / "input" / SERIES_FOLDER
        dest_dir.mkdir(parents=True, exist_ok=True)

    def get_model_name_among_libs(self, model_name: str) -> str:
        for lib_path in self.lib_paths:
            lib_name, file_model = self._extract_lib_and_model_ids(lib_path)
            if model_name in file_model:
                return lib_name
        return ANTARES_HISTORIC_LIB_ID

    def _convert_single_model(
        self,
        conversion_template: ConversionTemplate,
        virtual_objects: VirtualObjectsRepository,
        components: list[InputComponent],
        connections: list[InputPortConnections],
        area_connections: list[InputAreaConnections],
    ) -> None:
        self.logger.info(
            f"Converting components of model {conversion_template.name}..."
        )

        (
            components_from_model,
            connections_from_model,
            area_connections_from_model,
        ) = self._convert_model_to_component_list(conversion_template, virtual_objects)

        components.extend(components_from_model)
        connections.extend(connections_from_model)
        area_connections.extend(area_connections_from_model)

    def convert_study_to_input_system(self) -> InputSystem:
        self._copy_libs_to_model_librairies()
        self._copy_scenario_builder()
        self._create_dataseries_dir()
        model_conversion_templates = self._build_model_conversion_templates()
        self._check_converted_models_are_in_libs(model_conversion_templates)
        virtual_objects = self._build_virtual_objects_repo(model_conversion_templates)
        components: list[InputComponent] = []
        connections: list[InputPortConnections] = []
        area_connections: list[InputAreaConnections] = []
        for model in self.models_to_convert:
            conversion_template = model_conversion_templates[model]
            self._convert_single_model(
                conversion_template,
                virtual_objects,
                components,
                connections,
                area_connections,
            )
        if self.mode == ConversionMode.HYBRID:
            self._delete_legacy_objects()
        else:
            components.extend(
                self._convert_area_to_component_list(
                    ANTARES_HISTORIC_LIB_ID, virtual_objects.areas
                )
            )
        system = InputSystem(
            id=self.study.name,
            components=components,
            connections=connections or None,
            area_connections=area_connections or None,
        )
        data = system.model_dump(exclude_none=True)
        return InputSystem(**data)

    def _build_model_conversion_templates(self) -> dict[str, ConversionTemplate]:
        model_conversion_templates: dict[str, ConversionTemplate] = {}
        for model in self.models_to_convert:
            model_conversion_templates[model] = self._get_model_conversion_template(
                model
            )
        return model_conversion_templates

    def _build_virtual_objects_repo(
        self, model_conversion_templates: dict[str, ConversionTemplate]
    ) -> VirtualObjectsRepository:
        virtual_objects = VirtualObjectsRepository()
        for model in self.models_to_convert:
            virtual_objects_this_model = model_conversion_templates[
                model
            ].get_excluded_objects_ids()
            virtual_objects.add(virtual_objects_this_model)
        return virtual_objects

    def process_all(self) -> None:
        system = self.convert_study_to_input_system()
        self.logger.info("Dumping input system into yaml file...")
        dump_to_yaml(model=system, output_path=self.output_system_path)

    def _copy_scenario_builder(self) -> None:
        if not self.modeler_scenario_builder_file:
            return

        dest = self.output_folder / "input" / "data-series"
        dest.mkdir(parents=True, exist_ok=True)

        dest_file = dest / "modeler-scenariobuilder.dat"  # enforce name

        try:
            shutil.copy2(self.modeler_scenario_builder_file, dest_file)
            self.logger.info(f"Copied scenario builder file to {dest_file}")
        except Exception as e:
            self.logger.warning(f"Failed to copy scenario builder file: {e}")
