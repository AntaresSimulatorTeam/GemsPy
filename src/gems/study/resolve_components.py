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
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from gems.model import Model
from gems.model.library import Library
from gems.study import (
    Component,
    ConstantData,
    DataBase,
    PortRef,
    PortsConnection,
    System,
)
from gems.study.data import (
    AbstractDataStructure,
    ScenarioSeriesData,
    TimeScenarioSeriesData,
    TimeSeriesData,
    dataframe_to_scenario_series,
    dataframe_to_time_series,
    load_ts_from_file,
)
from gems.study.parsing import ComponentSchema, PortConnectionsSchema, SystemSchema
from gems.study.scenario_builder import ScenarioBuilder


def resolve_system(input_system: SystemSchema, libraries: dict[str, Library]) -> System:
    """
    Resolves:
    - components to be used for study
    - connections between components"""
    components_list = [
        _resolve_component(libraries, m) for m in input_system.components
    ]

    s = System("study")
    for component in components_list:
        s.add_component(component)

    connections_input = getattr(input_system, "connections", []) or []
    for cnx in connections_input:
        port_ref_1, port_ref_2 = _resolve_port_refs(cnx, components_list)
        s.connect(port_ref_1, port_ref_2)

    return s


def _resolve_component(
    libraries: dict[str, Library], component: ComponentSchema
) -> Component:
    lib_id, model_id = component.model.split(".")
    model = libraries[lib_id].models[f"{lib_id}.{model_id}"]

    return Component(
        model=model,
        id=component.id,
        scenario_group=component.scenario_group,
    )


def _resolve_port_refs(
    connection: PortConnectionsSchema,
    all_components: List[Component],
) -> Tuple[PortRef, PortRef]:
    component_1 = _get_component_by_id(all_components, connection.component1)
    component_2 = _get_component_by_id(all_components, connection.component2)
    assert component_1 is not None and component_2 is not None
    return PortRef(component_1, connection.port1), PortRef(
        component_2, connection.port2
    )


def _get_component_by_id(
    all_components: List[Component], component_id: str
) -> Optional[Component]:
    components_dict = {component.id: component for component in all_components}
    return components_dict.get(component_id)


def consistency_check(system: System, input_models: Dict[str, Model]) -> bool:
    """
    Checks if all components in the System have a valid model from the library.
    Returns True if all components are consistent, raises ValueError otherwise.
    """
    # TODO: Update this consistency check to check if each component have a valid model from the lib it refers to (and not all libs)
    model_ids_set = input_models.keys()
    for component in system.all_components:
        if component.model.id not in model_ids_set:
            raise ValueError(
                f"Error: Component {component.id} has invalid model ID: {component.model.id}"
            )
    return True


def build_data_base(
    input_system: SystemSchema,
    timeseries_dir: Optional[Path],
    scenario_builder: Optional[ScenarioBuilder] = None,
) -> DataBase:
    """Build a DataBase from the system description and optional ScenarioBuilder.

    When a ``ScenarioBuilder`` is provided, each parameter's ``scenario_group``
    is recorded so that ``DataBase.get_values()`` can resolve MC scenario indices
    to data-series column indices at use time.
    """
    database = DataBase(scenario_builder=scenario_builder)
    for comp in input_system.components:
        for param in comp.parameters or []:
            group = param.scenario_group or comp.scenario_group
            param_value = _build_data(
                param.time_dependent,
                param.scenario_dependent,
                param.value,
                timeseries_dir,
            )
            database.add_data(comp.id, param.id, param_value, scenario_group=group)

    return database


def _build_data(
    time_dependent: bool,
    scenario_dependent: bool,
    param_value: Union[float, str],
    timeseries_dir: Optional[Path],
) -> AbstractDataStructure:
    if isinstance(param_value, str):
        ts_data = load_ts_from_file(param_value, timeseries_dir)
        if time_dependent and scenario_dependent:
            return TimeScenarioSeriesData(ts_data)
        elif time_dependent:
            return TimeSeriesData(dataframe_to_time_series(ts_data))
        elif scenario_dependent:
            return ScenarioSeriesData(dataframe_to_scenario_series(ts_data))
        else:
            raise ValueError(
                f"A float value is expected for constant data, got {param_value}"
            )
    else:
        if time_dependent or scenario_dependent:
            raise ValueError(
                f"A timeseries name is expected for time or scenario dependent data, got {param_value}"
            )
        return ConstantData(float(param_value))
