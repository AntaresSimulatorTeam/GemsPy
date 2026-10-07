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
from typing import Any, Dict, List, Optional, Tuple, Union

from gems_craft.model.library import Library
from gems_craft.study import (
    Component,
    ConstantData,
    DataBase,
    PortRef,
    System,
)
from gems_craft.study.data import (
    AbstractDataStructure,
    ScenarioSeriesData,
    TimeScenarioSeriesData,
    TimeSeriesData,
    dataframe_to_scenario_series,
    dataframe_to_set_indexed_series,
    dataframe_to_time_series,
    load_tidy_series_from_file,
    load_ts_from_file,
    nested_dict_to_set_indexed_series,
)
from gems_craft.study.parsing import (
    ComponentPropertySchema,
    ComponentSchema,
    PortConnectionsSchema,
    SystemSchema,
)
from gems_craft.study.scenario_builder import ScenarioBuilder


def _resolve_properties_raw_to_dict(
    raw: Optional[List[ComponentPropertySchema]],
    component_id: str,
) -> Dict[str, str]:
    """Turn parsed ``properties`` (list of ``ComponentPropertySchema``) into ``Component``'s dict."""
    if raw is None:
        return {}
    properties: Dict[str, str] = {}
    for item in raw:
        k = item.id
        if k in properties:
            raise ValueError(
                f"Component {component_id!r}: duplicate properties id {k!r}"
            )
        properties[k] = item.value
    return properties


def resolve_system(input_system: SystemSchema, libraries: dict[str, Library]) -> System:
    """
    Resolves:
    - components to be used for study
    - connections between components"""
    components_list = [
        _resolve_component(libraries, m) for m in input_system.components
    ]

    s = System(
        "study",
        global_sets={
            inst.id: tuple(inst.elements)  # type: ignore[arg-type]
            for inst in input_system.sets or []
        },
    )
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

    # TODO: this is validation (missing parameters/properties), not resolution —
    # extract into a dedicated validation module, as done for optim_config/.
    provided_param_ids = {p.id for p in component.parameters or []}
    missing_params = sorted(k for k in model.parameters if k not in provided_param_ids)
    if missing_params:
        raise ValueError(
            f"Component {component.id!r} (model {model.id!r}) is missing "
            f"parameter{'s' if len(missing_params) > 1 else ''} declared by the model: "
            f"{missing_params}."
        )

    properties = _resolve_properties_raw_to_dict(component.properties, component.id)
    missing = sorted(k for k in model.properties if k not in properties)
    if missing:
        raise ValueError(
            f"Component {component.id!r} (model {model.id!r}) is missing "
            f"propert{'y' if len(missing) == 1 else 'ies'} declared by the model: "
            f"{missing}."
        )

    return Component(
        model=model,
        id=component.id,
        scenario_group=component.scenario_group,
        properties=properties,
        integer_strategy=component.integer_strategy,
        local_sets={
            inst.id: tuple(inst.elements)  # type: ignore[arg-type]
            for inst in component.sets or []
        },
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
    global_sets = {s.id: list(s.elements) for s in input_system.sets or []}
    for comp in input_system.components:
        local_sets = {s.id: list(s.elements) for s in comp.sets or []}
        for param in comp.parameters or []:
            group = param.scenario_group or comp.scenario_group
            set_elements: Dict[str, List[Union[str, int]]] = {}
            for set_id in param.indexed_by:
                elements = local_sets.get(set_id, global_sets.get(set_id))
                if elements is None:
                    raise ValueError(
                        f"Component {comp.id!r}, parameter {param.id!r}: "
                        f"indexed-by set {set_id!r} is not instantiated."
                    )
                set_elements[set_id] = elements
            param_value = _build_data(
                param.time_dependent,
                param.scenario_dependent,
                param.value,
                timeseries_dir,
                set_elements,
            )
            database.add_data(comp.id, param.id, param_value, scenario_group=group)

    return database


def _build_data(
    time_dependent: bool,
    scenario_dependent: bool,
    param_value: Union[float, str, Dict[Union[str, int], Any]],
    timeseries_dir: Optional[Path],
    set_elements: Optional[Dict[str, List[Union[str, int]]]] = None,
) -> AbstractDataStructure:
    if isinstance(param_value, dict):
        if not set_elements:
            raise ValueError(
                "Inline values are only allowed for parameters with 'indexed-by'."
            )
        if time_dependent or scenario_dependent:
            raise ValueError(
                "Inline values are only allowed for parameters that depend on sets "
                "only; use a tidy CSV for time/scenario-dependent data."
            )
        return nested_dict_to_set_indexed_series(param_value, set_elements)
    if set_elements:
        if not isinstance(param_value, str):
            raise ValueError(
                f"A series name is expected for set-indexed data, got {param_value}"
            )
        return dataframe_to_set_indexed_series(
            load_tidy_series_from_file(param_value, timeseries_dir),
            time_dependent,
            scenario_dependent,
            set_elements,
        )
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
