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

"""Cross-validation of a resolved system against the models and data it refers to.

Kept apart from `resolve_components.py` (which only resolves the parsed system
into its runtime objects) and from `study.py` (which only holds them together)
— mirroring `optim_config/parsing.py` and `optim_config/validation.py`.
"""

from typing import Dict, List, Tuple

from gems_craft.expression import ExpressionNode
from gems_craft.expression.print import print_expr
from gems_craft.model import Model
from gems_craft.model.model import time_sum_bound_parameters
from gems_craft.study.data import TimeScenarioSeriesData, TimeSeriesData
from gems_craft.study.study import Study
from gems_craft.study.system import System


def check_component_models(system: System, input_models: Dict[str, Model]) -> bool:
    """
    Checks if all components in the System have a valid model from the library.
    Returns True if all components are consistent, raises ValueError otherwise.
    """
    # TODO: Update this check to verify that each component has a valid model from the lib it refers to (and not all libs)
    model_ids_set = input_models.keys()
    for component in system.all_components:
        if component.model.id not in model_ids_set:
            raise ValueError(
                f"Error: Component {component.id} has invalid model ID: {component.model.id}"
            )
    return True


def check_data_requirements(study: Study) -> None:
    """Validate that the database supplies data for every parameter of every
    component defined in the system.

    Raises
    ------
    ValueError
        If a required data entry is missing or its time/scenario structure
        does not match what the model parameter expects.
    """
    for component in study.system.components:
        for param in component.model.parameters.values():
            data_structure = study.database.get_data(component.id, param.name)

            if not data_structure.check_requirement(
                component.model.parameters[param.name].structure.time,
                component.model.parameters[param.name].structure.scenario,
            ):
                raise ValueError(
                    f"Data inconsistency for component: {component.id}, "
                    f"parameter: {param.name}. Requirement not met."
                )
    check_time_sum_bounds(study)


def check_time_sum_bounds(study: Study) -> None:
    """Validate that the parameters used in time sum bounds are fixed in time.

    As in Antares Simulator, this depends on the data given to each component,
    not on the model declaration: a parameter declared time-dependent may be
    used in a bound if the component gives it a constant value.

    Raises
    ------
    ValueError
        If a component gives a time series to a parameter used in a time sum
        bound.
    """
    # Components often share a model: list its bound parameters only once.
    bounds_by_model: Dict[int, List[Tuple[str, ExpressionNode, str]]] = {}
    for component in study.system.components:
        model_key = id(component.model)
        if model_key not in bounds_by_model:
            bounds_by_model[model_key] = list(
                time_sum_bound_parameters(component.model)
            )
        for parameter, bound, where in bounds_by_model[model_key]:
            data = study.database.get_data(component.id, parameter)
            if isinstance(data, (TimeSeriesData, TimeScenarioSeriesData)):
                raise ValueError(
                    f"Component '{component.id}' (model '{component.model.id}'): "
                    f"a time sum bound must be fixed in time, but parameter "
                    f"'{parameter}' of '{print_expr(bound)}' in {where} is a "
                    "time series."
                )
