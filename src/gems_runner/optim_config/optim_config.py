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

from typing import TYPE_CHECKING, List, Optional, Set


from gems_runner.expression.expression import (
    AdditionNode,
    BinaryOperatorNode,
    ExpressionNode,
    MaxNode,
    MinNode,
    UnaryOperatorNode,
    VariableNode,
)
from gems_craft.optim_config.parsing import (
    ElementLocation,
    OutOfBoundsProcessingConfig,
    ModelDecompositionConfig,
    OptimConfig,
)

if TYPE_CHECKING:
    from gems_runner.model.model import Model
    from gems_craft.study.scenario_builder import ScenarioBuilder
    from gems_runner.study.system import System

_MASTER_LOCS: Set[ElementLocation] = {
    ElementLocation.MASTER,
    ElementLocation.MASTER_AND_SUBPROBLEMS,
}


def _collect_variable_names(expr: ExpressionNode) -> Set[str]:
    """Recursively collect all variable names referenced in an expression."""
    if isinstance(expr, VariableNode):
        return {expr.name}
    if isinstance(expr, (AdditionNode, MaxNode, MinNode)):
        result: Set[str] = set()
        for operand in expr.operands:
            result |= _collect_variable_names(operand)
        return result
    if isinstance(expr, UnaryOperatorNode):
        return _collect_variable_names(expr.operand)
    if isinstance(expr, BinaryOperatorNode):
        return _collect_variable_names(expr.left) | _collect_variable_names(expr.right)
    return set()


def _check_oob_constraint_ids(
    oob_processing: OutOfBoundsProcessingConfig,
    model: "Model",
    model_config_id: str,
    errors: List[str],
) -> None:
    for constraint_config in oob_processing.constraints:
        if (
            constraint_config.id not in model.constraints
            and constraint_config.id not in model.binding_constraints
        ):
            errors.append(
                f"Out-of-bounds constraint '{constraint_config.id}' not found in model '{model_config_id}'"
            )


def _check_id_existence(
    decomposition: ModelDecompositionConfig,
    model: "Model",
    model_config_id: str,
    errors: List[str],
) -> None:
    for variable_config in decomposition.variables:
        if variable_config.id not in model.variables:
            errors.append(
                f"Variable '{variable_config.id}' not found in model '{model_config_id}'"
            )
    for constraint_config in decomposition.constraints:
        if (
            constraint_config.id not in model.constraints
            and constraint_config.id not in model.binding_constraints
        ):
            errors.append(
                f"Constraint '{constraint_config.id}' not found in model '{model_config_id}'"
            )
    obj_keys = set(model.objective_contributions or {})
    for obj_config in decomposition.objective_contributions:
        if obj_config.id not in obj_keys:
            errors.append(
                f"Objective-contribution '{obj_config.id}' not found in model '{model_config_id}'"
            )


def _check_master_variables_not_time_dependent(
    decomposition: ModelDecompositionConfig,
    model: "Model",
    model_config_id: str,
    errors: List[str],
) -> None:
    """Variables assigned to master or master-and-subproblems must not depend on time."""
    for variable_config in decomposition.variables:
        if (
            variable_config.location in _MASTER_LOCS
            and variable_config.id in model.variables
        ):
            if model.variables[variable_config.id].structure.time:
                errors.append(
                    f"Variable '{variable_config.id}' in model '{model_config_id}' is time-dependent "
                    f"but is assigned to '{variable_config.location.value}'; "
                    "master variables must not depend on time"
                )


def _check_master_constraints_use_master_variables(
    decomposition: ModelDecompositionConfig,
    model: "Model",
    model_config_id: str,
    errors: List[str],
) -> None:
    """Constraints in master must only reference variables in master or master-and-subproblems."""
    master_var_ids = {
        variable_config.id
        for variable_config in decomposition.variables
        if variable_config.location in _MASTER_LOCS
        and variable_config.id in model.variables
    }
    for constraint_config in decomposition.constraints:
        if constraint_config.location == ElementLocation.MASTER:
            constraint = model.constraints.get(
                constraint_config.id
            ) or model.binding_constraints.get(constraint_config.id)
            if constraint is not None:
                for var_name in sorted(
                    _collect_variable_names(constraint.expression) - master_var_ids
                ):
                    errors.append(
                        f"Constraint '{constraint_config.id}' in model '{model_config_id}' references variable '{var_name}' "
                        "which is not assigned to master or master-and-subproblems"
                    )


def _check_master_objectives_use_master_variables(
    decomposition: ModelDecompositionConfig,
    model: "Model",
    model_config_id: str,
    errors: List[str],
) -> None:
    """Objective contributions in master must only reference variables in master or master-and-subproblems."""
    master_var_ids = {
        variable_config.id
        for variable_config in decomposition.variables
        if variable_config.location in _MASTER_LOCS
        and variable_config.id in model.variables
    }
    obj_contribs = model.objective_contributions or {}
    for obj_config in decomposition.objective_contributions:
        if obj_config.location == ElementLocation.MASTER:
            expr = obj_contribs.get(obj_config.id)
            if expr is not None:
                for var_name in sorted(_collect_variable_names(expr) - master_var_ids):
                    errors.append(
                        f"Objective contribution '{obj_config.id}' in model '{model_config_id}' references variable '{var_name}' "
                        "which is not assigned to master or master-and-subproblems"
                    )


def validate_optim_config(
    config: OptimConfig,
    system: "System",
    scenario_builder: Optional["ScenarioBuilder"] = None,
) -> None:
    """Cross-validate optim-config entries against the resolved system.

    Performs the following checks:

    - Every model ID referenced in ``config.models`` exists in the system.
    - Master variables are time-independent.
    - Master constraints and objective contributions only reference variables
      assigned to ``master`` or ``master-and-subproblems``.
    - If ``scenario_builder`` is provided, every scenario index in
      ``config.scenario_scope.scenario_ids`` is defined for every scenario
      group in the builder.

    Raises ``ValueError`` listing all violations found.
    """
    models_in_system = {c.model.id: c.model for c in system.all_components}
    errors: List[str] = []

    if scenario_builder is not None:
        errors.extend(
            scenario_builder.validate_mc_scenarios(config.scenario_scope.scenario_ids)
        )

    for model_config in config.models:
        model = models_in_system.get(model_config.id)
        if model is None:
            errors.append(f"Model '{model_config.id}' not found in system")
        else:
            if model_config.model_decomposition is not None:
                decomposition = model_config.model_decomposition
                _check_id_existence(decomposition, model, model_config.id, errors)
                _check_master_variables_not_time_dependent(
                    decomposition, model, model_config.id, errors
                )
                _check_master_constraints_use_master_variables(
                    decomposition, model, model_config.id, errors
                )
                _check_master_objectives_use_master_variables(
                    decomposition, model, model_config.id, errors
                )
            if model_config.out_of_bounds_processing is not None:
                _check_oob_constraint_ids(
                    model_config.out_of_bounds_processing,
                    model,
                    model_config.id,
                    errors,
                )

    if errors:
        raise ValueError(
            f"Errors in optim config file:\n" + "\n".join(f"  - {e}" for e in errors)
        )
