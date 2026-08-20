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

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

from gems_craft.expression.expression import (
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
    HeuristicConfig,
    ModelDecompositionConfig,
    ModelElementAccessType,
    OptimConfig,
    OutOfBoundsProcessingConfig,
    ResolutionMode,
)
from gems_craft.study.parsing import IntegerStrategyId
from gems_craft.study.system import Component

if TYPE_CHECKING:
    from gems_craft.model.model import Model
    from gems_craft.model.parameter import Parameter
    from gems_craft.model.variable import Variable
    from gems_craft.study.scenario_builder import ScenarioBuilder
    from gems_craft.study.system import System

_MASTER_LOCS: Set[ElementLocation] = {
    ElementLocation.MASTER,
    ElementLocation.MASTER_AND_SUBPROBLEMS,
}

_HEURISTIC_ELEMENT_TIME_DEPENDENCE: Dict[str, Optional[bool]] = {
    # True: must vary per timestep. False: must be a single constant value for the whole
    # block. None: either is accepted, the heuristic function broadcasts a constant.
    "min_up_duration": False,
    "min_down_duration": False,
    "min_power_per_unit": False,
    "max_power_per_unit": False,
    "num_units_on_opt": True,
    "generation_power": True,
    "minimum_num_units_on": True,
    "minimum_generation_power": True,
    "maximum_generation_power": True,
    "num_units_max": None,
    "cluster_max_generation": None,
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


def get_heuristic_config_map(
    config: "OptimConfig",
) -> Dict[Tuple[str, str], HeuristicConfig]:
    """Map every declared heuristic to its config, keyed by (model_id, heuristic_id)."""
    return {
        (model_cfg.id, heuristic_cfg.id.value): heuristic_cfg
        for model_cfg in config.models
        for heuristic_cfg in (model_cfg.heuristics or [])
    }


def _check_heuristic_ids_declared(
    config: "OptimConfig",
    system: "System",
    errors: List[str],
) -> None:
    """Check that every HEURISTIC component references a declared heuristic in optim-config."""
    declared = get_heuristic_config_map(config)

    for comp in system.all_components:
        if comp.integer_strategy.id != IntegerStrategyId.HEURISTIC:
            continue
        assert comp.integer_strategy.heuristic_id is not None
        key = (comp.model.id, comp.integer_strategy.heuristic_id.value)
        if key not in declared:
            errors.append(
                f"Component '{comp.id}' references heuristic "
                f"'{comp.integer_strategy.heuristic_id.value}' on model '{comp.model.id}', "
                f"but this heuristic is not declared in optim-config."
            )


def _check_heuristic_elements(
    heuristic_config: HeuristicConfig,
    model: "Model",
    errors: List[str],
) -> None:
    """Every heuristic input/output id must exist in the model — as a parameter or a
    variable depending on its declared access type — and have the time-dependence the
    heuristic function expects."""
    for element_config in [*heuristic_config.inputs, *heuristic_config.outputs]:
        declared: Optional[Union["Parameter", "Variable"]]
        if element_config.type == ModelElementAccessType.PARAMETER:
            declared = model.parameters.get(element_config.id)
        else:
            declared = model.variables.get(element_config.id)

        if declared is None:
            errors.append(
                f"Heuristic '{heuristic_config.id.value}' in model '{model.id}': "
                f"'{element_config.id}' bound to heuristic element "
                f"'{element_config.heuristic_element}' not found in model."
            )
            continue

        expected = _HEURISTIC_ELEMENT_TIME_DEPENDENCE[element_config.heuristic_element]
        if expected is not None and declared.structure.time != expected:
            errors.append(
                f"Heuristic '{heuristic_config.id.value}' in model '{model.id}': "
                f"'{element_config.id}' bound to heuristic element "
                f"'{element_config.heuristic_element}' must be "
                f"'time-dependent:{expected}'."
            )


def _check_no_heuristic_with_benders(
    system: "System",
    errors: List[str],
) -> None:
    """Check that no component uses integer-strategy 'heuristic' with Benders decomposition."""
    errors.extend(
        f"Component '{comp.id}' uses integer-strategy 'heuristic', "
        f"which is incompatible with Benders decomposition."
        for comp in system.all_components
        if comp.integer_strategy.id == IntegerStrategyId.HEURISTIC
    )


def _check_no_integer_variables_in_subproblems(
    config: "OptimConfig",
    system: "System",
    errors: List[str],
) -> None:
    """Check that no integer or binary variable is assigned to Benders subproblems."""
    from gems_craft.model.common import ValueType

    subproblem_locs = {
        ElementLocation.SUBPROBLEMS,
        ElementLocation.MASTER_AND_SUBPROBLEMS,
    }

    model_components: Dict[str, List[Component]] = {}
    for c in system.all_components:
        model_components.setdefault(c.model.id, []).append(c)

    explicit_locs_by_model: Dict[str, Dict[str, ElementLocation]] = {}
    for model_config in config.models:
        if model_config.model_decomposition is not None:
            explicit_locs_by_model[model_config.id] = {
                variable_config.id: variable_config.location
                for variable_config in model_config.model_decomposition.variables
            }

    for model_id, comps in model_components.items():
        if any(c.integer_strategy.id == IntegerStrategyId.EXACT for c in comps):
            model = comps[0].model
            explicit_locs = explicit_locs_by_model.get(model_id, {})

            for var_name, var in model.variables.items():
                if var.data_type in (ValueType.INTEGER, ValueType.BINARY):
                    if (
                        explicit_locs.get(var_name, ElementLocation.SUBPROBLEMS)
                        in subproblem_locs
                    ):
                        errors.append(
                            f"Integer variable '{var_name}' of model '{model_id}' "
                            f"is assigned to subproblems, which is forbidden in Benders "
                            f"decomposition. Consider using a continuous variable or changing the resolution mode."
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
    - Heuristic inputs/outputs reference existing model parameters/variables with the
      time-dependence the heuristic function expects.
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
            for heuristic_config in model_config.heuristics:
                _check_heuristic_elements(heuristic_config, model, errors)

    _check_heuristic_ids_declared(config, system, errors)

    if config.resolution.mode == ResolutionMode.BENDERS_DECOMPOSITION:
        _check_no_heuristic_with_benders(system, errors)
        _check_no_integer_variables_in_subproblems(config, system, errors)

    if errors:
        raise ValueError(
            f"Errors in optim config file:\n" + "\n".join(f"  - {e}" for e in errors)
        )
