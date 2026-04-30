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

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from pydantic import Field, ValidationError, model_validator
from yaml import safe_load

from gems.expression.expression import (
    AdditionNode,
    BinaryOperatorNode,
    ExpressionNode,
    MaxNode,
    MinNode,
    UnaryOperatorNode,
    VariableNode,
)
from gems.utils import ModifiedBaseModel

if TYPE_CHECKING:
    from gems.model.model import Model
    from gems.study.system import System


class ElementLocation(str, Enum):
    MASTER = "master"
    SUBPROBLEMS = "subproblems"
    MASTER_AND_SUBPROBLEMS = "master-and-subproblems"


class ElementLocationConfig(ModifiedBaseModel):
    id: str
    location: ElementLocation


class ModelDecompositionConfig(ModifiedBaseModel):
    variables: List[ElementLocationConfig] = Field(default_factory=list)
    constraints: List[ElementLocationConfig] = Field(default_factory=list)
    objective_contributions: List[ElementLocationConfig] = Field(default_factory=list)


class OutOfBoundsMode(str, Enum):
    CYCLIC = "cyclic"
    DROP = "drop"


class OutOfBoundsConstraintConfig(ModifiedBaseModel):
    id: str
    mode: OutOfBoundsMode


class OutOfBoundsProcessingConfig(ModifiedBaseModel):
    constraints: List[OutOfBoundsConstraintConfig] = Field(default_factory=list)


class ModelOptimConfig(ModifiedBaseModel):
    id: str
    model_decomposition: Optional[ModelDecompositionConfig] = None
    out_of_bounds_processing: Optional[OutOfBoundsProcessingConfig] = None


class ResolutionMode(str, Enum):
    FRONTAL = "frontal"
    SEQUENTIAL_SUBPROBLEMS = "sequential-subproblems"
    PARALLEL_SUBPROBLEMS = "parallel-subproblems"
    BENDERS_DECOMPOSITION = "benders-decomposition"


class ResolutionConfig(ModifiedBaseModel):
    mode: ResolutionMode = ResolutionMode.FRONTAL
    block_length: Optional[int] = None
    block_overlap: int = 0

    @model_validator(mode="after")
    def _block_length_required_for_windowed_modes(self) -> "ResolutionConfig":
        windowed = {
            ResolutionMode.SEQUENTIAL_SUBPROBLEMS,
            ResolutionMode.PARALLEL_SUBPROBLEMS,
        }
        if self.mode in windowed and self.block_length is None:
            raise ValueError(f"'block_length' is required for mode '{self.mode.value}'")
        return self


class TimeScopeConfig(ModifiedBaseModel):
    first_time_step: int = 0
    last_time_step: int = 0


class SolverOptionsConfig(ModifiedBaseModel):
    name: str = "highs"
    logs: bool = False
    parameters: str = ""

    def parsed_parameters(self) -> Dict[str, Any]:
        """Parse 'KEY VALUE KEY2 VALUE2 ...' into a dict with numeric coercion."""
        if not self.parameters.strip():
            return {}
        tokens = self.parameters.split()
        if len(tokens) % 2 != 0:
            raise ValueError(
                f"parameters must be space-separated key-value pairs, got: {self.parameters!r}"
            )
        result: Dict[str, Any] = {}
        for i in range(0, len(tokens), 2):
            key, raw = tokens[i], tokens[i + 1]
            try:
                result[key] = int(raw)
            except ValueError:
                try:
                    result[key] = float(raw)
                except ValueError:
                    result[key] = raw
        return result


class ScenarioScopeConfig(ModifiedBaseModel):
    nb_scenarios: int = 1

    @property
    def scenario_ids(self) -> List[int]:
        return list(range(self.nb_scenarios))


class OptimConfig(ModifiedBaseModel):
    time_scope: TimeScopeConfig = Field(default_factory=TimeScopeConfig)
    solver_options: SolverOptionsConfig = Field(default_factory=SolverOptionsConfig)
    scenario_scope: ScenarioScopeConfig = Field(default_factory=ScenarioScopeConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    models: List[ModelOptimConfig] = Field(default_factory=list)


def load_optim_config(config_path: Path) -> Optional[OptimConfig]:
    """Load optim-config.yml from the same directory as components_path.

    Returns None if the file does not exist.
    Raises ValueError on parsing or validation failure.
    """
    if not config_path.exists():
        return None
    try:
        with config_path.open() as config_file:
            return OptimConfig.model_validate(safe_load(config_file))
    except ValidationError as e:
        raise ValueError(f"Invalid {config_path.stem}: {e}")


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


def validate_optim_config(config: OptimConfig, system: "System") -> None:
    """Cross-validate optim-config entries against the resolved system.

    Checks that every referenced ID exists, that master variables do not
    depend on time, and that master constraints and objectives only reference
    variables assigned to master or master-and-subproblems.
    Raises ValueError listing all violations.
    """
    models_in_system = {c.model.id: c.model for c in system.all_components}
    errors: List[str] = []

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
