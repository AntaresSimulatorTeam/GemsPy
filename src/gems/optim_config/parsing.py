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

import json
import re
import warnings
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from pydantic import (
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)
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
    from gems.study.scenario_builder import ScenarioBuilder
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


class HeuristicType(str, Enum):
    FAST = "fast"
    ACCURATE = "accurate"


class HeuristicBoundField(str, Enum):
    LOWER_BOUND = "lower-bound"
    UPPER_BOUND = "upper-bound"
    BOTH_BOUNDS = "both-bounds"


class HeuristicInputConfig(ModifiedBaseModel):
    id: str
    value: str


class HeuristicOutputConfig(ModifiedBaseModel):
    id: str
    variable: str
    field: HeuristicBoundField


class HeuristicConfig(ModifiedBaseModel):
    id: HeuristicType
    inputs: List[HeuristicInputConfig] = Field(default_factory=list)
    outputs: List[HeuristicOutputConfig] = Field(default_factory=list)


class ModelOptimConfig(ModifiedBaseModel):
    id: str
    model_decomposition: Optional[ModelDecompositionConfig] = None
    out_of_bounds_processing: Optional[OutOfBoundsProcessingConfig] = None
    heuristic: List[HeuristicConfig] = Field(default_factory=list)


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


def _expand_entries(entries: List[Union[int, str]]) -> Set[int]:
    """Expand a list of scenario specifiers into a set of 0-based indices.

    Each entry may be:
    - a non-negative integer (e.g. ``5``),
    - a string integer (e.g. ``"5"``), or
    - an inclusive range string (e.g. ``"0-9"``).

    Booleans are rejected upstream by the Pydantic field validator and will
    never reach this function.
    """
    result: Set[int] = set()
    for entry in entries:
        if isinstance(entry, int):
            if entry < 0:
                raise ValueError(f"Scenario index must be >= 0, got {entry}")
            result.add(entry)
        else:
            s = str(entry).strip()
            if re.fullmatch(r"\d+", s):
                val = int(s)
                result.add(val)
            else:
                match = re.fullmatch(r"(\d+)-(\d+)", s)
                if not match:
                    raise ValueError(
                        f"Invalid entry {entry!r}: expected an integer or a range 'a-b'"
                        " (e.g. '5' or '0-9')"
                    )
                a, b = int(match.group(1)), int(match.group(2))
                if a > b:
                    raise ValueError(f"Range start must be <= end, got {entry!r}")
                result.update(range(a, b + 1))
    return result


class ScenarioScopeConfig(ModifiedBaseModel):
    """Declares which Monte-Carlo scenarios to simulate.

    Two mutually exclusive ways to define the base scenario set:

    - **Inline** (``include``): a list of 0-based integers, string-integers,
      and/or ``"a-b"`` range strings.
    - **File** (``playlist_file``): path to a flat JSON array of 0-based
      integers, resolved relative to ``optim-config.yml`` by
      ``load_optim_config()``.

    ``exclude`` is optional and compatible with *both* forms.  It subtracts a
    set of scenarios from the base set using the same entry format.  Entries
    in ``exclude`` that are not in the base set are silently ignored (a
    ``UserWarning`` is emitted).

    ``include`` and ``playlist_file`` are mutually exclusive.
    ``exclude`` without any base set raises ``ValueError``.

    All indices are 0-based, consistent with
    ``modeler-scenariobuilder.dat``.

    The resolved list is computed lazily on first access to
    ``scenario_ids`` and cached for the lifetime of the object.
    ``load_optim_config()`` triggers eager resolution so that file I/O
    errors surface at load time.
    """

    include: Optional[List[Union[int, str]]] = None
    exclude: Optional[List[Union[int, str]]] = None
    playlist_file: Optional[Path] = None

    _scenario_ids: Optional[List[int]] = PrivateAttr(default=None)

    @field_validator("include", "exclude", mode="before")
    @classmethod
    def _reject_booleans(cls, v: object) -> object:
        if isinstance(v, list):
            for item in v:
                if isinstance(item, bool):
                    raise ValueError(
                        f"Scenario index must be an integer, got boolean {item!r}"
                    )
        return v

    @model_validator(mode="after")
    def _check_constraints(self) -> "ScenarioScopeConfig":
        has_inline = self.include is not None
        has_file = self.playlist_file is not None
        if has_inline and has_file:
            raise ValueError("'include' and 'playlist-file' are mutually exclusive")
        if self.exclude is not None and not has_inline and not has_file:
            raise ValueError("'exclude' requires 'include' or 'playlist-file'")
        return self

    @property
    def scenario_ids(self) -> List[int]:
        if self._scenario_ids is None:
            self._scenario_ids = self._compute_scenario_ids()
        return self._scenario_ids

    def _compute_scenario_ids(self) -> List[int]:
        if self.playlist_file is not None:
            included = self._load_playlist()
        elif self.include is not None:
            included = _expand_entries(self.include)
        else:
            return [0]

        if self.exclude is not None:
            excluded = _expand_entries(self.exclude)
            orphans = excluded - included
            if orphans:
                warnings.warn(
                    f"Excluded scenario indices {sorted(orphans)} "
                    "are not in the base set and have no effect",
                    UserWarning,
                    stacklevel=2,
                )
            included -= excluded

        return sorted(included)

    def _load_playlist(self) -> Set[int]:
        try:
            with self.playlist_file.open() as f:  # type: ignore[union-attr]
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Playlist file not found: '{self.playlist_file}'")
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in playlist file '{self.playlist_file}': {exc}"
            ) from exc
        if not isinstance(data, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in data
        ):
            raise ValueError(
                f"'{self.playlist_file}' must contain a flat JSON array of integers"
            )
        if any(x < 0 for x in data):
            raise ValueError(
                f"'{self.playlist_file}': all scenario indices must be >= 0"
            )
        return set(data)


class OptimConfig(ModifiedBaseModel):
    time_scope: TimeScopeConfig = Field(default_factory=TimeScopeConfig)
    solver_options: SolverOptionsConfig = Field(default_factory=SolverOptionsConfig)
    scenario_scope: ScenarioScopeConfig = Field(default_factory=ScenarioScopeConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    models: List[ModelOptimConfig] = Field(default_factory=list)


def load_optim_config(config_path: Path) -> Optional[OptimConfig]:
    """Load and fully resolve an ``optim-config.yml`` file.

    Returns ``None`` if the file does not exist.
    Raises ``ValueError`` on any parsing, validation, or playlist I/O failure.

    Beyond plain YAML parsing, this function:

    - Resolves a relative ``playlist-file`` path against the directory that
      contains the config file.
    - Eagerly populates ``scenario_ids`` so that playlist file errors surface
      immediately rather than at first use.
    """
    if not config_path.exists():
        return None
    try:
        with config_path.open() as config_file:
            config = OptimConfig.model_validate(safe_load(config_file))
    except ValidationError as e:
        raise ValueError(f"Invalid {config_path.stem}: {e}")

    pf = config.scenario_scope.playlist_file
    if pf is not None and not pf.is_absolute():
        config.scenario_scope.playlist_file = config_path.parent / pf

    # Resolve and cache scenario_ids eagerly so that the playlist file is read
    # exactly once and any I/O or format errors surface at load time.
    _ = config.scenario_scope.scenario_ids
    return config


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
