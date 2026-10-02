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

"""
The model module defines the data model for user-defined models.
A model allows to define the behaviour for components, by
defining parameters, variables, and equations.
"""

import itertools
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from gems_craft.expression import ExpressionNode
from gems_craft.expression.degree import is_linear
from gems_craft.expression.indexing import (
    IndexingStructureProvider,
    IndexingUsageError,
    UnresolvedPortFieldError,
    compute_indexation,
)
from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.model.constraint import Constraint
from gems_craft.model.parameter import Parameter
from gems_craft.model.port import PortFieldDefinition, PortFieldId, PortType
from gems_craft.model.variable import Variable


def _safe_indexation(
    expr: ExpressionNode, provider: IndexingStructureProvider, context: str
) -> Optional[IndexingStructure]:
    """
    Computes the indexation of `expr`, which also validates time and set index
    usage (errors name `context`). Returns None if `expr` still contains an
    unresolved port field (sum_connections): the check is then deferred until
    ports are resolved.
    """
    try:
        return compute_indexation(expr, provider)
    except IndexingUsageError as e:
        raise ValueError(f"{e} (in {context})") from e
    except UnresolvedPortFieldError:
        return None


# TODO: Introduce bool_variable ?
def _make_structure_provider(
    parameters: Dict[str, Parameter],
    variables: Dict[str, Variable],
    constraints: Optional[Dict[str, Constraint]] = None,
) -> IndexingStructureProvider:
    # Pre-compute constraint structures using a params/vars-only base provider.
    # Constraint expressions cannot contain dual()/reduced_cost(), so the base
    # provider's get_constraint_structure is never invoked during this step.
    constraint_structures: Dict[str, IndexingStructure] = {}
    if constraints:

        class _BaseProvider(IndexingStructureProvider):
            def get_parameter_structure(self, name: str) -> IndexingStructure:
                return parameters[name].structure

            def get_variable_structure(self, name: str) -> IndexingStructure:
                return variables[name].structure

            def get_constraint_structure(self, name: str) -> IndexingStructure:
                raise NotImplementedError(
                    f"Constraint structure for '{name}' not available at this stage."
                )

        base = _BaseProvider()
        for cname, c in constraints.items():
            structure = _safe_indexation(c.expression, base, f"constraint '{cname}'")
            # Constraints containing unresolved port fields (sum_connections)
            # cannot be indexed before port resolution; fall back to the most
            # general structure so callers can still proceed.
            constraint_structures[cname] = structure or IndexingStructure(
                time=True, scenario=True
            )

    class Provider(IndexingStructureProvider):
        def get_parameter_structure(self, name: str) -> IndexingStructure:
            return parameters[name].structure

        def get_variable_structure(self, name: str) -> IndexingStructure:
            return variables[name].structure

        def get_constraint_structure(self, name: str) -> IndexingStructure:
            return constraint_structures[name]

    return Provider()


def _normalize_objective_contributions(
    contributions: Dict[str, ExpressionNode],
    parameters: Dict[str, Parameter],
    variables: Dict[str, Variable],
) -> Dict[str, ExpressionNode]:
    """
    Tolerate absence of expec() in objective contributions that carry a residual
    scenario dimension (IndexingStructure(time=False, scenario=True)).

    Such contributions are automatically wrapped with expec(), applying
    expectation (average-over-scenarios) semantics, and a UserWarning is emitted
    so authors can add expec() explicitly at their convenience.

    Contributions that are already fully scalar, or already wrapped in expec(),
    are returned unchanged with no warning.

    This implements the iso-format behaviour of Antares Simulator v10.0.0 (Issue #76).

    TODO: This auto-wrapping is a temporary compatibility shim. Once Antares Simulator
    natively supports the expec() operator in objective contributions, this function
    should be removed and authors should be required to write expec() explicitly.
    """

    provider = _make_structure_provider(parameters, variables)
    result: Dict[str, ExpressionNode] = {}
    for contrib_id, expr in contributions.items():
        structure = _safe_indexation(
            expr, provider, f"objective contribution '{contrib_id}'"
        )
        if structure is None:
            raise ValueError(
                f"Objective contribution '{contrib_id}' contains an unresolved port field."
            )
        if structure == IndexingStructure(time=False, scenario=True):
            warnings.warn(
                f"Objective contribution '{contrib_id}' has a scenario dimension "
                "but no explicit expec() operator. "
                "Expectation semantics (average over scenarios) are applied "
                "automatically. Add expec() explicitly to suppress this warning.",
                UserWarning,
                stacklevel=4,
            )
            expr = expr.expec()
        result[contrib_id] = expr
    return result


def _is_objective_contribution_valid(
    model: "Model", objective_contribution: ExpressionNode
) -> bool:
    if not is_linear(objective_contribution):
        raise ValueError("Objective contribution must be a linear expression.")

    data_structure_provider = _make_structure_provider(
        model.parameters,
        model.variables,
        {**model.constraints, **model.binding_constraints},
    )
    objective_structure = compute_indexation(
        objective_contribution, data_structure_provider
    )

    if objective_structure != IndexingStructure(time=False, scenario=False):
        raise ValueError("Objective contribution should be a real-valued expression.")
    # TODO: We should also check that the number of instances is equal to 1, but this would require a linearization here, do not want to do that for now...
    return True


def _check_bound_set_consistency(
    bound_expr: ExpressionNode,
    var: Variable,
    provider: IndexingStructureProvider,
    kind: str,
) -> None:
    structure = _safe_indexation(
        bound_expr, provider, f"{kind.lower()} of variable '{var.name}'"
    )
    if structure is None:
        return
    stray = structure.sets - var.structure.sets
    if stray:
        raise ValueError(
            f"{kind} of variable '{var.name}' is indexed by set(s) "
            f"{sorted(stray)} not declared in '{var.name}''s own indexed_by."
        )


def _check_local_set_crosses_port(
    definition: PortFieldDefinition,
    local_sets: FrozenSet[str],
    provider: IndexingStructureProvider,
) -> None:
    port_field = f"{definition.port_field.port_name}.{definition.port_field.field_name}"
    structure = _safe_indexation(
        definition.definition, provider, f"port field definition '{port_field}'"
    )
    if structure is None:
        return
    crossing = structure.sets & local_sets
    if crossing:
        raise ValueError(
            f"Port field definition for '{port_field}' is still indexed by local "
            f"set(s) {sorted(crossing)}; wrap it in sum_over(...) before it "
            "can cross a port."
        )


def _check_model_set_indexing(model: "Model") -> None:
    provider = _make_structure_provider(
        model.parameters,
        model.variables,
        {**model.constraints, **model.binding_constraints},
    )

    for c in model.get_all_constraints():
        _safe_indexation(c.expression, provider, f"constraint '{c.name}'")

    for oid, expr in (model.objective_contributions or {}).items():
        _safe_indexation(expr, provider, f"objective contribution '{oid}'")

    for eo_id, eo_expr in (model.extra_outputs or {}).items():
        _safe_indexation(eo_expr, provider, f"extra-output '{eo_id}'")

    for var in model.variables.values():
        if var.lower_bound is not None:
            _check_bound_set_consistency(var.lower_bound, var, provider, "Lower bound")
        if var.upper_bound is not None:
            _check_bound_set_consistency(var.upper_bound, var, provider, "Upper bound")

    for definition in model.port_fields_definitions.values():
        _check_local_set_crosses_port(definition, model.local_sets, provider)


@dataclass(frozen=True)
class ModelPort:
    """
    Instance of a port as a model member.

    A model may carry multiple ports of the same type.
    For example, the 2 ports at line extremities.
    """

    port_type: PortType
    port_name: str

    def replicate(self, /, **changes: Any) -> "ModelPort":
        return replace(self, **changes)


@dataclass(frozen=True)
class Model:
    """
    Defines a model that can be referenced by actual components.
    A model defines the behaviour of those components.
    """

    id: str
    constraints: Dict[str, Constraint] = field(default_factory=dict)
    binding_constraints: Dict[str, Constraint] = field(default_factory=dict)
    inter_block_dyn: bool = False
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    variables: Dict[str, Variable] = field(default_factory=dict)
    objective_contributions: Optional[Dict[str, ExpressionNode]] = None
    ports: Dict[str, ModelPort] = field(default_factory=dict)
    port_fields_definitions: Dict[PortFieldId, PortFieldDefinition] = field(
        default_factory=dict
    )
    extra_outputs: Optional[Dict[str, ExpressionNode]] = None
    properties: List[str] = field(default_factory=list)
    local_sets: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Validate each contribution if present
        if self.objective_contributions:
            for expr in self.objective_contributions.values():
                _is_objective_contribution_valid(self, expr)

        _check_model_set_indexing(self)

        for definition in self.port_fields_definitions.values():
            port_name = definition.port_field.port_name
            port_field = definition.port_field.field_name
            port = self.ports.get(port_name, None)
            if port is None:
                raise ValueError(f"Invalid port in port field definition: {port_name}")
            if port_field not in [f.name for f in port.port_type.fields]:
                raise ValueError(
                    f"Invalid port field in port field definition: {port_field}"
                )

    def get_all_constraints(self) -> Iterable[Constraint]:
        """
        Get binding constraints and inner constraints altogether.
        """
        return itertools.chain(
            self.binding_constraints.values(), self.constraints.values()
        )

    def replicate(self, /, **changes: Any) -> "Model":
        # Shallow copy
        return replace(self, **changes)


def model(
    id: str,
    constraints: Optional[Iterable[Constraint]] = None,
    binding_constraints: Optional[Iterable[Constraint]] = None,
    parameters: Optional[Iterable[Parameter]] = None,
    variables: Optional[Iterable[Variable]] = None,
    objective_contributions: Optional[Dict[str, ExpressionNode]] = None,
    inter_block_dyn: bool = False,
    ports: Optional[Iterable[ModelPort]] = None,
    port_fields_definitions: Optional[Iterable[PortFieldDefinition]] = None,
    extra_outputs: Optional[Dict[str, ExpressionNode]] = None,
    properties: Optional[Iterable[str]] = None,
    local_sets: Optional[Iterable[str]] = None,
) -> Model:
    """
    Utility method to create Models from relaxed arguments
    """
    # Build dicts upfront so we can inspect indexing structure before Model construction.
    params_dict = {p.name: p for p in parameters} if parameters else {}
    vars_dict = {v.name: v for v in variables} if variables else {}

    # Auto-wrap any objective contribution that has a residual scenario dimension
    # without an explicit expec() (Issue #76 / Antares Simulator v10.0.0 iso-format).
    if objective_contributions:
        objective_contributions = _normalize_objective_contributions(
            objective_contributions, params_dict, vars_dict
        )

    existing_port_names = {}
    if ports:
        for port in ports:
            port_name = port.port_name
            if port_name not in existing_port_names:
                existing_port_names[port_name] = port
            else:
                raise ValueError(
                    f"2 ports have the same name inside the model, it's not authorized : {port_name}"
                )
    return Model(
        id=id,
        constraints={c.name: c for c in constraints} if constraints else {},
        binding_constraints=(
            {c.name: c for c in binding_constraints} if binding_constraints else {}
        ),
        parameters=params_dict,
        variables=vars_dict,
        objective_contributions=objective_contributions,
        inter_block_dyn=inter_block_dyn,
        ports=existing_port_names,
        port_fields_definitions=(
            {d.port_field: d for d in port_fields_definitions}
            if port_fields_definitions
            else {}
        ),
        extra_outputs=extra_outputs,
        properties=list(properties) if properties else [],
        local_sets=frozenset(local_sets) if local_sets else frozenset(),
    )
