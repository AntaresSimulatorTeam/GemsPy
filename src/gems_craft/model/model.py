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

import dataclasses
import itertools
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

from gems_craft.expression import ExpressionNode
from gems_craft.expression.degree import is_constant, is_linear
from gems_craft.expression.expression import (
    AllTimeSumNode,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    RelativeTimeNode,
    TimeEvalNode,
    TimeSumNode,
)
from gems_craft.expression.indexing import IndexingStructureProvider, compute_indexation
from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.expression.print import print_expr
from gems_craft.model.constraint import Constraint
from gems_craft.model.parameter import Parameter
from gems_craft.model.port import PortFieldDefinition, PortFieldId, PortType
from gems_craft.model.variable import Variable


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
            try:
                constraint_structures[cname] = compute_indexation(c.expression, base)
            except ValueError:
                # Constraints containing unresolved port fields (sum_connections)
                # cannot be indexed before port resolution; fall back to the most
                # general structure so callers can still proceed.
                constraint_structures[cname] = IndexingStructure(
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

    This implements the iso-format behaviour of the GEMS format (Issue #76).

    TODO: This auto-wrapping is a temporary compatibility shim. Once the expec()
    operator is required in objective contributions, this function should be
    removed and authors should be required to write expec() explicitly.
    """

    provider = _make_structure_provider(parameters, variables)
    result: Dict[str, ExpressionNode] = {}
    for contrib_id, expr in contributions.items():
        structure = compute_indexation(expr, provider)
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


def _iter_nodes(
    node: ExpressionNode, skip: Tuple[Type[ExpressionNode], ...] = ()
) -> Iterator[ExpressionNode]:
    """*node* and every node below it, without going below a *skip* node."""
    yield node
    if isinstance(node, skip):
        return
    for f in dataclasses.fields(node):
        value = getattr(node, f.name)
        for child in value if isinstance(value, list) else [value]:
            if isinstance(child, ExpressionNode):
                yield from _iter_nodes(child, skip)


def _model_expressions(model: "Model") -> Iterator[Tuple[str, ExpressionNode]]:
    """Every expression of *model*, with a description of where it is."""
    for c in model.get_all_constraints():
        for expr in (c.expression, c.lower_bound, c.upper_bound):
            yield f"constraint '{c.name}'", expr
    for name, expr in (model.objective_contributions or {}).items():
        yield f"objective contribution '{name}'", expr
    for name, expr in (model.extra_outputs or {}).items():
        yield f"extra output '{name}'", expr
    for port_field, definition in model.port_fields_definitions.items():
        yield (
            f"port field definition '{port_field.port_name}.{port_field.field_name}'",
            definition.definition,
        )


def _check_time_sum_bounds(model: "Model") -> None:
    """A time sum bound (an absolute bound, or the offset of a bound relative
    to ``t``) must be time-independent and contain no variable. This depends on
    the declared structure only, not on the data."""
    provider = _make_structure_provider(model.parameters, model.variables)
    for where, expr in _model_expressions(model):
        for node in _iter_nodes(expr):
            if not isinstance(node, TimeSumNode):
                continue
            for bound in (node.from_time, node.to_time):
                value = bound.offset if isinstance(bound, RelativeTimeNode) else bound
                _check_time_sum_bound(model.id, where, bound, value, provider)


def _check_time_sum_bound(
    model_id: str,
    where: str,
    bound: ExpressionNode,
    value: ExpressionNode,
    provider: IndexingStructureProvider,
) -> None:
    """Check *value*: the bound itself, or the offset of a relative *bound*."""
    prefix = f"Model '{model_id}', {where}: the time sum bound '{print_expr(bound)}'"
    port_field_types = (PortFieldNode, PortFieldAggregatorNode)
    if any(isinstance(n, port_field_types) for n in _iter_nodes(value)):
        raise ValueError(f"{prefix} cannot contain a port field.")
    if not is_constant(value):
        raise ValueError(f"{prefix} cannot contain a variable.")
    if not compute_indexation(value, provider).time:
        return
    # Parameters that make the bound time-dependent: those outside an operator
    # that removes the time dimension (e.g. d[0] or sum(d)).
    names = sorted(
        {
            n.name
            for n in _iter_nodes(
                value, skip=(TimeEvalNode, AllTimeSumNode, TimeSumNode)
            )
            if isinstance(n, ParameterNode)
            and provider.get_parameter_structure(n.name).time
        }
    )
    hint = (
        " Declare " + ", ".join(f"'{name}'" for name in names) + " with "
        "`time-dependent: false`."
        if names
        else ""
    )
    raise ValueError(f"{prefix} depends on time, but must be time-independent.{hint}")


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

    def __post_init__(self) -> None:
        _check_time_sum_bounds(self)
        # Validate each contribution if present
        if self.objective_contributions:
            for expr in self.objective_contributions.values():
                _is_objective_contribution_valid(self, expr)

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
) -> Model:
    """
    Utility method to create Models from relaxed arguments
    """
    # Build dicts upfront so we can inspect indexing structure before Model construction.
    params_dict = {p.name: p for p in parameters} if parameters else {}
    vars_dict = {v.name: v for v in variables} if variables else {}

    # Auto-wrap any objective contribution that has a residual scenario dimension
    # without an explicit expec() (Issue #76, iso-format behaviour).
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
    )
