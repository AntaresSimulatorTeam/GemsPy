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
from typing import Any, Dict, Iterable, Optional

from gems.expression import ExpressionNode
from gems.expression.degree import is_linear
from gems.expression.indexing import IndexingStructureProvider, compute_indexation
from gems.expression.indexing_structure import IndexingStructure
from gems.model.constraint import Constraint
from gems.model.parameter import Parameter
from gems.model.port import PortFieldDefinition, PortFieldId, PortType
from gems.model.variable import Variable


# TODO: Introduce bool_variable ?
def _make_structure_provider(
    parameters: Dict[str, Parameter],
    variables: Dict[str, Variable],
) -> IndexingStructureProvider:
    class Provider(IndexingStructureProvider):
        def get_parameter_structure(self, name: str) -> IndexingStructure:
            return parameters[name].structure

        def get_variable_structure(self, name: str) -> IndexingStructure:
            return variables[name].structure

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
        model.parameters, model.variables
    )
    objective_structure = compute_indexation(
        objective_contribution, data_structure_provider
    )

    if objective_structure != IndexingStructure(time=False, scenario=False):
        raise ValueError("Objective contribution should be a real-valued expression.")
    # TODO: We should also check that the number of instances is equal to 1, but this would require a linearization here, do not want to do that for now...
    return True


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

    def __post_init__(self) -> None:
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
    )
