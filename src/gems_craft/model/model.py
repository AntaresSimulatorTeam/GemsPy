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
from gems_craft.expression.expression import (
    AbsNode,
    AdditionNode,
    AllTimeSumNode,
    CeilNode,
    ComparisonNode,
    DivisionNode,
    DualNode,
    FloorNode,
    LiteralNode,
    LowerBoundNode,
    MaxNode,
    MinNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ReducedCostNode,
    RoundNode,
    ScenarioOperatorNode,
    SetIndexNode,
    SumOverNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
    UpperBoundNode,
    VariableNode,
)
from gems_craft.expression.indexing import IndexingStructureProvider, compute_indexation
from gems_craft.expression.indexing_structure import IndexingStructure
from gems_craft.expression.visitor import ExpressionVisitor, visit
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


class _DimensionExistenceVisitor(ExpressionVisitor[None]):
    """
    Raises if an expression indexes a dimension that the indexed sub-expression
    does not actually vary over:
      - a custom set, e.g. `X[fuel]` on an `X` not `indexed_by: [fuel]`;
      - a time index/shift, absolute or relative (`X[5]`/`TimeEvalNode`,
        `X[t+1]`/`X.shift(...)`/`TimeShiftNode`), on a non-time-dependent `X`.
    """

    def __init__(self, provider: IndexingStructureProvider, context: str) -> None:
        self._provider = provider
        self._context = context

    def _structure_of(self, expr: ExpressionNode) -> Optional[IndexingStructure]:
        try:
            return compute_indexation(expr, self._provider)
        except ValueError:
            # Contains an unresolved port field (sum_connections); dimension
            # existence for it can only be checked once ports are resolved.
            return None

    def literal(self, node: LiteralNode) -> None:
        pass

    def negation(self, node: NegationNode) -> None:
        visit(node.operand, self)

    def addition(self, node: AdditionNode) -> None:
        for o in node.operands:
            visit(o, self)

    def multiplication(self, node: MultiplicationNode) -> None:
        visit(node.left, self)
        visit(node.right, self)

    def division(self, node: DivisionNode) -> None:
        visit(node.left, self)
        visit(node.right, self)

    def comparison(self, node: ComparisonNode) -> None:
        visit(node.left, self)
        visit(node.right, self)

    def variable(self, node: VariableNode) -> None:
        pass

    def parameter(self, node: ParameterNode) -> None:
        pass

    def _check_time_dependent(self, operand: ExpressionNode, kind: str) -> None:
        structure = self._structure_of(operand)
        if structure is not None and not structure.time:
            raise ValueError(
                f"Time {kind} used in {self._context}, but the indexed "
                "expression is not time-dependent."
            )

    def time_shift(self, node: TimeShiftNode) -> None:
        self._check_time_dependent(node.operand, "shift")
        visit(node.operand, self)

    def time_eval(self, node: TimeEvalNode) -> None:
        self._check_time_dependent(node.operand, "index")
        visit(node.operand, self)

    def time_sum(self, node: TimeSumNode) -> None:
        visit(node.operand, self)

    def all_time_sum(self, node: AllTimeSumNode) -> None:
        visit(node.operand, self)

    def set_index(self, node: SetIndexNode) -> None:
        structure = self._structure_of(node.operand)
        if structure is not None and node.set_id not in structure.sets:
            raise ValueError(
                f"'{node.set_id}' index used in {self._context}, but the "
                f"indexed expression is not indexed by '{node.set_id}'."
            )
        visit(node.operand, self)

    def sum_over(self, node: SumOverNode) -> None:
        visit(node.operand, self)

    def scenario_operator(self, node: ScenarioOperatorNode) -> None:
        visit(node.operand, self)

    def port_field(self, node: PortFieldNode) -> None:
        pass

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> None:
        pass

    def floor(self, node: FloorNode) -> None:
        visit(node.operand, self)

    def ceil(self, node: CeilNode) -> None:
        visit(node.operand, self)

    def abs(self, node: AbsNode) -> None:
        visit(node.operand, self)

    def round(self, node: RoundNode) -> None:
        visit(node.operand, self)

    def maximum(self, node: MaxNode) -> None:
        for o in node.operands:
            visit(o, self)

    def minimum(self, node: MinNode) -> None:
        for o in node.operands:
            visit(o, self)

    def dual(self, node: DualNode) -> None:
        pass

    def reduced_cost(self, node: ReducedCostNode) -> None:
        pass

    def lower_bound(self, node: LowerBoundNode) -> None:
        pass

    def upper_bound(self, node: UpperBoundNode) -> None:
        pass


def _check_dimension_existence(
    expr: ExpressionNode, provider: IndexingStructureProvider, context: str
) -> None:
    visit(expr, _DimensionExistenceVisitor(provider, context))


def _safe_indexation(
    expr: ExpressionNode, provider: IndexingStructureProvider
) -> Optional[IndexingStructure]:
    try:
        return compute_indexation(expr, provider)
    except ValueError:
        # Unresolved port field (sum_connections); deferred, see
        # _DimensionExistenceVisitor._structure_of.
        return None


def _check_bound_set_consistency(
    bound_expr: ExpressionNode,
    var: Variable,
    provider: IndexingStructureProvider,
    context: str,
) -> None:
    structure = _safe_indexation(bound_expr, provider)
    if structure is None:
        return
    stray = structure.sets - var.structure.sets
    if stray:
        raise ValueError(
            f"{context} of variable '{var.name}' is indexed by set(s) "
            f"{sorted(stray)} not declared in '{var.name}''s own indexed_by."
        )


def _check_local_set_crosses_port(
    definition: PortFieldDefinition,
    local_sets: FrozenSet[str],
    provider: IndexingStructureProvider,
) -> None:
    structure = _safe_indexation(definition.definition, provider)
    if structure is None:
        return
    crossing = structure.sets & local_sets
    if crossing:
        raise ValueError(
            f"Port field definition for '{definition.port_field.port_name}."
            f"{definition.port_field.field_name}' is still indexed by local "
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
        _check_dimension_existence(c.expression, provider, f"constraint '{c.name}'")

    if model.objective_contributions:
        for oid, expr in model.objective_contributions.items():
            _check_dimension_existence(
                expr, provider, f"objective contribution '{oid}'"
            )

    if model.extra_outputs:
        for eo_id, eo_expr in model.extra_outputs.items():
            _check_dimension_existence(eo_expr, provider, f"extra-output '{eo_id}'")

    for var in model.variables.values():
        if var.lower_bound is not None:
            _check_dimension_existence(
                var.lower_bound, provider, f"lower bound of variable '{var.name}'"
            )
            _check_bound_set_consistency(
                var.lower_bound, var, provider, "Lower bound"
            )
        if var.upper_bound is not None:
            _check_dimension_existence(
                var.upper_bound, provider, f"upper bound of variable '{var.name}'"
            )
            _check_bound_set_consistency(
                var.upper_bound, var, provider, "Upper bound"
            )

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
