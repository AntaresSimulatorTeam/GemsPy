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

from dataclasses import dataclass, replace
from typing import Any, List

from gems.expression import (
    AdditionNode,
    ComparisonNode,
    DivisionNode,
    ExpressionNode,
    ExpressionVisitor,
    LiteralNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    VariableNode,
)
from gems.expression.expression import (
    AllTimeSumNode,
    BinaryOperatorNode,
    ComponentParameterNode,
    ComponentVariableNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ProblemParameterNode,
    ProblemVariableNode,
    ScenarioOperatorNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
)
from gems.expression.visitor import visit


@dataclass(frozen=True)
class PortField:
    name: str


@dataclass
class PortType:
    """
    Defines a port type.

    A port is an external interface of a model, where other
    ports can be connected.
    Only compatible ports may be connected together (?)
    """

    id: str
    fields: List[PortField]  # TODO: should we rename with "pin" ?


@dataclass(frozen=True)
class PortFieldId:
    port_name: str
    field_name: str

    def replicate(self, /, **changes: Any) -> "PortFieldId":
        return replace(self, **changes)


@dataclass(frozen=True)
class PortFieldDefinition:
    """
    Defines the values of one port field
    """

    port_field: PortFieldId
    definition: ExpressionNode

    def __post_init__(self) -> None:
        _validate_port_field_expression(self)

    def replicate(self, /, **changes: Any) -> "PortFieldDefinition":
        return replace(self, **changes)


def port_field_def(
    port_name: str, field_name: str, definition: ExpressionNode
) -> PortFieldDefinition:
    return PortFieldDefinition(PortFieldId(port_name, field_name), definition)


class _PortFieldExpressionChecker(ExpressionVisitor[None]):
    """
    Visits the whole expression to check there is no:
    comparison, other port field, component-associated parametrs or variables...
    """

    def literal(self, node: LiteralNode) -> None:
        pass

    def negation(self, node: NegationNode) -> None:
        visit(node.operand, self)

    def _visit_binary_op(self, node: BinaryOperatorNode) -> None:
        visit(node.left, self)
        visit(node.right, self)

    def addition(self, node: AdditionNode) -> None:
        for n in node.operands:
            visit(n, self)

    def multiplication(self, node: MultiplicationNode) -> None:
        self._visit_binary_op(node)

    def division(self, node: DivisionNode) -> None:
        self._visit_binary_op(node)

    def comparison(self, node: ComparisonNode) -> None:
        raise ValueError("Port definition cannot contain a comparison operator.")

    def variable(self, node: VariableNode) -> None:
        pass

    def parameter(self, node: ParameterNode) -> None:
        pass

    def comp_parameter(self, node: ComponentParameterNode) -> None:
        raise ValueError(
            "Port definition must not contain a parameter associated to a component."
        )

    def comp_variable(self, node: ComponentVariableNode) -> None:
        raise ValueError(
            "Port definition must not contain a variable associated to a component."
        )

    def pb_parameter(self, node: ProblemParameterNode) -> None:
        raise ValueError(
            "Port definition must not contain a parameter associated to a component."
        )

    def pb_variable(self, node: ProblemVariableNode) -> None:
        raise ValueError(
            "Port definition must not contain a variable associated to a component."
        )

    def time_shift(self, node: TimeShiftNode) -> None:
        visit(node.operand, self)

    def time_eval(self, node: TimeEvalNode) -> None:
        visit(node.operand, self)

    def time_sum(self, node: TimeSumNode) -> None:
        visit(node.operand, self)

    def all_time_sum(self, node: AllTimeSumNode) -> None:
        visit(node.operand, self)

    def scenario_operator(self, node: ScenarioOperatorNode) -> None:
        visit(node.operand, self)

    def port_field(self, node: PortFieldNode) -> None:
        raise ValueError("Port definition cannot reference another port field.")

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> None:
        raise ValueError("Port definition cannot contain port field aggregation.")


def _validate_port_field_expression(definition: PortFieldDefinition) -> None:
    visit(definition.definition, _PortFieldExpressionChecker())
