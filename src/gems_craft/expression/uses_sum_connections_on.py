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

from gems_craft.expression.expression import (
    AbsNode,
    AdditionNode,
    AllTimeSumNode,
    CeilNode,
    ComparisonNode,
    DivisionNode,
    DualNode,
    ExpressionNode,
    FloorNode,
    LiteralNode,
    MaxNode,
    MinNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    PowerNode,
    ReducedCostNode,
    RoundNode,
    ScenarioOperatorNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
    VariableNode,
)

from .visitor import ExpressionVisitor, visit


class UsesSumConnectionsOnVisitor(ExpressionVisitor[bool]):
    """Returns True if the expression contains sum_connections(port_name.field_name)."""

    def __init__(self, port_name: str, field_name: str) -> None:
        self._port_name = port_name
        self._field_name = field_name

    def literal(self, node: LiteralNode) -> bool:
        return False

    def negation(self, node: NegationNode) -> bool:
        return visit(node.operand, self)

    def addition(self, node: AdditionNode) -> bool:
        return any(visit(o, self) for o in node.operands)

    def multiplication(self, node: MultiplicationNode) -> bool:
        return visit(node.left, self) or visit(node.right, self)

    def division(self, node: DivisionNode) -> bool:
        return visit(node.left, self) or visit(node.right, self)

    def power(self, node: PowerNode) -> bool:
        return visit(node.left, self) or visit(node.right, self)

    def comparison(self, node: ComparisonNode) -> bool:
        return visit(node.left, self) or visit(node.right, self)

    def variable(self, node: VariableNode) -> bool:
        return False

    def parameter(self, node: ParameterNode) -> bool:
        return False

    def time_shift(self, node: TimeShiftNode) -> bool:
        return visit(node.operand, self)

    def time_eval(self, node: TimeEvalNode) -> bool:
        return visit(node.operand, self)

    def time_sum(self, node: TimeSumNode) -> bool:
        return visit(node.operand, self)

    def all_time_sum(self, node: AllTimeSumNode) -> bool:
        return visit(node.operand, self)

    def scenario_operator(self, node: ScenarioOperatorNode) -> bool:
        return visit(node.operand, self)

    def port_field(self, node: PortFieldNode) -> bool:
        return False

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> bool:
        if (
            isinstance(node.operand, PortFieldNode)
            and node.operand.port_name == self._port_name
            and node.operand.field_name == self._field_name
        ):
            return True
        return visit(node.operand, self)

    def floor(self, node: FloorNode) -> bool:
        return visit(node.operand, self)

    def ceil(self, node: CeilNode) -> bool:
        return visit(node.operand, self)

    def abs(self, node: AbsNode) -> bool:
        return visit(node.operand, self)

    def round(self, node: RoundNode) -> bool:
        return visit(node.operand, self)

    def maximum(self, node: MaxNode) -> bool:
        return any(visit(o, self) for o in node.operands)

    def minimum(self, node: MinNode) -> bool:
        return any(visit(o, self) for o in node.operands)

    def dual(self, node: DualNode) -> bool:
        return False

    def reduced_cost(self, node: ReducedCostNode) -> bool:
        return False


def uses_sum_connections_on(
    expr: ExpressionNode, port_name: str, field_name: str
) -> bool:
    """Return True if expr contains sum_connections(port_name.field_name)."""
    return visit(expr, UsesSumConnectionsOnVisitor(port_name, field_name))
