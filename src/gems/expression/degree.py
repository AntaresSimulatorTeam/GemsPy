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

import math

import gems.expression.scenario_operator
from gems.expression.expression import (
    AllTimeSumNode,
    CeilNode,
    ComponentParameterNode,
    ComponentVariableNode,
    FloorNode,
    MaxNode,
    MinNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
)

from .expression import (
    AdditionNode,
    ComparisonNode,
    DivisionNode,
    ExpressionNode,
    LiteralNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    ScenarioOperatorNode,
    VariableNode,
)
from .visitor import ExpressionVisitor, T, visit


class ExpressionDegreeVisitor(ExpressionVisitor[int | float]):
    """
    Computes degree of expression with respect to variables.
    """

    def literal(self, node: LiteralNode) -> int | float:
        return 0

    def negation(self, node: NegationNode) -> int | float:
        return visit(node.operand, self)

    # TODO: Take into account simplification that can occur with literal coefficient for add, sub, mult, div
    def addition(self, node: AdditionNode) -> int | float:
        degrees = [visit(o, self) for o in node.operands]
        return max(degrees)

    def multiplication(self, node: MultiplicationNode) -> int | float:
        return visit(node.left, self) + visit(node.right, self)

    def division(self, node: DivisionNode) -> int | float:
        right_degree = visit(node.right, self)
        if right_degree != 0:
            raise ValueError("Degree computation not implemented for divisions.")
        return visit(node.left, self)

    def comparison(self, node: ComparisonNode) -> int | float:
        return max(visit(node.left, self), visit(node.right, self))

    def variable(self, node: VariableNode) -> int | float:
        return 1

    def parameter(self, node: ParameterNode) -> int | float:
        return 0

    def comp_variable(self, node: ComponentVariableNode) -> int | float:
        return 1

    def comp_parameter(self, node: ComponentParameterNode) -> int | float:
        return 0

    def time_shift(self, node: TimeShiftNode) -> int | float:
        return visit(node.operand, self)

    def time_eval(self, node: TimeEvalNode) -> int | float:
        return visit(node.operand, self)

    def time_sum(self, node: TimeSumNode) -> int | float:
        return visit(node.operand, self)

    def all_time_sum(self, node: AllTimeSumNode) -> int | float:
        return visit(node.operand, self)

    def scenario_operator(self, node: ScenarioOperatorNode) -> int | float:
        scenario_operator_cls = getattr(gems.expression.scenario_operator, node.name)
        # TODO: Carefully check if this formula is correct
        return scenario_operator_cls.degree() * visit(node.operand, self)

    def port_field(self, node: PortFieldNode) -> int | float:
        return 1

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> int | float:
        return visit(node.operand, self)

    def floor(self, node: FloorNode) -> int | float:
        d = visit(node.operand, self)
        return 0 if d == 0 else math.inf

    def ceil(self, node: CeilNode) -> int | float:
        d = visit(node.operand, self)
        return 0 if d == 0 else math.inf

    def maximum(self, node: MaxNode) -> int | float:
        return 0 if all(visit(op, self) == 0 for op in node.operands) else math.inf

    def minimum(self, node: MinNode) -> int | float:
        return 0 if all(visit(op, self) == 0 for op in node.operands) else math.inf


def compute_degree(expression: ExpressionNode) -> int | float:
    return visit(expression, ExpressionDegreeVisitor())


def is_constant(expr: ExpressionNode) -> bool:
    """
    True if the expression has no variable.
    """
    return compute_degree(expr) == 0


def is_linear(expr: ExpressionNode) -> bool:
    """
    True if the expression is linear with respect to variables.
    """
    return compute_degree(expr) <= 1
